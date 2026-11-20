"""Tests for src/train_multitask.py (D20).

B-06: a reflection with no annotated span (no distortion, or one of the 13 rows whose
quote could not be located) adds nothing to the span-selection loss and sends no gradient
into the selection head. Everything runs on a tiny random RoBERTa (no download).
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import RobertaConfig

from src.train_multitask import (
    IGNORE,
    ID_COL,
    build_model,
    collate,
    compute_loss,
    selection_loss,
    sentence_targets,
    token_sentence_ids,
)

N_TYPES = 10


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = RobertaConfig(vocab_size=99, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=37, max_position_embeddings=64, pad_token_id=1)
    return build_model(cfg, dropout=0.0)


def item(n_tok, n_sent, sent_labels, y_bin, seed):
    """One encoded row: <s> + tokens split evenly over n_sent sentences + </s>."""
    rng = np.random.RandomState(seed)
    ids = [0] + rng.randint(5, 99, size=n_tok).tolist() + [2]
    per = max(n_tok // n_sent, 1)
    ts = [IGNORE] + [min(i // per, n_sent - 1) for i in range(n_tok)] + [IGNORE]
    y_ml = np.zeros(N_TYPES, dtype=np.float32)
    if y_bin:
        y_ml[seed % N_TYPES] = 1.0
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "token_sent": ts,
            "y_bin": y_bin, "y_ml": y_ml, "sent_labels": sent_labels}


LABELLED = item(12, 3, [0, 1, 0], 1, seed=1)
NO_DISTORTION = item(10, 2, [IGNORE, IGNORE], 0, seed=2)
UNLOCATED = item(9, 3, [IGNORE, IGNORE, IGNORE], 1, seed=3)
WEIGHTS = {"binary": 1.0, "multilabel": 1.0, "selection": 1.0}


def forward(model, rows):
    batch = collate(rows, pad_id=1)
    out = model(batch["input_ids"], batch["attention_mask"], batch["token_sent"],
                n_sent=batch["sent_labels"].shape[1])
    return batch, out


def sel_head_grads(model):
    return [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
            for p in model.selection_head.parameters()]


# ------------------------------------------------------------------ B-06

@pytest.mark.parametrize("masked_row", [NO_DISTORTION, UNLOCATED], ids=["no_distortion", "span_unlocated"])
def test_b06_masked_row_alone_gives_zero_selection_loss_and_zero_head_gradient(masked_row):
    model = tiny_model()
    model.train()
    batch, out = forward(model, [masked_row])
    loss = selection_loss(out["selection"], batch["sent_labels"], out["sent_has_tokens"],
                          pos_weight=torch.tensor(5.0))
    assert loss.item() == 0.0
    loss.backward()
    for g in sel_head_grads(model):
        assert torch.count_nonzero(g) == 0


@pytest.mark.parametrize("masked_row", [NO_DISTORTION, UNLOCATED], ids=["no_distortion", "span_unlocated"])
def test_b06_masked_row_adds_nothing_to_a_mixed_batch(masked_row):
    """Loss and selection-head gradient are identical with or without the masked row."""
    model = tiny_model()
    model.eval()   # dropout off so the two passes are comparable

    batch, out = forward(model, [LABELLED])
    alone = selection_loss(out["selection"], batch["sent_labels"], out["sent_has_tokens"])
    alone.backward()
    g_alone = sel_head_grads(model)
    model.zero_grad()

    batch, out = forward(model, [LABELLED, masked_row])
    out["selection"].retain_grad()
    mixed = selection_loss(out["selection"], batch["sent_labels"], out["sent_has_tokens"])
    mixed.backward()
    g_mixed = sel_head_grads(model)

    assert torch.allclose(alone, mixed, atol=1e-6)
    for a, b in zip(g_alone, g_mixed):
        assert torch.allclose(a, b, atol=1e-6)
    # the masked row's sentence logits receive exactly zero gradient
    assert torch.count_nonzero(out["selection"].grad[1]) == 0
    assert torch.count_nonzero(out["selection"].grad[0]) > 0


def test_b06_truncated_sentence_is_masked_even_when_labelled():
    logits = torch.tensor([[0.3, -1.2, 2.0]], requires_grad=True)
    labels = torch.tensor([[0.0, 1.0, 1.0]])
    has_tok = torch.tensor([[True, True, False]])   # sentence 3 cut off by max length
    loss = selection_loss(logits, labels, has_tok)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(logits[0, :2], labels[0, :2])
    assert torch.allclose(loss, ref)
    loss.backward()
    assert logits.grad[0, 2] == 0


# ------------------------------------------------------------------ shapes + one step

def test_forward_shapes():
    model = tiny_model()
    batch, out = forward(model, [LABELLED, NO_DISTORTION, UNLOCATED])
    assert out["binary"].shape == (3, 2)
    assert out["multilabel"].shape == (3, N_TYPES)
    assert out["selection"].shape == (3, 3)
    # NO_DISTORTION has 2 sentences, so its padded third slot has no tokens
    assert out["sent_has_tokens"].tolist() == [[True, True, True], [True, True, False], [True, True, True]]


def test_one_optimisation_step_changes_all_heads_and_is_finite():
    model = tiny_model()
    model.train()
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch, out = forward(model, [LABELLED, NO_DISTORTION, UNLOCATED])
    loss, parts = compute_loss(out, batch, WEIGHTS, pw_sel=torch.tensor(3.0))
    assert torch.isfinite(loss) and set(parts) == {"binary", "multilabel", "selection"}
    assert parts["selection"] > 0   # the labelled row supervises the head
    loss.backward()
    opt.step()
    for head in ("binary_head", "multilabel_head", "selection_head", "encoder.embeddings"):
        assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n.startswith(head)), head


# ------------------------------------------------------------------ data plumbing

def test_token_sentence_ids_uses_char_overlap():
    spans = [(0, 10), (11, 20)]
    offsets = [(0, 4), (4, 10), (11, 15), (15, 20), (20, 21)]
    assert token_sentence_ids(offsets, spans) == [0, 0, 1, 1, IGNORE]


def test_sentence_targets_masks_unlabelled_rows_and_fails_on_unknown():
    df = pd.DataFrame({ID_COL: [1, 2, 3], "Patient Question": ["I fail. Always.", "Fine day. Nice.", "Bad. Worse."],
                       "y_bin": [1, 0, 1]})
    sent = pd.DataFrame({ID_COL: [1, 1], "sent_idx": [0, 1], "sent_start": [0, 8], "sent_end": [7, 15],
                         "label": [0, 1]})
    spans, labels, reasons = sentence_targets(df, sent, unlocated={3})
    assert reasons == ["labelled", "no_distortion", "span_unlocated"]
    assert labels[0] == [0, 1]
    assert all(l == IGNORE for l in labels[1] + labels[2]) and len(labels[1]) == 2
    with pytest.raises(ValueError):
        sentence_targets(df, sent, unlocated=set())   # row 3 distorted, unlabelled, not excluded
