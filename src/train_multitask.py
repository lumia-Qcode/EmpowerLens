"""
D20 - multi-task model: one shared encoder, three heads.

RUN (one seed; repeat for 42, 1337, 2024; D12 recipe):
    python -m src.train_multitask --model mental/mental-roberta-base --seed 42 --lr 3e-5 \
        --epochs 5 --weight-decay 0.1 --max-length 512 --batch-size 16
SMOKE (100 train / 100 val / 100 test rows, 1 epoch; writes nothing to results/):
    python -m src.train_multitask --smoke --max-length 128

Heads (all read the same encoder pass):
  binary      2-way linear head on the <s> state; class-weighted cross-entropy.
  multilabel  10-way linear head on the <s> state; BCE with pos_weight = neg/pos (train).
              Per-type thresholds fitted on val (best val F1 per type, same 0.05-0.95 grid
              as src.train_transformer.sweep_thresholds).
  selection   one score per sentence: token states are mean-pooled over each sentence's
              character span (tokenizer offset mapping), then a small MLP. BCE with
              pos_weight = neg/pos over the labelled train sentences.

Selection masking (test case B-06): a sentence contributes to the selection loss only if
its row has an annotated span AND the sentence kept at least one token after truncation.
Rows with no distortion and the 13 rows whose quote could not be located
(results/results_sentence_label/excluded_rows.csv) get label -1 on every sentence; their
contribution to the loss, and their gradient into the selection head, is exactly 0.
The loss is the mean over the unmasked sentences of the batch (0 if there are none).

Total loss = w_binary * L_bin + w_multilabel * L_ml + w_selection * L_sel.

Selection rules (val only; test.csv is opened once, after training, via
src.evaluate.load_split, and nothing is chosen on it):
  epoch       the epoch with the highest mean of three val scores: binary macro-F1 (argmax),
              multilabel macro-F1 (at that epoch's val-swept thresholds) and selector
              sentence F1 (at that epoch's val-best threshold).
  binary thr  lowest threshold on P(distorted) with val FPR <= --target-fpr (default 0.15),
              src.fit_binary_threshold.threshold_for_fpr.
  ml thr      per type, best val F1 (sweep_thresholds).
  sel thr     max val sentence F1 (src.train_span_selector.best_threshold).

Selector scores use the T12 sentence labels (results/results_sentence_label/) over the
distorted rows with a located span, the same rows and sentences D03 scored. A sentence
truncated away by --max-length gets score 0 (it can never be selected); the count is in
the eval file.

Outputs
  <out>/multitask_<model>_<seed>[_smoke]/  model.pt (all weights), encoder config,
                                           tokenizer, meta.json  (checkpoints/ is gitignored)
  <results-dir>/eval_multitask_<seed>.json  val + test scores, thresholds, settings
                                           (smoke: written into the checkpoint dir instead)
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows torch CPU OpenMP crash

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn

from src.data import DISTORTIONS
from src.fit_binary_threshold import prob_distorted, threshold_for_fpr
from src.span_check import split_sentences
from src.train_span_selector import best_threshold, score_part
from src.train_transformer import (
    ML_COLS,
    TEXT_COL,
    _special_token_ids,
    class_weights,
    pos_weights,
    resolve_device,
    sweep_thresholds,
)

ID_COL = "Id_Number"
LABELS_DIR = Path("results/results_sentence_label")
DEFAULT_RESULTS = Path("results/results_multitask")
IGNORE = -1   # sentence label meaning "no supervision for this sentence"


# ----------------------------------------------------------------------------- data

def load_frame(splits_dir: str, part: str) -> pd.DataFrame:
    """train/val via the same reader as train_transformer; test only through src.evaluate."""
    if part == "test":
        from src.evaluate import load_split   # the project's single test.csv reader
        return load_split(splits_dir, "test")
    return pd.read_csv(f"{splits_dir}/{part}.csv", encoding="utf-8-sig")


def load_sentence_labels(part: str, labels_dir: Path = LABELS_DIR) -> pd.DataFrame:
    df = pd.read_csv(labels_dir / f"sentence_labels_{part}.csv", encoding="utf-8")
    return df.rename(columns={"id": ID_COL})


def load_unlocated_ids(labels_dir: Path = LABELS_DIR) -> set:
    ex = pd.read_csv(labels_dir / "excluded_rows.csv", encoding="utf-8")
    return set(ex.loc[ex["reason"] == "span_unlocated", "id"].astype(int))


def sentence_targets(df: pd.DataFrame, sent_df: pd.DataFrame, unlocated: set):
    """Per row: (list of (start, end) char spans, list of labels in {0, 1, IGNORE}, reason).

    reason is "labelled", "no_distortion" or "span_unlocated". Unmasked rows take their
    sentences and offsets straight from the T12 file; masked rows are split with the same
    T11 splitter, so the model sees the same sentence structure, but every label is IGNORE.
    Fails loudly on any row that does not fit one of the three cases.
    """
    by_id = {rid: g.sort_values("sent_idx") for rid, g in sent_df.groupby(ID_COL)}
    spans, labels, reasons = [], [], []
    for rid, text, y in zip(df[ID_COL].astype(int), df[TEXT_COL].astype(str), df["y_bin"].astype(int)):
        if rid in by_id:
            if y != 1:
                raise ValueError(f"row {rid} has sentence labels but y_bin={y}")
            g = by_id[rid]
            sp = list(zip(g["sent_start"].astype(int), g["sent_end"].astype(int)))
            for a, b in sp:
                if not 0 <= a < b <= len(text):
                    raise ValueError(f"row {rid}: sentence offsets ({a}, {b}) outside the text")
            spans.append(sp)
            labels.append(g["label"].astype(int).tolist())
            reasons.append("labelled")
        elif y == 0:
            sp = split_sentences(text)
            spans.append(sp); labels.append([IGNORE] * len(sp)); reasons.append("no_distortion")
        elif rid in unlocated:
            sp = split_sentences(text)
            spans.append(sp); labels.append([IGNORE] * len(sp)); reasons.append("span_unlocated")
        else:
            raise ValueError(f"distorted row {rid} has no sentence labels and is not in excluded_rows.csv")
    return spans, labels, reasons


def token_sentence_ids(offsets, spans):
    """Sentence index for each token (IGNORE if it overlaps no sentence).

    A token belongs to the first sentence whose [start, end) character span it overlaps.
    """
    out = []
    for s, e in offsets:
        k = IGNORE
        for j, (a, b) in enumerate(spans):
            if s < b and max(e, s + 1) > a:
                k = j
                break
        out.append(k)
    return out


def encode_rows(texts, spans, tokenizer, max_length: int):
    """Head truncation like src.train_transformer.encode_texts, plus a token->sentence map.

    Special tokens are added by hand (transformers 5.x has no build_inputs_with_special_tokens)
    and map to IGNORE. Returns (encodings, truncation_rate).
    """
    if not getattr(tokenizer, "is_fast", False):
        raise SystemExit("train_multitask needs a fast tokenizer (offset mapping).")
    prefix, suffix = _special_token_ids(tokenizer)
    budget = max_length - len(prefix) - len(suffix)
    enc = tokenizer(list(texts), add_special_tokens=False, return_offsets_mapping=True)
    out, truncated = [], 0
    for ids, offs, sp in zip(enc["input_ids"], enc["offset_mapping"], spans):
        if len(ids) > budget:
            truncated += 1
            ids, offs = ids[:budget], offs[:budget]
        tsent = [IGNORE] * len(prefix) + token_sentence_ids(offs, sp) + [IGNORE] * len(suffix)
        input_ids = prefix + list(ids) + suffix
        out.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "token_sent": tsent})
    return out, truncated / max(len(out), 1)


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, y_bin, y_ml, sent_labels):
        self.encodings, self.y_bin, self.y_ml, self.sent_labels = encodings, y_bin, y_ml, sent_labels

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, i):
        return {**self.encodings[i], "y_bin": int(self.y_bin[i]), "y_ml": self.y_ml[i],
                "sent_labels": self.sent_labels[i]}


def collate(batch, pad_id):
    T = max(len(b["input_ids"]) for b in batch)
    S = max(max(len(b["sent_labels"]) for b in batch), 1)
    ids, att, tsent, slab = [], [], [], []
    for b in batch:
        p = T - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * p)
        att.append(b["attention_mask"] + [0] * p)
        tsent.append(b["token_sent"] + [IGNORE] * p)
        slab.append(list(b["sent_labels"]) + [IGNORE] * (S - len(b["sent_labels"])))
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(att, dtype=torch.long),
        "token_sent": torch.tensor(tsent, dtype=torch.long),
        "sent_labels": torch.tensor(slab, dtype=torch.float),
        "y_bin": torch.tensor([b["y_bin"] for b in batch], dtype=torch.long),
        "y_ml": torch.tensor(np.array([b["y_ml"] for b in batch]), dtype=torch.float),
    }


# ----------------------------------------------------------------------------- model

class MultiTaskModel(nn.Module):
    """Shared encoder + binary, multilabel and sentence-selection heads."""

    def __init__(self, encoder, dropout: float = 0.1, n_types: int = len(DISTORTIONS)):
        super().__init__()
        self.encoder = encoder
        h = encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.binary_head = nn.Linear(h, 2)
        self.multilabel_head = nn.Linear(h, n_types)
        self.selection_head = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Dropout(dropout), nn.Linear(h, 1))

    def forward(self, input_ids, attention_mask, token_sent, n_sent: int | None = None, **_):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.dropout(hidden[:, 0])
        S = n_sent if n_sent is not None else max(int(token_sent.max().item()) + 1, 1)
        # member[b, s, t] = 1 if token t of row b belongs to sentence s
        member = (token_sent.unsqueeze(1) == torch.arange(S, device=token_sent.device).view(1, S, 1)).to(hidden.dtype)
        n_tok = member.sum(-1)
        sent_repr = member @ hidden / n_tok.clamp(min=1).unsqueeze(-1)
        return {
            "binary": self.binary_head(pooled),
            "multilabel": self.multilabel_head(pooled),
            "selection": self.selection_head(self.dropout(sent_repr)).squeeze(-1),
            "sent_has_tokens": n_tok > 0,
        }


def selection_loss(logits, labels, has_tokens, pos_weight=None):
    """Masked sentence BCE: mean over sentences with a 0/1 label that kept >= 1 token.

    Masked entries are removed with torch.where, so their loss term AND their gradient are
    exactly zero. With no unmasked sentence in the batch the loss is exactly 0.
    """
    logits = logits.float()
    mask = (labels >= 0) & has_tokens
    per = F.binary_cross_entropy_with_logits(logits, labels.clamp(min=0), pos_weight=pos_weight, reduction="none")
    per = torch.where(mask, per, torch.zeros_like(per))
    return per.sum() / mask.sum().clamp(min=1)


def compute_loss(out, batch, weights, cw_bin=None, pw_ml=None, pw_sel=None):
    parts = {
        "binary": F.cross_entropy(out["binary"].float(), batch["y_bin"], weight=cw_bin),
        "multilabel": F.binary_cross_entropy_with_logits(out["multilabel"].float(), batch["y_ml"], pos_weight=pw_ml),
        "selection": selection_loss(out["selection"], batch["sent_labels"], out["sent_has_tokens"], pw_sel),
    }
    total = sum(weights[k] * v for k, v in parts.items())
    return total, {k: float(v.detach()) for k, v in parts.items()}


def build_model(name_or_config, dropout=0.1):
    from transformers import AutoModel, PretrainedConfig
    if isinstance(name_or_config, PretrainedConfig):
        encoder = AutoModel.from_config(name_or_config, add_pooling_layer=False)
    else:
        encoder = AutoModel.from_pretrained(name_or_config, add_pooling_layer=False)
    return MultiTaskModel(encoder.float(), dropout=dropout)


# ----------------------------------------------------------------------------- predict / score

@torch.no_grad()
def predict(model, ds, pad_id, device, batch_size):
    """Binary logits (N,2), multilabel logits (N,10), per-row sentence probs (NaN if truncated)."""
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         collate_fn=lambda b: collate(b, pad_id))
    b_out, m_out, s_out = [], [], []
    for batch in loader:
        S = batch["sent_labels"].shape[1]
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
            out = model(batch["input_ids"], batch["attention_mask"], batch["token_sent"], n_sent=S)
        b_out.append(out["binary"].float().cpu().numpy())
        m_out.append(out["multilabel"].float().cpu().numpy())
        p = torch.sigmoid(out["selection"].float())
        p = torch.where(out["sent_has_tokens"], p, torch.full_like(p, float("nan"))).cpu().numpy()
        s_out.extend(list(p))
    n_sent = [len(x) for x in ds.sent_labels]
    return np.concatenate(b_out), np.concatenate(m_out), [s[:n] for s, n in zip(s_out, n_sent)]


def selector_frame(df, sent_labels, reasons, sent_probs):
    """Sentence table over the labelled rows only (the D03 / D10 scoring population)."""
    rows = []
    for rid, labs, why, probs in zip(df[ID_COL].astype(int), sent_labels, reasons, sent_probs):
        if why != "labelled":
            continue
        for i, (lab, p) in enumerate(zip(labs, probs)):
            rows.append({ID_COL: rid, "sent_idx": i, "label": int(lab),
                         "truncated": bool(np.isnan(p)), "score": 0.0 if np.isnan(p) else float(p)})
    return pd.DataFrame(rows, columns=[ID_COL, "sent_idx", "label", "truncated", "score"])


def epoch_scores(y_bin, y_ml, sel_df, bin_logits, ml_logits):
    bin_f1 = f1_score(y_bin, bin_logits.argmax(1), average="macro", zero_division=0)
    probs = 1 / (1 + np.exp(-ml_logits))
    ml_thr = sweep_thresholds(probs, y_ml)
    ml_f1 = f1_score(y_ml, (probs >= np.array(ml_thr)).astype(int), average="macro", zero_division=0)
    if len(sel_df):
        sel_thr = best_threshold(sel_df["label"].values, sel_df["score"].values)
        sel_f1 = score_part(sel_df, sel_df["score"].values, sel_thr)["f1"]
    else:
        sel_thr, sel_f1 = 0.5, 0.0
    return {"binary_macro_f1": float(bin_f1), "multilabel_macro_f1": float(ml_f1), "selector_f1": float(sel_f1),
            "criterion": float((bin_f1 + ml_f1 + sel_f1) / 3), "ml_thresholds": ml_thr, "sel_threshold": float(sel_thr)}


def score_split(y_bin, y_ml, sel_df, reasons, bin_logits, ml_logits, thr):
    p = prob_distorted(bin_logits)
    pred = (p >= thr["binary"]).astype(int)
    neg, pos = y_bin == 0, y_bin == 1
    tn, fp = int((neg & (pred == 0)).sum()), int((neg & (pred == 1)).sum())
    tp = int((pos & (pred == 1)).sum())
    spec = tn / max(int(neg.sum()), 1)
    binary = {
        "specificity": spec, "fpr": 1 - spec, "sensitivity": tp / max(int(pos.sum()), 1),
        "macro_f1": float(f1_score(y_bin, pred, average="macro", zero_division=0)),
        "positive_class_f1": float(f1_score(y_bin, pred, zero_division=0)),
        "macro_f1_argmax": float(f1_score(y_bin, bin_logits.argmax(1), average="macro", zero_division=0)),
        "tn": tn, "fp": fp, "n_non_distorted": int(neg.sum()), "n_distorted": int(pos.sum()),
    }
    probs = 1 / (1 + np.exp(-ml_logits))
    ml_pred = (probs >= np.array(thr["multilabel"])).astype(int)
    multilabel = {
        "macro_f1": float(f1_score(y_ml, ml_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_ml, ml_pred, average="micro", zero_division=0)),
        # distorted rows only: comparable in population with the D17 cascade-input table
        "macro_f1_distorted_rows": float(f1_score(y_ml[pos], ml_pred[pos], average="macro", zero_division=0)),
        "per_type_f1": {d: float(f) for d, f in zip(
            DISTORTIONS, f1_score(y_ml, ml_pred, average=None, zero_division=0))},
        "n_rows": int(len(y_ml)),
    }
    sel = score_part(sel_df, sel_df["score"].values, thr["selector"])
    sel["n_sentences_truncated"] = int(sel_df["truncated"].sum())
    counts = pd.Series(reasons).value_counts().to_dict()
    return {"n_rows": int(len(y_bin)), "selection_rows_by_reason": {k: int(v) for k, v in counts.items()},
            "binary": binary, "multilabel": multilabel, "selector": sel}


# ----------------------------------------------------------------------------- main

def prepare(df, part, tokenizer, max_length, unlocated, labels_dir=LABELS_DIR):
    sent_df = load_sentence_labels(part, labels_dir)
    sent_df = sent_df[sent_df[ID_COL].isin(set(df[ID_COL].astype(int)))]
    spans, labels, reasons = sentence_targets(df, sent_df, unlocated)
    enc, trunc = encode_rows(df[TEXT_COL].astype(str), spans, tokenizer, max_length)
    y_bin = df["y_bin"].to_numpy().astype(np.int64)
    y_ml = df[ML_COLS].to_numpy().astype(np.float32)
    return MultiTaskDataset(enc, y_bin, list(y_ml), labels), reasons, trunc


def main(argv=None):
    ap = argparse.ArgumentParser(description="D20 multi-task model (binary + multilabel + sentence selection).")
    ap.add_argument("--model", default="mental/mental-roberta-base")
    ap.add_argument("--tag", default=None, help="recorded model name when --model is a local path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--w-binary", type=float, default=1.0, help="weight of the binary loss")
    ap.add_argument("--w-multilabel", type=float, default=1.0, help="weight of the multilabel loss")
    ap.add_argument("--w-selection", type=float, default=1.0, help="weight of the selection loss")
    ap.add_argument("--target-fpr", type=float, default=0.15, help="binary threshold: val FPR <= this")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--labels-dir", default=str(LABELS_DIR))
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true", help="100 rows per split, 1 epoch")
    a = ap.parse_args(argv)

    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    device = resolve_device(a.device)
    labels_dir = Path(a.labels_dir)
    unlocated = load_unlocated_ids(labels_dir)
    weights = {"binary": a.w_binary, "multilabel": a.w_multilabel, "selection": a.w_selection}

    train_df, val_df = load_frame(a.splits, "train"), load_frame(a.splits, "val")
    if a.smoke:
        train_df, val_df, a.epochs = train_df.head(100).reset_index(drop=True), val_df.head(100).reset_index(drop=True), 1

    tokenizer = AutoTokenizer.from_pretrained(a.model)
    model = build_model(a.model, a.dropout).to(device)
    pad_id = tokenizer.pad_token_id

    tr_ds, tr_reasons, _ = prepare(train_df, "train", tokenizer, a.max_length, unlocated, labels_dir)
    va_ds, va_reasons, va_trunc = prepare(val_df, "val", tokenizer, a.max_length, unlocated, labels_dir)
    print(f"[multitask] device={device} model={a.model} train={len(tr_ds)} val={len(va_ds)} epochs={a.epochs} "
          f"weights={weights}")
    print(f"  train selection rows: {pd.Series(tr_reasons).value_counts().to_dict()}")

    y_tr_bin = np.asarray(tr_ds.y_bin)
    y_tr_ml = np.array(tr_ds.y_ml)
    cw_bin = class_weights(y_tr_bin, 2, device)
    pw_ml = pos_weights(y_tr_ml.copy(), device)
    tr_sent = np.array([l for labs in tr_ds.sent_labels for l in labs if l >= 0])
    pw_sel = torch.tensor((len(tr_sent) - tr_sent.sum()) / max(tr_sent.sum(), 1), dtype=torch.float, device=device)

    y_va_bin = np.asarray(va_ds.y_bin)
    y_va_ml = np.array(va_ds.y_ml).astype(int)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = (len(tr_ds) + a.batch_size - 1) // a.batch_size
    total_steps = a.epochs * steps_per_epoch
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    # Length-sorted chunks, shuffled per epoch (seeded): cuts padding on skewed lengths.
    lengths = np.array([len(e["input_ids"]) for e in tr_ds.encodings])
    history, best = [], None
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        rng = np.random.RandomState(a.seed + ep)
        order = rng.permutation(len(tr_ds))
        chunks = [order[i:i + a.batch_size * 50] for i in range(0, len(order), a.batch_size * 50)]
        batches = []
        for c in chunks:
            c = c[np.argsort(lengths[c], kind="stable")]
            batches += [c[i:i + a.batch_size] for i in range(0, len(c), a.batch_size)]
        rng.shuffle(batches)
        running = {k: 0.0 for k in weights}
        for idx in batches:
            batch = collate([tr_ds[int(i)] for i in idx], pad_id)
            S = batch["sent_labels"].shape[1]
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                out = model(batch["input_ids"], batch["attention_mask"], batch["token_sent"], n_sent=S)
            loss, parts = compute_loss(out, batch, weights, cw_bin, pw_ml, pw_sel)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
            for k, v in parts.items():
                running[k] += v
        b, m, s = predict(model, va_ds, pad_id, device, a.batch_size * 2)
        sc = epoch_scores(y_va_bin, y_va_ml, selector_frame(val_df, va_ds.sent_labels, va_reasons, s), b, m)
        row = {"epoch": ep, **{f"train_loss_{k}": v / max(len(batches), 1) for k, v in running.items()},
               **{f"val_{k}": v for k, v in sc.items() if k not in ("ml_thresholds", "sel_threshold")}}
        history.append(row)
        print(f"epoch {ep}: " + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if k != "epoch")
              + f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
        if best is None or sc["criterion"] > best["criterion"]:
            best = {"criterion": sc["criterion"], "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

    model.load_state_dict(best["state"])
    model.to(device)

    # ---- thresholds: val only
    b, m, s = predict(model, va_ds, pad_id, device, a.batch_size * 2)
    va_sel = selector_frame(val_df, va_ds.sent_labels, va_reasons, s)
    bin_thr, bin_fpr, bin_sens = threshold_for_fpr(y_va_bin, prob_distorted(b), a.target_fpr)
    thr = {
        "binary": float(bin_thr),
        "multilabel": sweep_thresholds(1 / (1 + np.exp(-m)), y_va_ml),
        "selector": float(best_threshold(va_sel["label"].values, va_sel["score"].values)) if len(va_sel) else 0.5,
    }
    val_res = score_split(y_va_bin, y_va_ml, va_sel, va_reasons, b, m, thr)

    # ---- test: read once, scored once with the val thresholds; nothing is chosen here
    test_df = load_frame(a.splits, "test")
    if a.smoke:
        test_df = test_df.head(100).reset_index(drop=True)
    te_ds, te_reasons, te_trunc = prepare(test_df, "test", tokenizer, a.max_length, unlocated, labels_dir)
    b, m, s = predict(model, te_ds, pad_id, device, a.batch_size * 2)
    te_sel = selector_frame(test_df, te_ds.sent_labels, te_reasons, s)
    test_res = score_split(np.asarray(te_ds.y_bin), np.array(te_ds.y_ml).astype(int), te_sel, te_reasons, b, m, thr)

    # ---- save
    model_id = a.tag or a.model
    run_name = f"multitask_{model_id.split('/')[-1]}_{a.seed}" + ("_smoke" if a.smoke else "")
    out_dir = Path(a.out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], out_dir / "model.pt")
    model.encoder.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    pd.DataFrame(history).to_csv(out_dir / "epoch_history.csv", index=False, encoding="utf-8")

    settings = {
        "task": "multitask", "model": model_id, "init_from": a.model, "seed": a.seed, "lr": a.lr,
        "epochs": a.epochs, "weight_decay": a.weight_decay, "max_length": a.max_length, "truncation": "head",
        "batch_size": a.batch_size, "dropout": a.dropout, "loss_weights": weights, "device": device,
        "smoke": a.smoke, "splits": a.splits, "labels_dir": str(labels_dir),
        "losses": {"binary": "cross-entropy, class weights n/(2*count) from train",
                   "multilabel": "BCE, pos_weight neg/pos from train",
                   "selection": "BCE, pos_weight neg/pos over labelled train sentences; masked to 0 "
                                "for no-distortion rows, the 13 unlocated rows, and truncated sentences"},
        "selection_pos_weight": float(pw_sel),
        "epoch_criterion": "max over epochs of mean(val binary macro-F1 at argmax, val multilabel macro-F1 "
                           "at val-swept thresholds, val selector sentence F1 at val-best threshold)",
        "best_epoch": best["epoch"], "best_val_criterion": best["criterion"],
        "val_truncation_rate": va_trunc, "test_truncation_rate": te_trunc,
    }
    thresholds = {
        **thr,
        "binary_criterion": {"rule": "lowest threshold on P(distorted) with val FPR <= target",
                             "target_fpr": a.target_fpr, "val_fpr": round(bin_fpr, 4),
                             "val_sensitivity": round(bin_sens, 4)},
        "multilabel_criterion": "per type, best val F1 on the 0.05-0.95 grid",
        "selector_criterion": "max val sentence F1",
    }
    meta = {**settings, "thresholds": thresholds, "epoch_history": history, "val": val_res}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    result = {"settings": settings, "thresholds": thresholds, "epoch_history": history,
              "splits": {"val": val_res, "test": test_res}}
    res_dir = out_dir if a.smoke else Path(a.results_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    res_path = res_dir / (f"eval_multitask_{a.seed}" + ("_smoke" if a.smoke else "") + ".json")
    res_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for part, r in (("val", val_res), ("test", test_res)):
        print(f"{part}: spec {r['binary']['specificity']:.3f} fpr {r['binary']['fpr']:.3f} "
              f"sens {r['binary']['sensitivity']:.3f} | ml macro-F1 {r['multilabel']['macro_f1']:.3f} "
              f"micro {r['multilabel']['micro_f1']:.3f} | sel F1 {r['selector']['f1']:.3f} "
              f"hit {r['selector']['row_hit_rate']:.3f}")
    print(f"best epoch {best['epoch']} of {a.epochs}; checkpoint {out_dir}; eval {res_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
