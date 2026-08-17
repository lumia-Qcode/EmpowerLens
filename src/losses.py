"""
Optional loss functions and optimizer builders for src/train_transformer.py.

FocalLoss: alternative to plain weighted BCE for multilabel Stage 2, where a
few classes (all_or_nothing, mental_filter, personalization) have very few
positive examples relative to the rest. Focal loss down-weights the gradient
from easy, already-confident negatives so more signal reaches the hard /
minority positives, on top of whatever pos_weight class balancing already
does in src/train_transformer.py's pos_weights().

freeze_bottom_layers: coarse regularization for a backbone that's overfitting
a small train set (DeBERTa-v3-base showed the largest val->test gap of the
three backbones evaluated so far) — freeze the lower, more general encoder
layers and only backprop through the top layers + head.

build_llrd_optimizer: layer-wise learning-rate decay (LLRD). Assigns a
smaller learning rate to lower encoder layers and a larger rate to upper
layers + the classification head, instead of one flat --lr for the whole
model. A softer alternative to freeze_bottom_layers — every layer still
moves, just by different amounts.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def masked_mean(per_element: torch.Tensor, keep: Optional[torch.Tensor]) -> torch.Tensor:
    """Mean over elements, ignoring label columns where ``keep`` is 0.

    WHY MASKING AND NOT pos_weight=0
    --------------------------------
    The obvious way to make a label "not count" is to zero its pos_weight. That
    does NOT work. ``pos_weight`` scales only the POSITIVE term of BCE. For a
    column whose targets are all 0 — which is exactly the case this exists for —
    the entire loss is the negative term ``-log(1 - p)``, which pos_weight never
    touches. The model would still be trained to drive that logit to 0.

    Measured consequence of getting this wrong: in the 2026-08-17 sequential run,
    stage A trained on 7,846 PatternReframe rows with ml_emotional_reasoning = 0 in
    every one. That is not missing supervision, it is 7,846 assertions that the
    label never occurs, and stage B could not undo it — emotional_reasoning F1 fell
    from 0.368 (Annotated-only baseline) to 0.074, which alone accounted for ~72%
    of the sequential run's macro_f1 deficit.

    Dividing by the kept count rather than the full element count keeps the loss on
    the same scale as an unmasked run, so learning rates stay comparable.
    """
    if keep is None:
        return per_element.mean()
    per_element = per_element * keep                     # broadcasts over the batch
    denom = per_element.shape[0] * keep.sum()
    return per_element.sum() / denom.clamp(min=1.0)


class FocalLoss(nn.Module):
    """Multilabel focal loss layered on top of BCEWithLogitsLoss's pos_weight."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 label_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.label_mask = label_mask

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p_t = torch.exp(-bce)
        focal = ((1 - p_t) ** self.gamma) * bce
        return masked_mean(focal, self.label_mask)


class MaskedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss that ignores chosen label columns entirely.

    Used when a training set carries no information about a label — see
    ``masked_mean`` for why zeroing pos_weight is not equivalent.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None,
                 label_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.pos_weight = pos_weight
        self.label_mask = label_mask

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        return masked_mean(bce, self.label_mask)


def _find_encoder_layers(model):
    """
    Locate the list of transformer blocks regardless of backbone naming.
    HF's *ForSequenceClassification wrappers all expose the backbone at
    model.base_model (RoBERTa/MentalRoBERTa: .encoder.layer;
    DeBERTa-v3: .encoder.layer as well, just a different module class
    underneath) — walking base_model.encoder.layer covers every backbone
    used in this project. Raises loudly instead of silently freezing/
    decaying an empty list if a future backbone doesn't match.
    """
    base = model.base_model
    obj = base
    for part in ("encoder", "layer"):
        obj = getattr(obj, part, None)
        if obj is None:
            break
    if obj is not None:
        return list(obj)
    raise AttributeError(
        f"Could not find encoder.layer on {type(base).__name__} — inspect "
        f"the model's module tree (e.g. print(model)) and adjust "
        f"_find_encoder_layers in src/losses.py for this backbone."
    )


def freeze_bottom_layers(model, n_layers: int) -> int:
    """Freeze the bottom n_layers transformer blocks. Returns total layer count."""
    layers = _find_encoder_layers(model)
    for layer in layers[:n_layers]:
        for p in layer.parameters():
            p.requires_grad = False
    return len(layers)


def _find_head(model):
    for name in ("classifier", "classification_head", "score"):
        head = getattr(model, name, None)
        if head is not None:
            return head
    raise AttributeError(
        "Could not find a classification head attribute (tried classifier / "
        "classification_head / score) — inspect the model and adjust "
        "_find_head in src/losses.py for this backbone."
    )


def build_llrd_optimizer(
    model,
    base_lr: float = 3e-5,
    decay: float = 0.9,
    head_lr: Optional[float] = None,
    weight_decay: float = 0.01,
) -> torch.optim.Optimizer:
    """
    AdamW with a per-layer learning rate: the classification head gets
    head_lr (defaults to base_lr), each encoder layer going downward from
    the top gets base_lr * decay**depth, and everything else (word/position
    embeddings, DeBERTa's ContextPooler, any top-level LayerNorm not owned
    by a numbered layer) is swept into a final group at the lowest rate in
    the decay chain.

    That last group matters more than it looks: an earlier version of this
    function only grouped encoder layers + the head, which silently
    excluded embeddings/pooler from every param group — since AdamW only
    updates params it's explicitly given, those modules would never receive
    a gradient step at all. Every trainable parameter in the model is
    accounted for here (checked by ID, not by name, so nothing can be
    double-counted or dropped by a naming mismatch across backbones).
    """
    layers = _find_encoder_layers(model)
    head = _find_head(model)
    head_lr = head_lr if head_lr is not None else base_lr

    groups = [{"params": list(head.parameters()), "lr": head_lr, "weight_decay": weight_decay}]
    claimed_ids = {id(p) for p in head.parameters()}

    lr = base_lr
    lowest_lr = base_lr
    for layer in reversed(layers):
        params = list(layer.parameters())
        groups.append({"params": params, "lr": lr, "weight_decay": weight_decay})
        claimed_ids.update(id(p) for p in params)
        lowest_lr = lr
        lr *= decay

    remainder = [p for p in model.parameters() if p.requires_grad and id(p) not in claimed_ids]
    if remainder:
        groups.append({"params": remainder, "lr": lowest_lr, "weight_decay": weight_decay})

    return torch.optim.AdamW(groups)
