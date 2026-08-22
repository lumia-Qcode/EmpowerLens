"""Tests for src/losses.py and the class-imbalance ablation in
src/tutorial_distilbert.py.

The load-bearing claim is the first test: routing the ``--loss bce`` arm through
MaskedBCEWithLogitsLoss must give exactly what Hugging Face applies internally
for ``problem_type="multi_label_classification"``. If that drifts, the ablation
stops being a controlled comparison — the "tutorial baseline" row would no
longer be the tutorial's loss.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data import DISTORTIONS
from src.losses import AsymmetricLoss, FocalLoss, MaskedBCEWithLogitsLoss
from src.tutorial_distilbert import build_loss, pos_weights


@pytest.fixture
def batch():
    torch.manual_seed(0)
    logits = torch.randn(16, len(DISTORTIONS))
    targets = (torch.rand(16, len(DISTORTIONS)) < 0.08).float()  # ~8% positive
    return logits, targets


def test_bce_arm_matches_huggingface_builtin(batch):
    logits, targets = batch
    ours = MaskedBCEWithLogitsLoss()(logits, targets)
    hf = nn.BCEWithLogitsLoss()(logits, targets)
    assert torch.allclose(ours, hf), "the bce ablation arm is not the tutorial's loss"


def test_pos_weights_are_negatives_over_positives():
    y = np.zeros((100, len(DISTORTIONS)), dtype=np.float32)
    y[:5, 0] = 1.0            # 5 positives, 95 negatives
    y[:50, 1] = 1.0           # balanced
    w = pos_weights(y.copy())
    assert w[0] == pytest.approx(95 / 5)
    assert w[1] == pytest.approx(50 / 50)


def test_pos_weights_survive_an_absent_label():
    """A column with zero positives must not divide by zero."""
    y = np.zeros((10, len(DISTORTIONS)), dtype=np.float32)
    w = pos_weights(y)
    assert np.isfinite(w).all()


def test_pos_weight_raises_the_cost_of_missing_a_positive(batch):
    """The whole point: with pos_weight, missing a positive costs more."""
    logits, targets = batch
    pw = torch.full((len(DISTORTIONS),), 10.0)
    plain = MaskedBCEWithLogitsLoss()(logits, targets)
    weighted = MaskedBCEWithLogitsLoss(pos_weight=pw)(logits, targets)
    assert weighted > plain


def test_every_loss_arm_is_finite_and_differentiable(batch):
    logits, targets = batch
    y_train = targets.numpy()
    for name in ("bce", "pos_bce", "focal", "asl"):
        loss_fn, desc = build_loss(name, y_train.copy(), "cpu", focal_gamma=2.0)
        x = logits.clone().requires_grad_(True)
        loss = loss_fn(x, targets)
        assert torch.isfinite(loss), f"{name} produced {loss}"
        loss.backward()
        assert torch.isfinite(x.grad).all(), f"{name} produced non-finite gradients"
        assert desc, f"{name} has no description for the run log"


def test_unknown_loss_name_raises():
    y = np.zeros((4, len(DISTORTIONS)), dtype=np.float32)
    with pytest.raises(ValueError):
        build_loss("nope", y, "cpu", focal_gamma=2.0)


def test_asl_damps_easy_negatives_more_than_plain_bce():
    """An easy negative (very low probability) should contribute far less
    under AsymmetricLoss than under BCE — that is the mechanism."""
    easy_neg = torch.tensor([[-6.0]])
    target = torch.tensor([[0.0]])
    bce = MaskedBCEWithLogitsLoss()(easy_neg, target)
    asl = AsymmetricLoss()(easy_neg, target)
    assert asl < bce


def test_focal_reduces_to_bce_at_gamma_zero(batch):
    logits, targets = batch
    focal = FocalLoss(gamma=0.0)(logits, targets)
    bce = MaskedBCEWithLogitsLoss()(logits, targets)
    assert torch.allclose(focal, bce)


def test_sigmoid_not_softmax_in_the_multilabel_path():
    """Ten independent probabilities, so they must NOT sum to 1."""
    from src.tutorial_distilbert import sigmoid

    logits = np.array([[2.0] * len(DISTORTIONS)])
    probs = sigmoid(logits)
    assert probs.sum() > 1.0, "labels are competing — that would be softmax"
    assert ((probs >= 0) & (probs <= 1)).all()
