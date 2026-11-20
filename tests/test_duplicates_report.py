"""Tests for src/duplicates_report.py (C02 leakage bounds)."""

import numpy as np
import pandas as pd
import pytest

from src.duplicates_report import (
    binary_counts_bounds,
    binary_metrics,
    f1,
    macro_f1_from_counts,
    multilabel_counts,
    multilabel_macro_f1_bounds,
)


def test_f1_zero_division():
    assert f1(0, 0, 0) == 0.0
    assert f1(2, 1, 1) == pytest.approx(4 / 6)


def test_binary_bounds_cover_the_four_cases():
    tn, fp, tp, fn = 72, 20, 98, 63
    b = binary_counts_bounds(tn, fp, tp, fn)
    # removed ND row was a TN -> 71/91; was an FP -> 72/91
    assert b["specificity"] == pytest.approx((71 / 91, 72 / 91))
    assert b["fpr"] == pytest.approx((19 / 91, 20 / 91))
    # removed distorted row was a TP -> 97/160; an FN -> 98/160
    assert b["sensitivity"] == pytest.approx((97 / 160, 98 / 160))


def test_binary_bounds_skip_impossible_cases():
    # fp = 0, so the removed ND row must have been a TN.
    b = binary_counts_bounds(tn=10, fp=0, tp=5, fn=5)
    assert b["specificity"] == (1.0, 1.0)


def test_binary_macro_f1_matches_sklearn():
    from sklearn.metrics import f1_score
    y = [0] * 5 + [1] * 7
    p = [0, 0, 0, 1, 1] + [1, 1, 1, 1, 0, 0, 0]
    assert binary_metrics(3, 2, 4, 3)["macro_f1"] == pytest.approx(f1_score(y, p, average="macro"))


def _random_case(seed, k=5, n=30, cap=2):
    rng = np.random.default_rng(seed)
    Y = rng.integers(0, 2, (n, k))
    Y[0] = 0                          # the removed ND row
    Y[1] = 0; Y[1, 0] = 1             # the removed distorted row
    P = np.zeros((n, k), dtype=int)
    for i in range(n):                # at most `cap` fired types per row
        P[i, rng.choice(k, size=rng.integers(0, cap + 1), replace=False)] = 1
    tp = (Y & P).sum(0); fp = ((1 - Y) & P).sum(0); fn = (Y & (1 - P)).sum(0)
    return Y, P, tp, fp, fn


@pytest.mark.parametrize("seed", range(20))
def test_multilabel_bounds_contain_true_value(seed):
    from sklearn.metrics import f1_score
    Y, P, tp, fp, fn = _random_case(seed)
    actual = f1_score(Y[2:], P[2:], average="macro", zero_division=0)
    lo, hi = multilabel_macro_f1_bounds(tp, fp, fn, Y[1])
    clo, chi = multilabel_macro_f1_bounds(tp, fp, fn, Y[1], max_fired=2)
    assert lo - 1e-12 <= clo <= actual + 1e-12
    assert actual - 1e-12 <= chi <= hi + 1e-12


def test_multilabel_single_type_hand_case():
    # one type: tp=1, fp=1, fn=0; the distorted row is positive for it.
    # It must have been the TP (fn=0), so after removal tp=0 -> F1 = 0,
    # unless the ND row was the FP: then tp=0, fp=0 -> F1 = 0 too.
    assert multilabel_macro_f1_bounds([1], [1], [0], [1]) == (0.0, 0.0)
    # tp=2, fp=1, fn=0: remove TP -> (1,1,0)=2/3 or (1,0,0)=1
    assert multilabel_macro_f1_bounds([2], [1], [0], [1]) == pytest.approx((2 / 3, 1.0))


def test_multilabel_counts_roundtrip():
    tp = np.array([9, 4]); fp = np.array([16, 23]); fn = np.array([8, 24])
    pc = pd.DataFrame({
        "class": ["a", "b"],
        "precision": tp / (tp + fp),
        "recall": tp / (tp + fn),
        "support": tp + fn,
    })
    rtp, rfp, rfn = multilabel_counts(pc)
    assert rtp.tolist() == tp.tolist() and rfp.tolist() == fp.tolist() and rfn.tolist() == fn.tolist()
    assert macro_f1_from_counts(rtp, rfp, rfn) == pytest.approx((f1(9, 16, 8) + f1(4, 23, 24)) / 2)


def test_real_inputs_reproduce_saved_test_scores():
    """load_inputs asserts that counts-derived macro-F1s equal the eval JSONs."""
    from src.duplicates_report import load_inputs
    per_seed, true_304 = load_inputs()
    assert len(per_seed) == 3
    assert true_304[6] == 1   # fortune_telling
    assert per_seed["specificity_253"].mean() == pytest.approx(0.768, abs=5e-4)
