"""Unit tests for the D11 list-scoring helpers in src/cluster_label_errors.py (stage 3)."""
import numpy as np

from src.cluster_label_errors import average_precision, model_triage_score, rank_order


def test_ap_perfect_ranking_is_one():
    assert average_precision(np.array([0.9, 0.8, 0.1, 0.0]), np.array([1, 1, 0, 0])) == 1.0


def test_ap_worst_ranking():
    # errors at ranks 3 and 4: precisions 1/3 and 2/4 -> AP = (1/3 + 1/2) / 2
    ap = average_precision(np.array([0.9, 0.8, 0.1, 0.0]), np.array([0, 0, 1, 1]))
    assert abs(ap - (1 / 3 + 1 / 2) / 2) < 1e-12


def test_ties_broken_by_tiebreak():
    scores = np.array([0.5, 0.5, 0.5])
    assert list(rank_order(scores, np.array([0.1, 0.9, 0.5]))) == [1, 2, 0]


def test_model_triage_order_bucket_then_self_confidence():
    bucket = ["C", "B", "A", "A", "B"]
    conf = [0.05, 0.40, 0.70, 0.20, 0.30]
    order = list(rank_order(model_triage_score(bucket, conf)))
    # A (0.20, 0.70), then B (0.30, 0.40), then C even though its confidence is lowest
    assert order == [3, 2, 4, 1, 0]
