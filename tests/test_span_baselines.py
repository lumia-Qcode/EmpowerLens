"""Unit tests for the D10 rule-based sentence pickers in src/span_baselines.py."""
import numpy as np
import pandas as pd

from src.span_baselines import pick_first, pick_last, pick_random, pick_top1

# three rows with 1, 3 and 2 sentences, deliberately not in Id order
DF = pd.DataFrame({"Id_Number": [7, 7, 7, 3, 9, 9],
                   "sent_idx": [0, 1, 2, 0, 0, 1],
                   "label": [0, 0, 1, 1, 1, 0]})


def _one_per_row(sel):
    return (DF.assign(s=sel).groupby("Id_Number")["s"].sum() == 1).all()


def test_first_picks_sentence_zero():
    sel = pick_first(DF)
    assert _one_per_row(sel)
    assert list(sel) == [1, 0, 0, 1, 1, 0]


def test_last_handles_uneven_rows():
    sel = pick_last(DF)
    assert _one_per_row(sel)
    assert list(sel) == [0, 0, 1, 1, 0, 1]


def test_random_one_per_row_and_reproducible():
    a, b = pick_random(DF, seed=42), pick_random(DF, seed=42)
    assert _one_per_row(a)
    assert np.array_equal(a, b)


def test_top1_picks_highest_score():
    score = np.array([0.1, 0.9, 0.5, 0.2, 0.3, 0.4])
    assert list(pick_top1(DF, score)) == [0, 1, 0, 1, 0, 1]
