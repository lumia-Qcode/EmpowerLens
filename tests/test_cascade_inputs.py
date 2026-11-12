"""Unit tests for the D17 input builders in src/make_splits_cascade_inputs.py."""
import numpy as np
import pandas as pd

from src.make_splits_cascade_inputs import crop, mark, selector_pick

DOC = "I failed. Everyone hates me. It is raining."
SENTS = pd.DataFrame({"sentence": ["I failed.", "Everyone hates me.", "It is raining."],
                      "sent_start": [0, 10, 29], "sent_end": [9, 28, 43]})


def test_offsets_fixture_is_consistent():
    assert all(DOC[s:e] == t for s, e, t in SENTS[["sent_start", "sent_end", "sentence"]].to_numpy())


def test_crop_keeps_text_order():
    assert crop(SENTS, np.array([True, False, True])) == "I failed. It is raining."


def test_mark_wraps_each_picked_sentence():
    assert mark(DOC, SENTS, np.array([False, True, False])) == "I failed. @ Everyone hates me. @ It is raining."
    assert mark(DOC, SENTS, np.array([True, False, True])) == "@ I failed. @ Everyone hates me. @ It is raining. @"


def test_mark_merges_consecutive_sentences_into_one_span():
    assert mark(DOC, SENTS, np.array([True, True, False])) == "@ I failed. Everyone hates me. @ It is raining."


def test_selector_pick_uses_selected_then_falls_back_to_top1():
    s = SENTS.assign(selected=["no", "yes", "no"], score=[0.9, 0.6, 0.1])
    pick, fb = selector_pick(s)
    assert list(pick) == [False, True, False] and not fb
    s = SENTS.assign(selected=["no", "no", "no"], score=[0.2, 0.7, 0.1])
    pick, fb = selector_pick(s)
    assert list(pick) == [False, True, False] and fb
