"""Tests for src/synth_checks.py (S07)."""

import pandas as pd

from src import synth_checks as S


def test_keyword_label_picks_most_hits_and_defaults_to_no_distortion():
    pats = S._patterns(S.KEYWORDS)
    assert S.keyword_label("I should have known. I must work harder; I ought to.", pats) == "should_statements"
    assert S.keyword_label("The supplier was late, so I booked another one.", pats) == "no_distortion"
    assert S.keyword_label("It’s my fault, all because of me.", pats) == "personalization"   # curly apostrophe


def test_word_boundaries_and_stems():
    pats = S._patterns(S.KEYWORDS)
    assert S.keyword_hits("The shoulder hurt", pats)["should_statements"] == 0
    post = S._patterns(S.merge_lists(S.KEYWORDS, S.POST_HOC))
    assert S.keyword_hits("I fixated on it, ignoring the rest", post)["mental_filter"] == 2


def test_macro_scores_perfect_and_chance_lower():
    y = ["no_distortion", "labeling", "mind_reading", "labeling"] * 10
    assert S.macro_scores(y, y)["macro_f1"] > 0
    assert S.chance_scores(y, n=20)["macro_f1"] < S.macro_scores(y, y)["macro_f1"]


def test_overlaps_flags_only_above_threshold():
    syn = [{"id": "syn_1", "text": "we lost the client and the deal"},
           {"id": "syn_2", "text": "a completely different sentence here"}]
    sets = {"test": [("t1", "we lost the client and the deal today")]}
    ov = S.overlaps(syn, sets, threshold=0.6)
    assert list(ov.syn_id) == ["syn_1"] and ov.jaccard.iloc[0] > 0.6
    assert S.overlaps(syn, {"test": [("t2", "nothing alike")]}).empty


def test_load_survey_keeps_long_answers_only(tmp_path):
    p = tmp_path / "s.csv"
    pd.DataFrame({"Timestamp": ["1", "2"], "Gender": ["F", "M"],
                  "Q1": ["one two three four five six seven eight nine ten eleven", "too short to keep here ok"],
                  }).to_csv(p, index=False)
    out = S.load_survey(p)
    assert len(out) == 1 and out[0][0] == "survey_r2_q3"
    x = tmp_path / "s.xlsx"
    pd.read_csv(p, dtype=str).to_excel(x, index=False)
    assert S.load_survey(x) == out
