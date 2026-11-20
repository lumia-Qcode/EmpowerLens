"""Tests for src/synth_label_check.py (S10 scoring)."""

import pandas as pd
import pytest

from src import synth_label_check as C

KEY = pd.DataFrame({
    "review_id": ["R1", "R2", "R3", "R4"],
    "id": ["syn_1", "syn_2", "syn_3", "syn_4"],
    "panel_label": ["Labeling", "No Distortion", "Mind Reading", "Fortune-telling"],
    "random": ["no", "no", "yes", "no"],
    "flag_neighbours": ["yes", "yes", "no", "no"],
    "flag_model": ["yes", "no", "no", "yes"],
    "flag_close_votes": ["no", "no", "no", "yes"],
})


def test_precision_and_cp_equivalence():
    labels = pd.DataFrame({"review_id": ["R1", "R2", "R3", "R4"],
                           "dominant": ["Mind Reading", "No Distortion (calibrated pessimism)", "Mind Reading",
                                        "Fortune-telling"]})
    t, m = C.score(labels, KEY)
    t = t.set_index("check")
    assert list(m.wrong) == [True, False, False, False]          # CP option == No Distortion
    assert t.loc["neighbours", "rows_flagged"] == 2 and t.loc["neighbours", "precision"] == 0.5
    assert t.loc["model", "precision"] == 0.5 and t.loc["close votes", "precision"] == 0.0
    assert t.loc["random (baseline)", "rows_flagged"] == 1 and t.loc["random (baseline)", "precision"] == 0.0


def test_unknown_label_and_missing_rows_fail_loudly():
    with pytest.raises(ValueError):
        C.canon("Catastrophizing")
    with pytest.raises(SystemExit):
        C.score(pd.DataFrame({"review_id": ["R1"], "dominant": ["Labeling"]}), KEY)


def test_read_labels_refuses_unfinished_sheet(tmp_path):
    p = tmp_path / "l.csv"
    pd.DataFrame({"review_id": ["R1", "R2"], "dominant": ["Labeling", ""]}).to_csv(p, index=False)
    with pytest.raises(SystemExit):
        C.read_labels(p)
