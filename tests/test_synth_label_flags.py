"""Tests for src/synth_label_flags.py (S09) and the --input option of src/cluster_label_errors.py."""

import json

import openpyxl
import pandas as pd

from src import cluster_label_errors as C
from src import synth_label_flags as F
from src.data import DISTORTIONS

THR = {d: 0.5 for d in DISTORTIONS}


def _probs(**kw):
    p = {d: 0.05 for d in DISTORTIONS}
    p.update(kw)
    return p


def test_close_vote_flag():
    assert F.close_vote_flag({"dominant_distortion": "Labeling", "critic_votes": {"Labeling": 2}})
    assert not F.close_vote_flag({"dominant_distortion": "Labeling", "critic_votes": {"Labeling": 3}})
    assert F.close_vote_flag({"dominant_distortion": "No Distortion", "no_distortion_votes": 2})
    assert not F.close_vote_flag({"dominant_distortion": "No Distortion", "no_distortion_votes": 3})


def test_model_flag_rules():
    nd = {"dominant_distortion": "No Distortion"}
    assert F.model_flag(nd, 0.99, _probs(), THR, 0.977)[0]
    assert not F.model_flag(nd, 0.90, _probs(), THR, 0.977)[0]
    lab = {"dominant_distortion": "Labeling", "secondary_distortion": "Mind Reading"}
    assert F.model_flag(lab, 0.05, _probs(labeling=0.9), THR, 0.977)[0]           # confidently not distorted
    assert F.model_flag(lab, 0.9, _probs(fortune_telling=0.8), THR, 0.977)[0]      # other type, own type low
    assert not F.model_flag(lab, 0.9, _probs(mind_reading=0.8), THR, 0.977)[0]     # top = panel's secondary
    assert not F.model_flag(lab, 0.9, _probs(fortune_telling=0.8, labeling=0.6), THR, 0.977)[0]  # own type passes
    assert not F.model_flag(lab, 0.9, _probs(fortune_telling=0.4), THR, 0.977)[0]  # not confident


def test_neighbour_flag_cut():
    assert F.neighbour_flag(0.8) and not F.neighbour_flag(0.7)


def test_review_sheet_has_only_blind_columns(tmp_path):
    items = pd.DataFrame({"review_id": ["R001", "R002"], "text": ["one text here", "another text"]})
    p = tmp_path / "s.xlsx"
    F.write_review_sheet(p, items)
    ws = openpyxl.load_workbook(p)["Label"]
    headers = [c.value for c in ws[1]]
    assert headers[:2] == ["Review ID", "Text to read"]
    assert not any(h and ("route" in h.lower() or "panel" in h.lower() or "flag" in h.lower()) for h in headers)
    assert [ws.cell(row=r, column=1).value for r in (2, 3)] == ["R001", "R002"]
    assert all(ws.cell(row=2, column=c).value is None for c in range(3, 8))      # label columns empty


def test_cluster_input_maps_synthetic_schema(tmp_path):
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps({"id": f"syn_{i}", "text": f"text {i}", "dominant_distortion": lab})
                           for i, lab in enumerate(["No Distortion", "Labeling"])), encoding="utf-8")
    df = C.load_input(str(p))
    assert list(df[C.LABEL_COL]) == ["No Distortion", "Labeling"]
    assert list(df[C.ID_COL]) == ["syn_0", "syn_1"] and (df[C.SPAN_COL] == "").all()
