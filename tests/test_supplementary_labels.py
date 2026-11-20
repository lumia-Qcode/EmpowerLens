"""Tests for src/supplementary_labels.py (S03a / S03b labelling tooling).

Toy data only: no real labels exist yet, and these tests never invent any for the
real file.
"""

import pandas as pd
import pytest

from src.data import LABEL_CANON
from src.supplementary_labels import (
    ANNOTATORS, CP_LABEL, DOMINANT_CHOICES, SECONDARY_CHOICES, agreement, assign, merge,
    read_workbook, validate_labels, write_workbook,
)


def _supp(n=100, per_thread=4):
    return pd.DataFrame({
        "item_id": [f"sup_{i:03d}" for i in range(1, n + 1)],
        "text": [f"text number {i} about my startup" for i in range(1, n + 1)],
        "source": ["hackernews"] * n,
        "thread_id": [f"t{i // per_thread}" for i in range(n)],
        "collected_at": ["2026-11-06"] * n,
        "role": ["seed" if i % 4 == 0 else "evaluation" for i in range(n)],
    })


def test_choices_map_through_label_canon():
    for lab in DOMINANT_CHOICES:
        assert lab == CP_LABEL or lab in LABEL_CANON
    assert set(SECONDARY_CHOICES) == set(LABEL_CANON) - {"No Distortion"}


def test_assign_covers_everything_once():
    df = _supp()
    a = assign(df, n_shared=10, seed=42)
    shared, own = set(a["shared"]), [set(a[n]) for n in ANNOTATORS]
    assert len(shared) == 10
    assert not (own[0] & own[1]) and not (shared & (own[0] | own[1]))
    assert shared | own[0] | own[1] == set(df.item_id)
    assert abs(len(own[0]) - len(own[1])) <= 4  # within one thread
    assert set(df.loc[df.item_id.isin(shared), "role"]) == {"seed", "evaluation"}


def test_assign_keeps_threads_together():
    df = _supp()
    a = assign(df, n_shared=10, seed=42)
    who = {i: n for n in ANNOTATORS for i in a[n]}
    rest = df[~df.item_id.isin(a["shared"])]
    for _, g in rest.groupby("thread_id"):
        assert len({who[i] for i in g.item_id}) == 1


def test_assign_is_deterministic():
    df = _supp()
    assert assign(df, 10, 42) == assign(df, 10, 42)
    assert assign(df, 10, 42) != assign(df, 10, 7)


def _lab(item_id, dominant, secondary="", part="", unsure=False):
    return {"item_id": item_id, "dominant": dominant, "secondary": secondary,
            "distorted_part": part, "unsure": unsure, "notes": ""}


def test_validate_rejects_bad_rows():
    texts = {"a": "I always fail.", "b": "x", "c": "y", "d": "z"}
    labels = pd.DataFrame([
        _lab("a", "Overgeneralization", "Overgeneralization", "I always fail."),  # secondary == dominant
        _lab("b", "No Distortion", "Labeling"),                                     # secondary on no-distortion
        _lab("c", "Sadness"),                                                       # not a label
    ])
    errors, _ = validate_labels(labels, ["a", "b", "c", "d"], texts)
    joined = " ".join(errors)
    assert "a" in joined and "b" in joined and "c" in joined
    assert "d" in joined  # assigned but missing


def test_validate_warns_on_span_not_in_text():
    texts = {"a": "I always fail at everything."}
    labels = pd.DataFrame([_lab("a", "Overgeneralization", part="I never win")])
    errors, warnings = validate_labels(labels, ["a"], texts)
    assert not errors and warnings


def test_agreement_metrics():
    ids = ["1", "2", "3", "4"]
    l = pd.DataFrame([_lab("1", "Labeling", part="x"), _lab("2", "No Distortion"),
                      _lab("3", CP_LABEL), _lab("4", "Mind Reading", part="x")])
    i = pd.DataFrame([_lab("1", "Labeling", part="x"), _lab("2", CP_LABEL),
                      _lab("3", CP_LABEL), _lab("4", "Fortune-telling", part="x")])
    res = agreement(l, i, ids)
    assert res["type"]["raw"] == pytest.approx(0.5)
    assert res["type_cp_folded"]["raw"] == pytest.approx(0.75)
    assert res["binary"]["raw"] == pytest.approx(1.0)   # CP counts as not distorted
    assert res["binary"]["kappa"] == pytest.approx(1.0)
    assert set(res["disagreements"].item_id) == {"2", "4"}


def _setup_merge():
    df = _supp(n=8, per_thread=2)
    shared = ["sup_001", "sup_002"]
    lumia = pd.DataFrame([_lab("sup_001", "Labeling", part="text"), _lab("sup_002", "No Distortion")]
                         + [_lab(f"sup_00{k}", "No Distortion") for k in (3, 4, 5)])
    izza = pd.DataFrame([_lab("sup_001", "Labeling", part="text"), _lab("sup_002", "Magnification", part="text")]
                        + [_lab(f"sup_00{k}", CP_LABEL) for k in (6, 7, 8)])
    return df, shared, lumia, izza


def test_merge_requires_adjudication():
    df, shared, lumia, izza = _setup_merge()
    with pytest.raises(ValueError, match="sup_002"):
        merge(df, lumia, izza, shared, adjudication=None)


def test_merge_with_adjudication():
    df, shared, lumia, izza = _setup_merge()
    adj = pd.DataFrame([{"item_id": "sup_002", "final_dominant": "Magnification",
                         "final_secondary": "", "final_distorted_part": "text", "note": "agreed"}])
    out = merge(df, lumia, izza, shared, adjudication=adj)
    assert len(out) == 8 and out.item_id.is_unique
    r = out.set_index("item_id")
    assert r.loc["sup_002", "dominant"] == "Magnification" and r.loc["sup_002", "y_bin"] == 1
    assert r.loc["sup_001", "labelled_by"] == "both"
    assert r.loc["sup_006", "dominant"] == "No Distortion"
    assert bool(r.loc["sup_006", "calibrated_pessimism"]) and r.loc["sup_006", "y_bin"] == 0
    assert (out.role.values == df.role.values).all()


def test_workbook_roundtrip(tmp_path):
    df = _supp(n=6, per_thread=2)
    p = tmp_path / "labels_lumia.xlsx"
    write_workbook(p, df, shared_ids={"sup_001"}, annotator="lumia")
    back = read_workbook(p)
    assert list(back.item_id) == list(df.item_id)
    assert (back.dominant == "").all() and not back.unsure.any()


def test_workbook_reads_filled_and_sorted_rows(tmp_path):
    from openpyxl import load_workbook
    from src.supplementary_labels import SHEET_COLS
    df = _supp(n=4, per_thread=2)
    p = tmp_path / "labels_izza.xlsx"
    write_workbook(p, df, shared_ids={"sup_001"}, annotator="izza")
    wb = load_workbook(p)
    assert wb.sheetnames == ["Start here", "Label", "Label guide"]
    ws = wb["Label"]
    c = {n: SHEET_COLS.index(n) + 1 for n in SHEET_COLS}
    assert ws.cell(2, c["block"]).value == "Shared" and ws.cell(3, c["block"]).value == "Yours"
    # annotator sorts rows (reversed) and fills two of them
    vals = [[ws.cell(r, k).value for k in range(1, len(SHEET_COLS) + 1)] for r in range(2, 6)][::-1]
    for r, v in enumerate(vals, start=2):
        for k, x in enumerate(v, start=1):
            ws.cell(r, k).value = x
    ws.cell(2, c["dominant"]).value = "Labeling"
    ws.cell(2, c["distorted_part"]).value = "text"
    ws.cell(2, c["unsure"]).value = "Yes"
    wb.save(p)
    back = read_workbook(p).set_index("item_id")
    assert back.loc["sup_004", "dominant"] == "Labeling" and back.loc["sup_004", "unsure"]
    assert back.loc["sup_001", "dominant"] == "" and not back.loc["sup_001", "unsure"]
