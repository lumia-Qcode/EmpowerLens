"""Tests for src/data.py — the measured, non-negotiable dataset facts."""

import numpy as np
import pandas as pd
import pytest

from src.data import (
    DISTORTIONS,
    LABEL_CANON,
    MC_CLASSES,
    load_raw,
    make_targets,
)

CSV = "Annotated_data.csv"


@pytest.fixture(scope="module")
def loaded():
    df = load_raw(CSV)
    y_bin, y_mc, y_ml = make_targets(df)
    return df, y_bin, y_mc, y_ml


def test_row_count(loaded):
    df, *_ = loaded
    assert len(df) == 2530, "expected 2,530 rows to survive load_raw's filters"


def test_every_raw_label_maps():
    # Read raw (bypassing load_raw's own validation) and confirm every
    # non-null dominant/secondary string is covered by LABEL_CANON.
    raw = pd.read_csv(CSV, encoding="utf-8-sig")
    seen = set(raw["Dominant Distortion"].dropna().map(lambda s: str(s).strip()))
    seen |= set(
        raw["Secondary Distortion (Optional)"].dropna().map(lambda s: str(s).strip())
    )
    unmapped = seen - set(LABEL_CANON)
    assert not unmapped, f"unmapped label strings: {sorted(unmapped)}"


def test_y_bin_distorted_count(loaded):
    _, y_bin, _, _ = loaded
    assert int(y_bin.sum()) == 1597


def test_y_mc_no_distortion_count(loaded):
    _, _, y_mc, _ = loaded
    assert int((y_mc == 0).sum()) == 933


def test_y_mc_is_11_class_no_nulls(loaded):
    df, _, y_mc, _ = loaded
    assert len(MC_CLASSES) == 11
    assert not np.isnan(y_mc.astype(float)).any()
    assert set(np.unique(y_mc)) == set(range(11)), "all 11 classes must appear"
    assert len(y_mc) == len(df), "no rows may be dropped building y_mc"


def test_secondary_contribution_count(loaded):
    _, _, _, y_ml = loaded
    # No secondary duplicates its dominant and no No-Distortion row has a
    # secondary, so exactly the rows with a secondary have two positives.
    assert int((y_ml.sum(axis=1) >= 2).sum()) == 416


def test_y_bin_recoverable_from_y_ml(loaded):
    _, y_bin, _, y_ml = loaded
    assert np.array_equal(y_bin, (y_ml.any(axis=1)).astype(int))


def test_y_mc_matches_y_ml_for_distorted(loaded):
    _, y_bin, y_mc, y_ml = loaded
    # For distorted rows, the dominant class index (y_mc-1) is set in y_ml.
    for i in np.where(y_bin == 1)[0]:
        assert y_ml[i, y_mc[i] - 1] == 1.0


def test_shapes(loaded):
    df, y_bin, y_mc, y_ml = loaded
    assert y_bin.shape == (len(df),)
    assert y_mc.shape == (len(df),)
    assert y_ml.shape == (len(df), 10)
    assert len(DISTORTIONS) == 10


def test_unmapped_label_raises():
    from src.data import _canon

    with pytest.raises(ValueError):
        _canon("Catastrophizing")  # not a real label in this dataset


# --- split-level checks (require data/splits/ to have been generated) --------

SPLIT_DIR = "data/splits"
SPLIT_FILES = {n: f"{SPLIT_DIR}/{n}.csv" for n in ("train", "val", "test")}
_have_splits = all(__import__("os").path.exists(p) for p in SPLIT_FILES.values())
requires_splits = pytest.mark.skipif(
    not _have_splits, reason="run `python -m src.make_splits` first"
)


@pytest.fixture(scope="module")
def split_frames():
    return {n: pd.read_csv(p, encoding="utf-8-sig") for n, p in SPLIT_FILES.items()}


@requires_splits
def test_splits_disjoint_by_id(split_frames):
    ids = {n: set(f["Id_Number"]) for n, f in split_frames.items()}
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["val"].isdisjoint(ids["test"])


@requires_splits
def test_splits_cover_all_rows(split_frames):
    total = sum(len(f) for f in split_frames.values())
    assert total == 2530


@requires_splits
def test_all_11_classes_in_every_split(split_frames):
    for name, f in split_frames.items():
        present = set(f["y_mc"].unique())
        assert present == set(range(11)), f"{name} missing classes: {set(range(11)) - present}"
