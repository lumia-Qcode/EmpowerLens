"""Unit tests for the D08 grouping rule in reannotation_2026-09-27/scripts/label_changes.py."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "reannotation_2026-09-27" / "scripts" / "label_changes.py"
spec = importlib.util.spec_from_file_location("label_changes", SCRIPT)
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)

ND, CP = lc.NO_DISTORTION, lc.CP_LABEL


def test_unchanged_returns_none():
    assert lc.classify("Labeling", None, ["Labeling"]) == ("Labeling", None)
    # original ND, annotator picks calibrated pessimism: still "not distorted", no change
    assert lc.classify(ND, None, [CP]) == (ND, None)


def test_distortion_to_calibrated_pessimism_is_redefinition():
    assert lc.classify("Magnification", None, [CP]) == (ND, "redefinition")


def test_cp_vs_plain_nd_tie_is_redefinition():
    assert lc.classify("Magnification", None, [CP, ND, "Labeling"]) == (ND, "redefinition")


def test_distortion_to_plain_nd_is_genuine_error():
    assert lc.classify("Magnification", None, [ND]) == (ND, "genuine error")
    assert lc.classify("Magnification", None, [ND, ND, CP]) == (ND, "genuine error")


def test_nd_to_distortion_is_genuine_error():
    assert lc.classify(ND, None, ["Labeling"]) == ("Labeling", "genuine error")


def test_type_to_original_secondary_is_primary_swap():
    assert lc.classify("Labeling", "Mind Reading", ["Mind Reading"]) == ("Mind Reading", "primary swap")


def test_type_to_other_type_is_genuine_error():
    assert lc.classify("Labeling", "Mind Reading", ["Should statements"]) == ("Should statements", "genuine error")


def test_no_majority():
    assert lc.classify("Labeling", None, ["Mind Reading", "Labeling"]) == (None, None)
    assert lc.classify("Labeling", None, []) == (None, None)
