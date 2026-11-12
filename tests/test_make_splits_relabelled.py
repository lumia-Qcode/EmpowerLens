"""Unit tests for the D14 relabelling rule in src/make_splits_relabelled.py (no files touched)."""
from src.make_splits_relabelled import CP_LABEL, relabel_row

ND = "No Distortion"


def test_single_annotator_takes_their_pair():
    assert relabel_row("Labeling", None, [("Mind Reading", "Labeling")]) == ("Mind Reading", "Labeling")


def test_original_secondary_not_carried_over():
    assert relabel_row("Labeling", "Mind Reading", [("Labeling", None)]) == ("Labeling", None)


def test_calibrated_pessimism_becomes_no_distortion_without_secondary():
    assert relabel_row("Magnification", "Labeling", [(CP_LABEL, None)]) == (ND, None)


def test_secondary_needs_majority_of_agreeing_voters():
    votes = [("Labeling", "Mind Reading"), ("Labeling", "Mind Reading"), ("Labeling", None), ("Magnification", None)]
    assert relabel_row("Magnification", None, votes) == ("Labeling", "Mind Reading")
    votes = [("Labeling", "Mind Reading"), ("Labeling", "Magnification"), ("Labeling", None)]
    assert relabel_row("Magnification", None, votes) == ("Labeling", None)


def test_secondary_equal_to_dominant_is_dropped():
    assert relabel_row(ND, None, [("Labeling", "Labeling")]) == ("Labeling", None)


def test_no_majority_returns_none():
    assert relabel_row("Labeling", None, [("Mind Reading", None), ("Labeling", None)]) is None
    assert relabel_row("Labeling", None, []) is None
