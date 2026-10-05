"""Tests for build_inputs — the three input representations behind the span gate.

The gate (docs/FYP_PLAN.md section 4) compares a model trained on full documents
against one trained on gold spans. That comparison is only meaningful if the
span modes change the text and nothing else, so these tests pin the invariants
that would silently corrupt the result if they broke.
"""

import pandas as pd
import pytest

from src.train_transformer import SPAN_COL, SPAN_MARKER, TEXT_COL, build_inputs

SPLITS = "data/splits_stage2_annotated"


@pytest.fixture(scope="module")
def val():
    return pd.read_csv(f"{SPLITS}/val.csv", encoding="utf-8-sig")


def test_splits_carry_the_span_column(val):
    assert SPAN_COL in val.columns, (
        f"{SPLITS} lost '{SPAN_COL}'. Regenerate with src.make_splits_cascade — "
        "span training reads it from here."
    )


def test_document_mode_is_untouched(val):
    texts, stats = build_inputs(val, "document")
    assert texts.equals(val[TEXT_COL].astype(str))
    assert stats == {}, "document mode must not report span stats"


def test_span_crop_is_the_span_verbatim(val):
    texts, _ = build_inputs(val, "span_crop")
    assert texts.equals(val[SPAN_COL].astype(str))
    # The whole premise of the gate is that cropping discards most of the text.
    assert texts.str.len().median() < val[TEXT_COL].str.len().median() / 2


def test_span_marked_preserves_the_document(val):
    """Marking may only insert markers — if it drops or reorders characters, a
    score difference against `document` no longer isolates the marking."""
    docs, _ = build_inputs(val, "document")
    marked, _ = build_inputs(val, "span_marked")
    for doc, mark in zip(docs, marked):
        stripped = mark.replace(SPAN_MARKER, "")
        assert "".join(stripped.split()) == "".join(doc.split())


def test_span_marked_match_rate_is_near_total(val):
    _, stats = build_inputs(val, "span_marked")
    assert stats["span_match_rate"] >= 0.99, (
        "span alignment regressed; find_span resolves all but ~4 rows repo-wide"
    )
    assert stats["span_match_exact"] + stats["span_match_fuzzy"] + stats["span_match_none"] == len(val)


def test_unlocatable_span_is_left_unmarked_not_dropped():
    """A span that cannot be found must degrade to the plain document. Dropping
    the row instead would change the eval set size between arms."""
    df = pd.DataFrame({
        TEXT_COL: ["I always fail at everything I try."],
        SPAN_COL: ["completely unrelated text that appears nowhere"],
    })
    texts, stats = build_inputs(df, "span_marked")
    assert len(texts) == 1
    assert texts.iloc[0] == df[TEXT_COL].iloc[0]
    assert stats["span_match_none"] == 1


def test_span_modes_reject_splits_without_the_column():
    df = pd.DataFrame({TEXT_COL: ["some text"]})
    with pytest.raises(SystemExit, match=SPAN_COL):
        build_inputs(df, "span_crop")


def test_span_modes_reject_no_distortion_rows():
    """No-Distortion rows have no span to crop. Failing loudly beats training on
    the string 'nan'."""
    df = pd.DataFrame({TEXT_COL: ["some text"], SPAN_COL: [None]})
    with pytest.raises(SystemExit, match="empty"):
        build_inputs(df, "span_crop")
