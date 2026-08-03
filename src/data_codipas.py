"""
EmpowerLens — CODIPAS.json loader.

CODIPAS is annotated at a **different granularity** than Annotated_data.csv:
each row is one distortion *span* inside a source message, not one label per
whole message. The same message (``Id_Patient_Question``) can carry several
span rows with different — sometimes contradictory — dominant labels (714 /
3,277 source messages in the shipped file contain 2+ distinct dominant
distortion types across their spans). That is why this is a separate loader
rather than an extension of ``src/data.py``: whole-message classification
would see the identical input text under multiple different labels.

This module's job is narrow and mechanical:
  1. Canonicalize every raw label string onto the *same* ``DISTORTIONS`` /
     ``MC_CLASSES`` vocabulary already used for Annotated_data.csv, so a
     downstream model / metric can treat both datasets identically.
  2. Locate the annotated ``Distorted Part`` inside ``Patient Question`` as
     character offsets (``char_start``, ``char_end``), since the source
     dataset gives you the span *text* but not its position. Matching is
     exact-after-normalization first (quotes/whitespace stripped), then a
     bounded fuzzy fallback (longest common blocks spanning >=60% of the
     span). ~99.9% of rows resolve this way on the shipped file; a small
     number of rows are annotator paraphrase/commentary rather than a literal
     quote and cannot be located — these get ``match_type == "none"`` and
     ``char_start/char_end == None``, and should be **excluded from any
     span-extraction training** (they remain valid for whole-span
     classification, since the label itself is still trustworthy).

No leakage-relevant work happens here — this module only parses and aligns.
Splitting (which must be done at the ``group_id`` level, not the row level,
so the same source message never appears in two splits) lives in
``make_splits_codipas.py``.
"""

from __future__ import annotations

import difflib
import json
from typing import Optional

import pandas as pd

from src.data import DISTORTIONS, MC_CLASSES  # reuse the canonical vocabulary

TEXT_COL = "Patient Question"
SPAN_COL = "Distorted Part"
GROUP_COL = "Id_Patient_Question"
SPAN_ID_COL = "Id_Distorted_Part"

# Raw CODIPAS strings -> canonical snake_case, same targets as src/data.py's
# LABEL_CANON. No fuzzy matching on labels: an unmapped value raises.
LABEL_CANON = {
    "No distortion": "no_distortion",
    "Emotional reasoning": "emotional_reasoning",
    "Overgeneralization": "overgeneralization",
    "Mental filter": "mental_filter",
    "Should statements": "should_statements",
    "All-or-nothing thinking": "all_or_nothing",
    "Mind reading": "mind_reading",
    "Fortune-telling": "fortune_telling",
    "Magnification": "magnification",
    "Personalization": "personalization",
    "Labeling": "labeling",
}

_QUOTES = {'"', "'", "\u2018", "\u2019", "\u201c", "\u201d"}
_FUZZY_MIN_COVERAGE = 0.6  # fraction of the (normalized) span that must be matched


def _canon(raw) -> str:
    key = str(raw).strip()
    if key not in LABEL_CANON:
        raise ValueError(
            f"Unmapped CODIPAS label {raw!r}. Add it to LABEL_CANON in "
            f"src/data_codipas.py — no fuzzy matching on labels is allowed."
        )
    return LABEL_CANON[key]


def _build_norm_map(s: str):
    """Lowercase + strip quote chars + collapse whitespace, while keeping a
    map from each normalized-string index back to the original string index
    (so a match found in normalized space can be reported as real offsets)."""
    out_chars, idx_map = [], []
    prev_space = True
    for i, c in enumerate(s):
        if c in _QUOTES:
            continue
        if c.isspace():
            if not prev_space:
                out_chars.append(" ")
                idx_map.append(i)
            prev_space = True
            continue
        out_chars.append(c.lower())
        idx_map.append(i)
        prev_space = False
    return "".join(out_chars), idx_map


def find_span(text: str, span_text: str):
    """Return (char_start, char_end, match_type) locating span_text inside
    text. match_type is one of 'exact', 'fuzzy', 'none'. End is exclusive."""
    norm_text, idx_map = _build_norm_map(text)
    norm_span, _ = _build_norm_map(span_text)
    if not norm_span:
        return None, None, "none"

    pos = norm_text.find(norm_span)
    if pos != -1:
        start = idx_map[pos]
        end = idx_map[pos + len(norm_span) - 1] + 1
        return start, end, "exact"

    sm = difflib.SequenceMatcher(None, norm_text, norm_span, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size > 0]
    if not blocks:
        return None, None, "none"
    matched_chars = sum(b.size for b in blocks)
    if matched_chars < _FUZZY_MIN_COVERAGE * len(norm_span):
        return None, None, "none"

    a_start = min(b.a for b in blocks)
    a_end = max(b.a + b.size for b in blocks)
    start = idx_map[a_start]
    end = idx_map[a_end - 1] + 1
    return start, end, "fuzzy"


def load_raw(path: str = "CODIPAS.json") -> pd.DataFrame:
    """Load CODIPAS.json into a flat, row-per-span DataFrame with canonical
    labels and located char offsets. Every row of the source file survives —
    nothing is dropped here (unlocatable spans are flagged, not removed)."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for item in raw:
        text = str(item[TEXT_COL])
        span_text = str(item[SPAN_COL])
        dominant = _canon(item["Distortions"]["Dominant Distortion"])
        secondary_raw = item["Distortions"].get("Secondary Distortion")
        secondary = _canon(secondary_raw) if secondary_raw else None

        char_start, char_end, match_type = find_span(text, span_text)

        rows.append({
            "group_id": str(item[GROUP_COL]),
            "span_id": item[SPAN_ID_COL],
            "text": text,
            "span_text": span_text,
            "dominant": dominant,
            "secondary": secondary,
            "char_start": char_start,
            "char_end": char_end,
            "match_type": match_type,
        })

    df = pd.DataFrame(rows)
    df["y_mc"] = df["dominant"].map({c: i for i, c in enumerate(MC_CLASSES)}).astype(int)
    df["y_bin"] = (df["dominant"] != "no_distortion").astype(int)
    return df


def group_label_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """One row per group_id: a 10-column multi-hot of every distortion type
    that appears anywhere among that group's spans (union across its rows),
    plus y_bin (1 if the group has ANY distorted span). Used by
    make_splits_codipas.py to stratify the split at the group level — the
    unit that must not be split across train/val/test."""
    idx = {d: i for i, d in enumerate(DISTORTIONS)}
    out = {}
    for gid, sub in df.groupby("group_id"):
        vec = [0] * len(DISTORTIONS)
        for d in sub["dominant"]:
            if d != "no_distortion":
                vec[idx[d]] = 1
        out[gid] = vec
    mat = pd.DataFrame.from_dict(out, orient="index", columns=DISTORTIONS)
    mat["y_bin"] = (mat[DISTORTIONS].sum(axis=1) > 0).astype(int)
    mat.index.name = "group_id"
    return mat.reset_index()


def summary(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(len(df)),
        "n_groups": int(df["group_id"].nunique()),
        "match_type_counts": df["match_type"].value_counts().to_dict(),
        "dominant_counts": df["dominant"].value_counts().to_dict(),
        "groups_with_multiple_distinct_dominants": int(
            df.groupby("group_id")["dominant"].nunique().gt(1).sum()
        ),
    }
