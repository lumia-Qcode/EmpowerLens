"""
EmpowerLens — data loading and label derivation.

Single source of truth for turning ``Annotated_data.csv`` into the three
prediction targets used across the project. Nothing here reads or writes
split files, trains anything, or touches the network.

Targets
-------
``y_bin`` : int, 1 if the dominant label is a distortion, else 0.
``y_mc``  : int in 0..10 over **11 classes** — index 0 is ``no_distortion``,
            indices 1..10 are the ten distortions (order = ``DISTORTIONS``).
            Every row gets a label; there is no ``-1`` sentinel and no row is
            dropped. "No Distortion" is the eleventh class, matching the
            Shreevastava & Foltz (2021) annotation scheme and the frozen
            ``cd_pipeline.py``.
``y_ml``  : float vector of length 10, the union of the dominant and
            (optional) secondary distortion. An all-zeros row encodes
            "No Distortion", so no 11th column is needed.

``y_bin`` is recoverable from ``y_ml`` as ``any(y_ml)`` (a row is distorted iff
at least one of its ten multi-label columns is set); the two are constructed
independently here and the test suite asserts they agree.

Encoding note
-------------
``Annotated_data.csv`` is **UTF-8**, not latin-1. Reading it as latin-1
mojibake-corrupts the ``Patient Question`` text in ~86% of rows (the curly
quotes / em-dashes become ``â\\x80\\x99`` sequences). We read it with
``utf-8-sig`` so any byte-order mark is stripped too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CSV_ENCODING = "utf-8-sig"

# The ten cognitive distortions, in the canonical order used for every
# multi-label column (y_ml[:, j]) and every multi-class index (y_mc == j + 1).
DISTORTIONS = [
    "emotional_reasoning",   # 1
    "overgeneralization",    # 2
    "mental_filter",         # 3
    "should_statements",     # 4
    "all_or_nothing",        # 5
    "mind_reading",          # 6
    "fortune_telling",       # 7
    "magnification",         # 8
    "personalization",       # 9
    "labeling",              # 10
]

# The 11 multi-class labels: no_distortion (index 0) + the ten distortions.
MC_CLASSES = ["no_distortion", *DISTORTIONS]

# Fast lookups from canonical label -> index.
DISTORTION_INDEX = {name: i for i, name in enumerate(DISTORTIONS)}   # 0..9 (y_ml col)
MC_INDEX = {name: i for i, name in enumerate(MC_CLASSES)}            # 0..10 (y_mc)

# Explicit, hand-written mapping from every raw label string observed in
# Annotated_data.csv (both the Dominant and Secondary columns draw from the
# same set) onto a canonical snake_case label. No fuzzy matching: an unmapped
# value raises rather than passing through silently.
LABEL_CANON = {
    "No Distortion": "no_distortion",
    "Emotional Reasoning": "emotional_reasoning",
    "Overgeneralization": "overgeneralization",
    "Mental filter": "mental_filter",
    "Should statements": "should_statements",
    "All-or-nothing thinking": "all_or_nothing",
    "Mind Reading": "mind_reading",
    "Fortune-telling": "fortune_telling",
    "Magnification": "magnification",
    "Personalization": "personalization",
    "Labeling": "labeling",
}

TEXT_COL = "Patient Question"
DOMINANT_COL = "Dominant Distortion"
SECONDARY_COL = "Secondary Distortion (Optional)"
ID_COL = "Id_Number"
MIN_TEXT_CHARS = 10


def _canon(raw: str) -> str:
    """Map one raw label string to its canonical form, or raise if unmapped."""
    key = str(raw).strip()
    if key not in LABEL_CANON:
        raise ValueError(
            f"Unmapped label {raw!r} (normalized {key!r}). Add it to "
            f"LABEL_CANON in src/data.py — no fuzzy matching is allowed."
        )
    return LABEL_CANON[key]


def load_raw(path: str = "Annotated_data.csv") -> pd.DataFrame:
    """
    Load and minimally clean the annotated dataset.

    Reads UTF-8, drops rows with a null ``Patient Question`` or
    ``Dominant Distortion``, drops rows whose text is shorter than
    ``MIN_TEXT_CHARS`` after stripping, and resets the index. On the shipped
    ``Annotated_data.csv`` these filters remove zero rows (all 2,530 survive);
    they exist so the contract holds if the source ever changes.

    Also validates every non-null ``Dominant`` and ``Secondary`` label against
    ``LABEL_CANON`` and raises on any value it does not cover.
    """
    df = pd.read_csv(path, encoding=CSV_ENCODING)

    df = df.dropna(subset=[TEXT_COL, DOMINANT_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df = df[df[TEXT_COL].str.len() >= MIN_TEXT_CHARS].reset_index(drop=True)

    # Fail loudly on any label string not covered by LABEL_CANON.
    seen = set(df[DOMINANT_COL].dropna().map(lambda s: str(s).strip()))
    seen |= set(df[SECONDARY_COL].dropna().map(lambda s: str(s).strip()))
    unmapped = sorted(s for s in seen if s not in LABEL_CANON)
    if unmapped:
        raise ValueError(
            f"{len(unmapped)} label string(s) not in LABEL_CANON: {unmapped}. "
            f"Add explicit mappings in src/data.py."
        )
    return df


def make_targets(df: pd.DataFrame):
    """
    Derive ``(y_bin, y_mc, y_ml)`` from a frame produced by :func:`load_raw`.

    Returns
    -------
    y_bin : np.ndarray[int]   shape (N,)   values {0, 1}
    y_mc  : np.ndarray[int]   shape (N,)   values 0..10 (11 classes)
    y_ml  : np.ndarray[float] shape (N, 10) values {0.0, 1.0}
    """
    n = len(df)
    dom_canon = df[DOMINANT_COL].map(_canon)

    # y_bin: distorted iff the dominant label is not no_distortion.
    y_bin = (dom_canon != "no_distortion").astype(int).to_numpy()

    # y_mc: 0..10, index 0 == no_distortion, 1..10 == the ten distortions.
    y_mc = dom_canon.map(MC_INDEX).astype(int).to_numpy()

    # y_ml: length-10 union of dominant + secondary distortions.
    y_ml = np.zeros((n, len(DISTORTIONS)), dtype=float)
    for i, (dom, sec) in enumerate(zip(dom_canon, df[SECONDARY_COL])):
        if dom != "no_distortion":
            y_ml[i, DISTORTION_INDEX[dom]] = 1.0
        if pd.notna(sec):
            y_ml[i, DISTORTION_INDEX[_canon(sec)]] = 1.0

    return y_bin, y_mc, y_ml


def load_dataset(path: str = "Annotated_data.csv"):
    """Convenience wrapper: return ``(df, y_bin, y_mc, y_ml)`` in one call."""
    df = load_raw(path)
    y_bin, y_mc, y_ml = make_targets(df)
    return df, y_bin, y_mc, y_ml
