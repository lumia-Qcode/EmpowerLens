"""
Shared metric bundle + ``paper_comparison`` row shape.

Both ``baseline_classical.py`` and ``evaluate.py`` build their result rows here,
so the classical floor and the transformer runs are guaranteed to emit the
*identical* column set — that is what lets them be tabulated side by side (and
next to the frozen ``cd_pipeline.py`` numbers). Never report accuracy as a
headline; the headline numbers are F1 variants.

Note on ``no_distortion_wcontrib``: the spec text writes it as ``(933/N) *
f1``, but 933 is the full-dataset No-Distortion count, which is nonsensical
against a 253-row split (it exceeds N). We compute the *split-local* weighted
contribution ``(support_in_split / N_split) * f1`` — identical in form to the
``weighted_contribution`` column of the per-class table, and the quantity that
actually answers "how much of this split's weighted-F1 comes from
no_distortion". Documented here so the deviation is explicit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from src.data import DISTORTIONS, MC_CLASSES

BINARY_CLASSES = ["non_distorted", "distorted"]

# Column order for results/paper_comparison.csv. The first 13 are the spec's
# canonical columns, in the spec's order. Two trailing columns are appended so
# the --reference literature rows can literally carry their uncertainty markers
# (see evaluate.py --reference): `source` marks run vs. paper; `averaging`
# holds "UNSPECIFIED" for the paper's binary 0.79 (whose averaging is unstated).
PAPER_COMPARISON_COLUMNS = [
    "model", "task", "seed", "split",
    "n_classes",
    "weighted_f1", "macro_f1", "macro_f1_10", "micro_f1",
    "positive_class_f1",
    "no_distortion_f1", "no_distortion_wcontrib",
    "truncation_rate",
    "source", "averaging",
]

_PC_KEY = ["model", "task", "seed", "split"]


def per_class_table(task: str, y_true, y_pred) -> pd.DataFrame:
    """One row per class: precision, recall, f1, support, weighted_contribution."""
    if task == "binary":
        labels, names = [0, 1], BINARY_CLASSES
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
    elif task == "multiclass":
        labels, names = list(range(11)), MC_CLASSES
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
    elif task == "multilabel":
        names = DISTORTIONS
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
    else:
        raise ValueError(f"unknown task {task!r}")

    n = int(np.sum(s)) or 1
    df = pd.DataFrame(
        {"class": names, "precision": p, "recall": r, "f1": f, "support": s}
    )
    df["weighted_contribution"] = (df["support"] / n) * df["f1"]
    return df


def metric_bundle(task, y_true, y_pred, model, seed, split,
                  truncation_rate=0.0, source="empowerlens") -> dict:
    """Build one paper_comparison row (blank cells for fields N/A to the task)."""
    row = {c: "" for c in PAPER_COMPARISON_COLUMNS}
    row.update(model=model, task=task, seed=seed, split=split,
               truncation_rate=truncation_rate, source=source)

    if task == "binary":
        row["n_classes"] = 2
        row["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        row["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        row["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
        row["positive_class_f1"] = f1_score(
            y_true, y_pred, pos_label=1, average="binary", zero_division=0
        )

    elif task == "multiclass":
        row["n_classes"] = 11
        row["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        row["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        # macro over the ten distortion classes only, no_distortion (0) dropped.
        row["macro_f1_10"] = f1_score(
            y_true, y_pred, labels=list(range(1, 11)), average="macro", zero_division=0
        )
        row["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
        pc = per_class_table("multiclass", y_true, y_pred)
        nd = pc.loc[pc["class"] == "no_distortion"].iloc[0]
        row["no_distortion_f1"] = float(nd["f1"])
        row["no_distortion_wcontrib"] = float(nd["weighted_contribution"])

    elif task == "multilabel":
        row["n_classes"] = 10
        row["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        row["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        row["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

    else:
        raise ValueError(f"unknown task {task!r}")

    return row


def upsert_paper_comparison(rows, path) -> pd.DataFrame:
    """
    Write/merge rows into results/paper_comparison.csv, keyed by
    (model, task, seed, split). Re-running a config overwrites its old row
    rather than appending a duplicate, so the classical baseline and the
    transformer runs accumulate into one table across invocations.
    """
    path = Path(path)
    new = pd.DataFrame(rows, columns=PAPER_COMPARISON_COLUMNS)
    if path.exists():
        old = pd.read_csv(path)
        for c in PAPER_COMPARISON_COLUMNS:
            if c not in old.columns:
                old[c] = ""
        old = old[PAPER_COMPARISON_COLUMNS]
        combined = pd.concat([old, new], ignore_index=True)
        combined = combined.drop_duplicates(subset=_PC_KEY, keep="last").reset_index(drop=True)
    else:
        combined = new
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined
