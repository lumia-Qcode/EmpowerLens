"""
Specificity and false-positive rate for every binary run, from the eval JSONs.

Specificity = the share of non-distorted texts the model correctly leaves
unflagged. It is the recall of the ``non_distorted`` class, which evaluate.py
already writes into each ``eval_*.json`` under ``splits/<split>/per_class``, so
nothing is re-predicted and ``test.csv`` is never opened here.

For each (model, seed, split) this reports:

  * ``specificity``  — recall of ``non_distorted`` (true-negative rate)
  * ``fpr``          — false-positive rate, ``1 - specificity``
  * ``sensitivity``  — recall of ``distorted`` (true-positive rate)
  * ``tn`` / ``fp``  — counts behind the rate, out of ``n_non_distorted``

and then ``mean ± std`` across seeds per (model, split), std with ddof=0 to
match src.aggregate.

Usage
-----
    python -m src.specificity [--results results/results_RUN1/results] [--out FILE.csv]

Nothing is written unless ``--out`` is given: the RUN 1 folder is frozen history.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROW_COLUMNS = [
    "model", "seed", "split", "specificity", "fpr", "sensitivity",
    "tn", "fp", "n_non_distorted", "n_distorted",
]
RATE_COLS = ["specificity", "fpr", "sensitivity"]


def _class_row(per_class: list[dict], name: str, path: Path) -> dict:
    for r in per_class:
        if r.get("class") == name:
            return r
    raise ValueError(f"{path}: binary run has no '{name}' row in per_class")


def load_binary_rows(results_dir: str) -> pd.DataFrame:
    """One row per (binary checkpoint, split)."""
    rows = []
    for p in sorted(Path(results_dir).glob("eval_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        if meta.get("task") != "binary":
            continue
        for split, blk in data.get("splits", {}).items():
            per_class = blk.get("per_class", [])
            neg = _class_row(per_class, "non_distorted", p)
            pos = _class_row(per_class, "distorted", p)
            n_neg = int(neg["support"])
            tn = int(round(neg["recall"] * n_neg))
            rows.append({
                "model": meta.get("model", ""),
                "seed": meta.get("seed"),
                "split": split,
                "specificity": neg["recall"],
                "fpr": 1.0 - neg["recall"],
                "sensitivity": pos["recall"],
                "tn": tn,
                "fp": n_neg - tn,
                "n_non_distorted": n_neg,
                "n_distorted": int(pos["support"]),
            })
    return pd.DataFrame(rows, columns=ROW_COLUMNS)


def _fmt(mean, std) -> str:
    if np.isnan(mean):
        return ""
    return f"{mean:.3f} ± {std:.3f}"


def build_meanstd(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, split), sub in df.groupby(["model", "split"]):
        r = {"model": model, "split": split, "n_seeds": sub["seed"].nunique()}
        for c in RATE_COLS:
            r[c] = _fmt(sub[c].mean(), sub[c].std(ddof=0))
        r["mean_tn"] = f"{sub['tn'].mean():.1f} of {int(sub['n_non_distorted'].iloc[0])}"
        out.append(r)
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Specificity / FPR for binary eval JSONs.")
    ap.add_argument("--results", default="results/results_RUN1/results")
    ap.add_argument("--out", default=None, help="optional CSV path for the per-run table")
    args = ap.parse_args(argv)

    df = load_binary_rows(args.results)
    if df.empty:
        print(f"No binary {args.results}/eval_*.json files found.")
        return 0
    df = df.sort_values(["model", "split", "seed"]).reset_index(drop=True)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(f"Specificity from {len(df)} binary (checkpoint × split) rows in {args.results}\n")
    print("=== per run ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== mean ± std across seeds ===")
    print(build_meanstd(df).to_string(index=False))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
