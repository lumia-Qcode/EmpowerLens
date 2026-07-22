"""
Classical baseline: TF-IDF + Logistic Regression — the floor the transformer
must beat to justify its cost.

TF-IDF (1-2 grams, ``min_df=2``, ``sublinear_tf=True``) is fit on **train
only** and applied to val; LogisticRegression uses ``class_weight="balanced"``.
Evaluated on val for all three tasks (binary / multiclass / multilabel). Emits
the same metric bundle and the same ``paper_comparison.csv`` row shape as the
transformer runs, via :mod:`src.metrics`, so the rows tabulate side by side.

The test set is never read here — only train and val.

Usage
-----
    python -m src.baseline_classical [--seeds 42,1337,2024]
                                     [--splits data/splits] [--out results]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from src.data import DISTORTIONS
from src.metrics import (
    metric_bundle,
    per_class_table,
    upsert_paper_comparison,
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

TEXT_COL = "Patient Question"
ML_COLS = [f"ml_{d}" for d in DISTORTIONS]
MODEL_NAME = "tfidf+logreg"


def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _logreg(seed: int) -> LogisticRegression:
    return LogisticRegression(
        class_weight="balanced", max_iter=2000, random_state=seed
    )


def run_seed(train: pd.DataFrame, val: pd.DataFrame, seed: int):
    """Return (list-of-rows, multiclass_per_class_df) for one seed."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_tr = vec.fit_transform(train[TEXT_COL])   # fit on train ONLY
    x_va = vec.transform(val[TEXT_COL])

    rows = []

    # binary
    clf = _logreg(seed).fit(x_tr, train["y_bin"])
    pred = clf.predict(x_va)
    rows.append(metric_bundle("binary", val["y_bin"].to_numpy(), pred,
                              MODEL_NAME, seed, "val"))

    # multiclass (11 classes)
    clf = _logreg(seed).fit(x_tr, train["y_mc"])
    pred_mc = clf.predict(x_va)
    y_mc = val["y_mc"].to_numpy()
    rows.append(metric_bundle("multiclass", y_mc, pred_mc, MODEL_NAME, seed, "val"))
    pc_mc = per_class_table("multiclass", y_mc, pred_mc)

    # multilabel (10 sigmoid one-vs-rest)
    clf = OneVsRestClassifier(_logreg(seed)).fit(x_tr, train[ML_COLS].to_numpy())
    pred_ml = clf.predict(x_va)
    rows.append(metric_bundle("multilabel", val[ML_COLS].to_numpy(), pred_ml,
                              MODEL_NAME, seed, "val"))

    return rows, pc_mc


def _fmt(series):
    return f"{series.mean():.3f} ± {series.std(ddof=0):.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Classical TF-IDF + LogReg baseline.")
    ap.add_argument("--seeds", default="42,1337,2024",
                    help="comma-separated seeds (default 42,1337,2024)")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",")]
    train = _load(f"{args.splits}/train.csv")
    val = _load(f"{args.splits}/val.csv")

    all_rows, pc_mc_by_seed = [], {}
    for seed in seeds:
        rows, pc_mc = run_seed(train, val, seed)
        all_rows.extend(rows)
        pc_mc_by_seed[seed] = pc_mc

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = upsert_paper_comparison(all_rows, out / "paper_comparison.csv")
    df = df[df["model"] == MODEL_NAME]  # for the console summary below

    # Persist the seed-42 multiclass per-class table (the highlighted artifact).
    pc_mc_by_seed[seeds[0]].to_csv(
        out / f"per_class_tfidf_logreg_multiclass_{seeds[0]}.csv", index=False
    )

    # ---- console report -----------------------------------------------------
    pd.set_option("display.width", 120)
    print(f"\nSeeds: {seeds}   (train={len(train)}, val={len(val)})\n")

    for task in ("binary", "multiclass", "multilabel"):
        sub = df[df["task"] == task]
        print(f"=== {task.upper()} (val) — mean ± std over {len(seeds)} seeds ===")
        print(f"  weighted_f1 : {_fmt(sub['weighted_f1'])}")
        print(f"  macro_f1    : {_fmt(sub['macro_f1'])}")
        if task == "multiclass":
            print(f"  macro_f1_10 : {_fmt(sub['macro_f1_10'])}  (distortion classes only)")
        print(f"  micro_f1    : {_fmt(sub['micro_f1'])}")
        if task == "binary":
            print(f"  pos_class_f1: {_fmt(sub['positive_class_f1'])}  (Distorted class)")
        print()

    # ---- the headline the prompt asks to see FIRST --------------------------
    mc = df[df["task"] == "multiclass"]
    nd_f1 = mc["no_distortion_f1"].astype(float)
    nd_wc = mc["no_distortion_wcontrib"].astype(float)
    wf1 = mc["weighted_f1"].astype(float)
    share = (nd_wc / wf1)
    print("-" * 72)
    print("NO_DISTORTION weighted-F1 contribution (11-class multiclass, val):")
    print(f"  no_distortion F1            : {nd_f1.mean():.3f}")
    print(f"  weighted contribution       : {nd_wc.mean():.3f}  "
          f"= (support/N) * F1")
    print(f"  multiclass weighted_f1      : {wf1.mean():.3f}")
    print(f"  share of weighted_f1 from   : {share.mean()*100:.1f}%  "
          f"of the headline number is just the easy majority class")
    print("-" * 72)
    print(f"\nWrote {out/'paper_comparison.csv'} "
          f"and per-class table for seed {seeds[0]}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
