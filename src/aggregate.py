"""
Aggregate all per-checkpoint eval JSONs into the Month-1 summary tables.

Reads every ``results/eval_*.json`` (written by evaluate.py) and emits:

  * ``results/month1_summary.csv``          — one row per (model, task, seed,
    split): the headline metrics, machine-readable.
  * ``results/month1_summary_meanstd.csv``  — the same metrics aggregated to
    ``mean ± std`` across seeds (one row per model/task/split), three decimals.
  * ``results/no_distortion_contribution.md`` — a generated thesis note: for
    every 11-class run, the share of the weighted-F1 attributable to the single
    easy ``no_distortion`` class.

Usage
-----
    python -m src.aggregate [--results results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.metrics import PAPER_COMPARISON_COLUMNS

# Columns reported in month1_summary.csv (a readable subset of the full bundle).
SUMMARY_COLUMNS = [
    "model", "task", "n_classes", "seed", "split",
    "weighted_f1", "macro_f1", "macro_f1_10", "micro_f1",
    "positive_class_f1", "no_distortion_f1", "truncation_rate",
]
METRIC_COLS = [
    "weighted_f1", "macro_f1", "macro_f1_10", "micro_f1",
    "positive_class_f1", "no_distortion_f1", "truncation_rate",
]


def load_eval_rows(results_dir: str) -> pd.DataFrame:
    """One row per (checkpoint, split), with the full metric bundle."""
    rows = []
    for p in sorted(Path(results_dir).glob("eval_*.json")):
        data = json.loads(p.read_text())
        for split, blk in data.get("splits", {}).items():
            m = blk.get("metrics", {})
            rows.append({c: m.get(c, "") for c in PAPER_COMPARISON_COLUMNS})
    df = pd.DataFrame(rows, columns=PAPER_COMPARISON_COLUMNS)
    for c in METRIC_COLS + ["no_distortion_wcontrib"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _fmt(mean, std) -> str:
    if np.isnan(mean):
        return ""
    return f"{mean:.3f} ± {std:.3f}"


def build_meanstd(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, task, n_classes, split), sub in df.groupby(
        ["model", "task", "n_classes", "split"], dropna=False
    ):
        r = {"model": model, "task": task, "n_classes": n_classes,
             "split": split, "n_seeds": sub["seed"].nunique()}
        for c in METRIC_COLS:
            vals = sub[c].dropna()
            r[c] = _fmt(vals.mean(), vals.std(ddof=0)) if len(vals) else ""
        out.append(r)
    return pd.DataFrame(out)


def build_no_distortion_md(df: pd.DataFrame) -> str:
    mc = df[df["task"] == "multiclass"].copy()
    lines = [
        "# No-Distortion contribution to weighted-F1 (11-class multiclass)",
        "",
        "For each 11-class run, the share of the weighted-F1 attributable to the "
        "single `no_distortion` class — the large, easy majority category. A high "
        "share means the headline weighted-F1 mostly reflects detecting the "
        "*absence* of distortion, not discriminating *between* distortions.",
        "",
    ]
    if mc.empty:
        lines.append("_No 11-class runs found in results/eval_*.json yet._")
        return "\n".join(lines) + "\n"

    lines += [
        "| model | seed | split | no_distortion F1 | weighted_contribution | weighted_f1 | share of weighted_f1 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in mc.sort_values(["model", "split", "seed"]).iterrows():
        wf1 = r["weighted_f1"]
        wc = r["no_distortion_wcontrib"]
        share = (wc / wf1 * 100) if wf1 else float("nan")
        lines.append(
            f"| {r['model']} | {r['seed']} | {r['split']} | "
            f"{r['no_distortion_f1']:.3f} | {wc:.3f} | {wf1:.3f} | {share:.1f}% |"
        )

    # mean share on test, across seeds, per model
    test = mc[mc["split"] == "test"]
    if not test.empty:
        lines += ["", "## Mean share on **test** (across seeds)", ""]
        for model, sub in test.groupby("model"):
            shares = (sub["no_distortion_wcontrib"] / sub["weighted_f1"] * 100).dropna()
            if len(shares):
                lines.append(
                    f"- **{model}**: {shares.mean():.1f}% ± {shares.std(ddof=0):.1f}% "
                    f"of the test weighted-F1 comes from `no_distortion` "
                    f"(n={sub['seed'].nunique()} seeds)."
                )
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Aggregate eval JSONs into Month-1 tables.")
    ap.add_argument("--results", default="results")
    args = ap.parse_args(argv)

    df = load_eval_rows(args.results)
    if df.empty:
        print(f"No {args.results}/eval_*.json files found — run src.evaluate first.")
        return 0

    out = Path(args.results)
    summary = df[SUMMARY_COLUMNS].copy()
    summary.to_csv(out / "month1_summary.csv", index=False)

    meanstd = build_meanstd(df)
    meanstd.to_csv(out / "month1_summary_meanstd.csv", index=False)

    md = build_no_distortion_md(df)
    (out / "no_distortion_contribution.md").write_text(md, encoding="utf-8")

    # console echo
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(f"Aggregated {len(df)} (checkpoint × split) rows from {args.results}/eval_*.json\n")
    print("=== month1_summary (per run) ===")
    print(summary.to_string(index=False))
    print("\n=== mean ± std across seeds ===")
    print(meanstd.to_string(index=False))
    print(f"\nWrote month1_summary.csv, month1_summary_meanstd.csv, "
          f"no_distortion_contribution.md to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
