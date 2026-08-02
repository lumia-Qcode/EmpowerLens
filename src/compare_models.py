"""
Compare two (or more) fine-tuned model tags head-to-head — e.g. the RoBERTa
baseline vs. the MentalBERT swap — across every task, using files already
produced by ``evaluate.py`` / ``aggregate.py``. Reads nothing raw: no
``test.csv``, no re-scoring. It only pivots ``results/eval_*.json`` (the same
source ``aggregate.py`` uses) and the per-class CSVs ``evaluate.py`` writes.

Outputs (written to ``--out``, default ``results/``):
  * ``model_comparison.csv``          — long format: model, task, split, metric, mean, std, n_seeds
  * ``model_comparison_headline.md``  — a wide, human-readable table with a winner marked per row
  * ``model_comparison_per_class_<task>.csv`` — per-distortion F1 for each model + the delta, sorted by |delta|

Usage
-----
    python -m src.compare_models
    python -m src.compare_models --models roberta-base,mental-bert-base-uncased
    python -m src.compare_models --split val   # compare on val instead of test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.aggregate import METRIC_COLS, load_eval_rows

# Which direction is "better" — every metric here is an F1 variant, so higher
# always wins. truncation_rate is informational only (lower is "better" input
# fit, but it is not a model-quality metric) and is excluded from winner-marking.
WINNER_METRICS = [
    "weighted_f1", "macro_f1", "macro_f1_10", "micro_f1",
    "positive_class_f1", "no_distortion_f1",
]

TASK_HEADLINE_METRIC = {
    "binary": "positive_class_f1",
    "multiclass": "macro_f1_10",
    "multilabel": "macro_f1",
}


def _tag(model: str) -> str:
    """Match the tag scheme used everywhere else (train_transformer.py,
    evaluate.py): the part of the HF model id after the last '/'."""
    return model.split("/")[-1]


def build_long_table(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """One row per (model, task, split, metric): mean ± std across seeds."""
    df = df[df["model"].isin(models)].copy()
    if df.empty:
        raise SystemExit(
            f"No rows found for models {models} in results/eval_*.json. "
            f"Have you run evaluate.py for each checkpoint yet?"
        )
    rows = []
    for (model, task, split), sub in df.groupby(["model", "task", "split"]):
        for metric in METRIC_COLS:
            vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
            if vals.empty:
                continue
            rows.append({
                "model": model, "task": task, "split": split, "metric": metric,
                "mean": vals.mean(), "std": vals.std(ddof=0), "n_seeds": sub["seed"].nunique(),
            })
    return pd.DataFrame(rows)


def build_headline_md(long_df: pd.DataFrame, models: list[str], split: str) -> str:
    sub = long_df[long_df["split"] == split]
    lines = [
        f"# Model comparison — {split} split",
        "",
        "Every metric is an F1 variant (higher is better). **Bold** marks the "
        "winner per row; a tie within 0.005 is marked as a tie rather than forced.",
        "",
    ]
    header = "| task | metric | " + " | ".join(models) + " |"
    sep = "|---|---|" + "---|" * len(models)
    lines += [header, sep]

    for task in ("binary", "multiclass", "multilabel"):
        task_sub = sub[sub["task"] == task]
        if task_sub.empty:
            continue
        metrics_present = [m for m in WINNER_METRICS if m in task_sub["metric"].unique()]
        for metric in metrics_present:
            row = task_sub[task_sub["metric"] == metric]
            cells = []
            values = {}
            for model in models:
                r = row[row["model"] == model]
                if r.empty:
                    cells.append("—")
                    continue
                m, s = r["mean"].iloc[0], r["std"].iloc[0]
                values[model] = m
                cells.append(f"{m:.3f} ± {s:.3f}")
            if len(values) >= 2:
                best_model = max(values, key=values.get)
                spread = max(values.values()) - min(values.values())
                if spread > 0.005:
                    for i, model in enumerate(models):
                        if model == best_model:
                            cells[i] = f"**{cells[i]}**"
            flag = "  *(headline)*" if metric == TASK_HEADLINE_METRIC.get(task) else ""
            lines.append(f"| {task} | {metric}{flag} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def per_class_delta(results_dir: Path, task: str, model_a: str, model_b: str,
                     seeds=(42, 1337, 2024)) -> pd.DataFrame | None:
    """Average per-class F1 across available seeds for each model, then diff
    (model_b - model_a). Silently skips seeds whose CSV is missing."""
    def avg_for(model):
        tag = _tag(model)
        frames = []
        for seed in seeds:
            p = results_dir / f"per_class_{tag}_{task}_{seed}.csv"
            if p.exists():
                frames.append(pd.read_csv(p).set_index("class")["f1"])
        if not frames:
            return None
        return pd.concat(frames, axis=1).mean(axis=1)

    fa, fb = avg_for(model_a), avg_for(model_b)
    if fa is None or fb is None:
        return None
    out = pd.DataFrame({_tag(model_a): fa, _tag(model_b): fb})
    out["delta"] = out[_tag(model_b)] - out[_tag(model_a)]
    return out.sort_values("delta", key=np.abs, ascending=False).reset_index().rename(columns={"index": "class"})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare fine-tuned model tags head-to-head.")
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", default="roberta-base,mental/mental-bert-base-uncased",
                     help="comma-separated HF model ids/tags exactly as passed to --model in train_transformer.py")
    ap.add_argument("--split", default="test", choices=["val", "test"],
                     help="which split drives the headline table (default test — the honest number)")
    ap.add_argument("--out", default=None, help="defaults to --results")
    args = ap.parse_args(argv)

    results_dir = Path(args.results)
    out_dir = Path(args.out) if args.out else results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",")]

    df = load_eval_rows(str(results_dir))
    long_df = build_long_table(df, models)
    long_df.to_csv(out_dir / "model_comparison.csv", index=False)

    md = build_headline_md(long_df, models, args.split)
    (out_dir / "model_comparison_headline.md").write_text(md, encoding="utf-8")

    print(md)

    model_a, model_b = models[0], models[-1]
    for task in ("multiclass", "multilabel"):
        pc = per_class_delta(results_dir, task, model_a, model_b)
        if pc is None:
            print(f"[skip] no per-class CSVs found for both models on task={task}")
            continue
        pc_path = out_dir / f"model_comparison_per_class_{task}.csv"
        pc.to_csv(pc_path, index=False)
        print(f"\n=== per-class F1 delta ({_tag(model_b)} - {_tag(model_a)}), task={task} ===")
        print(pc.round(3).to_string(index=False))

    print(f"\nWrote model_comparison.csv, model_comparison_headline.md, "
          f"model_comparison_per_class_{{multiclass,multilabel}}.csv to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
