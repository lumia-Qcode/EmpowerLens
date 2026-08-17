"""
Compile every experiment run so far into ONE table: results/all_experiments.csv.

Walks every ``results*/`` directory, reads each ``eval_*.json``, and emits one row
per (run, task, split) with the model, the dataset it was trained on, and whether
the numbers are trustworthy.

Why a separate compiler: the runs are scattered across seven directories written at
different times by two scripts with two different JSON shapes, and — critically —
**not all of them are valid**. `results_combined/` was trained on splits with ~75%
train/test overlap, so its numbers sit alongside clean ones with nothing marking
the difference. This adds the `splits` and `valid` columns that make that visible.

`meta.json` only started recording `splits` on 2026-08-16, so for older runs the
dataset is inferred from the results directory (see SPLITS_BY_DIR). Where the
recorded value exists it always wins.

Usage
-----
    python -m src.compile_results
    python -m src.compile_results --out results/all_experiments.csv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd

METRICS = [
    "weighted_f1", "macro_f1", "macro_f1_10", "micro_f1",
    "positive_class_f1", "no_distortion_f1",
]

# Which splits dir each results dir was produced from, for runs whose meta.json
# predates the `splits` field. `valid=False` means the numbers are inflated by
# train/test leakage and must not be compared against anything.
SPLITS_BY_DIR = {
    "results":               ("data/splits",             True,  "Annotated_data only, frozen 80/10/10"),
    "results_codipas":       ("data/splits_codipas_cls", True,  "CODIPAS message-level; y_mc is DERIVED not annotated"),
    "results_combined":      ("data/splits_combined",    False, "LEAKED: 194/253 val + 189/253 test rows also in train"),
    "results_stage1":        ("data/splits",             True,  "cascade Stage 1 (binary)"),
    "results_stage2":        ("data/splits_stage2",      True,  "cascade Stage 2 ISOLATED - distorted-only, not a cascade result"),
    "results_cascade":       ("data/splits",             True,  "cascade end-to-end, composed prediction"),
    "results_multiclass_v2": ("data/splits",             True,  "flat 11-class, clean splits"),
    "results_multilabel_flat": ("data/splits",           True,  "flat multilabel — the cascade's matched competitor"),
    # CODIPAS variants — the bootstrap suffixes every output dir from PARENT_SPLITS,
    # so a CODIPAS run cannot overwrite the Annotated results above.
    "results_stage1_codipas_cls":        ("data/splits_codipas_cls",        True, "CODIPAS cascade Stage 1"),
    "results_stage2_codipas_cls":        ("data/splits_stage2_codipas_cls", True, "CODIPAS Stage 2 ISOLATED - not a cascade result"),
    "results_cascade_codipas_cls":       ("data/splits_codipas_cls",        True, "CODIPAS cascade end-to-end"),
    "results_multilabel_flat_codipas_cls": ("data/splits_codipas_cls",      True, "CODIPAS flat multilabel baseline"),
    "results_multiclass_v2_codipas_cls": ("data/splits_codipas_cls",        True, "CODIPAS flat 11-class"),
    # TRANSFER: trained on CODIPAS, evaluated on the FROZEN Annotated test set.
    # Leakage-free (train-in-test = 0), but scores are depressed by label
    # DISAGREEMENT, not by data volume — CODIPAS's derived y_mc matches the human
    # annotation on only 36.9% of the 2,017 shared texts. Read as label transfer.
    "results_stage1_codipas_transfer":        ("data/splits",        True, "TRANSFER: CODIPAS-trained, Annotated test; see label_agreement"),
    "results_stage2_codipas_transfer":        ("data/splits_stage2_codipas_transfer", True, "TRANSFER Stage 2 ISOLATED - not a cascade result"),
    "results_cascade_codipas_transfer":       ("data/splits",        True, "TRANSFER cascade end-to-end on Annotated test"),
    "results_multilabel_flat_codipas_transfer": ("data/splits",      True, "TRANSFER flat multilabel on Annotated test"),
    "results_multiclass_v2_codipas_transfer": ("data/splits",        True, "TRANSFER flat 11-class on Annotated test"),
    # Size-matched variant: train downsampled to 2,024 to equal Annotated's train,
    # stratified on y_mc. Without it, a transfer result confounds label convention
    # with a 37% larger training set.
    "results_stage1_codipas_transfer_matched":        ("data/splits", True, "TRANSFER size-matched Stage 1"),
    "results_stage2_codipas_transfer_matched":        ("data/splits_stage2_codipas_transfer_matched", True, "TRANSFER size-matched Stage 2 ISOLATED"),
    "results_cascade_codipas_transfer_matched":       ("data/splits", True, "TRANSFER size-matched cascade on Annotated test"),
    "results_multilabel_flat_codipas_transfer_matched": ("data/splits", True, "TRANSFER size-matched flat multilabel"),
    "results_multiclass_v2_codipas_transfer_matched": ("data/splits", True, "TRANSFER size-matched flat 11-class"),
    # SEQUENTIAL fine-tuning (notebooks/kaggle_runner_sequential.ipynb). Both stages
    # are distorted-only, so these are ISOLATED numbers comparable ONLY to
    # results_stage2 (macro_f1 0.277) — never to a flat or cascade result.
    "results_seq_stageA":         ("data/splits_pr_only", True, "SEQ stage A: PatternReframe only; scored on Annotated test = ZERO-SHOT transfer"),
    "results_seq_stageB":         ("data/splits_stage2",  True, "SEQ stage B: continued on Annotated from stage A; compare to results_stage2"),
    "results_seq_stageA_holdout": ("data/splits_pr_only_holdout", True, "SEQ stage A IN-DOMAIN diagnostic (PatternReframe held-out), not a target-task result"),
}


def _rows_from_eval(path: str, folder: str):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = d.get("meta", {})
    splits_dir, valid, note = SPLITS_BY_DIR.get(folder, ("?", None, ""))
    # A recorded value always beats the directory-based guess.
    splits_dir = meta.get("splits", splits_dir)

    for split, blocks in d.get("splits", {}).items():
        # evaluate.py writes {"metrics": ...}; evaluate_cascade.py writes
        # {"binary_metrics": ..., "multilabel_metrics": ...}.
        for block, m in blocks.items():
            if not (isinstance(m, dict) and block.endswith("metrics")):
                continue
            row = {
                "results_dir": folder,
                "splits": splits_dir,
                "valid": valid,
                "note": note,
                "model": m.get("model", meta.get("model", "")),
                "task": m.get("task", meta.get("task", block.replace("_metrics", ""))),
                "seed": m.get("seed", meta.get("seed", "")),
                "split": split,
                "max_length": meta.get("max_length", ""),
                "truncation": meta.get("truncation", ""),
                "loss": meta.get("loss", ""),
                "source_file": os.path.basename(path),
            }
            row.update({k: m.get(k) for k in METRICS})
            yield row


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compile every results*/ dir into one CSV.")
    ap.add_argument("--out", default="results/all_experiments.csv")
    ap.add_argument("--summary", default="results/all_experiments_summary.csv",
                    help="mean +/- std across seeds, test split only")
    args = ap.parse_args(argv)

    rows = []
    for folder in sorted(SPLITS_BY_DIR):
        files = sorted(glob.glob(os.path.join(folder, "eval_*.json")))
        if not files:
            print(f"[skip] {folder}/ — no eval_*.json (results not downloaded?)")
            continue
        for f in files:
            rows.extend(_rows_from_eval(f, folder))
        print(f"[ok]   {folder}/ — {len(files)} eval files")

    if not rows:
        print("\nNothing to compile.")
        return 1

    df = pd.DataFrame(rows)
    # Metrics that don't apply to a task come back as None (macro_f1_10 on binary,
    # positive_class_f1 on multiclass, ...), which makes the whole column object
    # dtype and breaks .mean(). Coerce so they become NaN and are skipped.
    for c in METRICS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(
        ["valid", "results_dir", "task", "model", "split", "seed"],
        ascending=[False, True, True, True, True, True],
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # Mean +/- std across seeds, test only — the shape you actually cite.
    test = df[df["split"] == "test"]
    agg = (test.groupby(["valid", "results_dir", "splits", "task", "model"])[METRICS]
                .agg(["mean", "std"]).round(3))
    agg.to_csv(args.summary)

    print(f"\nWrote {len(df)} rows -> {args.out}")
    print(f"Wrote seed-aggregated summary -> {args.summary}")
    print(f"\n  runs: {test.groupby(['results_dir','task','model']).ngroups} distinct (dir, task, model) combinations")
    n_bad = int((~df["valid"].fillna(True)).sum())
    if n_bad:
        print(f"  !! {n_bad} rows are flagged valid=False (leaked splits) — do not cite or compare them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
