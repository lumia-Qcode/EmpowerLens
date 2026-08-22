"""
Score one checkpoint four ways: flat 0.5 vs swept thresholds, capped vs uncapped.

WHY FOUR
--------
Two independent decisions sit between a model's probabilities and its reported
score, and neither has an obviously right answer on this data:

**The threshold.** ``0.5`` is the tutorial's. The swept per-label cut points are
fitted on val by ``sweep_thresholds``. On an under-firing model the flat 0.5
reports almost nothing, so the two can differ enormously.

**The cap.** ``--max-labels 2`` is ``src/evaluate.py``'s default and how every
model in ``results/`` was scored — justified because no row in the corpus carries
more than 2 labels. ``0`` means uncapped, which is what the tutorial does.

They interact in a way that is easy to miss. Measured on simulated probabilities
matching the two regimes seen in this project:

===================  ====  ==========  ====================
probabilities        cap   labels/row  rows predicted empty
===================  ====  ==========  ====================
spread @ 0.5          2       1.98              0
spread @ 0.05         2       2.00              0
collapsed @ 0.5       2       0.00            253 of 253
collapsed @ 0.05      2       2.00              0
===================  ====  ==========  ====================

Once **two or more** labels clear the threshold, the cap keeps the top 2 by
probability — and the top 2 are the same wherever the line sits. So under a cap
the threshold's only remaining job is deciding whether to predict *nothing*.
That is invisible unless you score all four combinations, which is what this
does. (True rate on val: 0.80 labels/row, 95 of 253 rows carrying none.)

HOW
---
No retraining. The weights are identical in all four passes; only the
``thresholds`` field of ``meta.json`` and the ``--max-labels`` flag change. The
original ``meta.json`` is restored in a ``finally`` block, so an interrupted run
cannot leave a checkpoint mislabelled.

Multilabel only — binary and multiclass use argmax and have no threshold.

Usage
-----
    python -m src.eval_thresholds --checkpoint checkpoints/tutorial_multilabel_... \\
        --out results_tutorial_distilbert
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

METRICS = ["macro_f1", "micro_f1", "weighted_f1"]


def run_eval(checkpoint: Path, out_dir: Path, max_labels: int, splits: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [sys.executable, "-m", "src.evaluate", "--checkpoint", str(checkpoint),
         "--splits", splits, "--out", str(out_dir),
         "--max-labels", str(max_labels)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"src.evaluate failed:\n{r.stdout}\n{r.stderr}")
    latest = max(out_dir.glob("eval_*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(latest.read_text(encoding="utf-8"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=None,
                    help="defaults to the splits recorded in the checkpoint")
    ap.add_argument("--caps", default="2,0",
                    help="comma-separated --max-labels values to try")
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint)
    meta_path = ckpt / "meta.json"
    original = meta_path.read_text(encoding="utf-8")
    meta = json.loads(original)

    if meta.get("task") != "multilabel":
        raise SystemExit(f"{ckpt.name} is task={meta.get('task')!r}; thresholds "
                         f"only apply to multilabel (argmax has no cut point).")

    splits = args.splits or meta.get("splits", "data/splits")
    caps = [int(c) for c in args.caps.split(",") if c.strip()]
    modes = {"fixed": meta.get("thresholds_fixed") or [0.5] * 10,
             "tuned": meta.get("thresholds_tuned")}
    if not modes["tuned"]:
        raise SystemExit(f"{meta_path} has no thresholds_tuned — retrain with a "
                         f"current src/tutorial_distilbert.py.")

    rows = []
    out_root = Path(args.out)
    try:
        for mode, thr in modes.items():
            # Only this field changes between passes; the weights never move.
            meta["thresholds"] = thr
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            for cap in caps:
                d = run_eval(ckpt, out_root / f"thr_{mode}_cap{cap}", cap, splits)
                for split_name, block in d.get("splits", {}).items():
                    m = block.get("metrics", {})
                    rows.append({
                        "checkpoint": ckpt.name, "model": meta.get("model"),
                        "seed": meta.get("seed"), "loss": meta.get("loss"),
                        "trained_on": splits, "split": split_name,
                        "threshold_mode": mode,
                        "threshold": ("0.5 flat" if mode == "fixed"
                                      else f"swept {min(thr)}-{max(thr)}"),
                        "max_labels": cap if cap else "uncapped",
                        **{k: (float(m[k]) if isinstance(m.get(k), (int, float))
                               and m[k] != "" else None) for k in METRICS},
                    })
    finally:
        # Always put the checkpoint back exactly as it was, even on Ctrl-C —
        # a checkpoint whose meta disagrees with how it was scored is worse
        # than no result at all.
        meta_path.write_text(original, encoding="utf-8")

    df = pd.DataFrame(rows)
    csv = out_root / "threshold_grid.csv"
    if csv.exists():
        df = (pd.concat([pd.read_csv(csv), df], ignore_index=True)
                .drop_duplicates(subset=["checkpoint", "split", "threshold_mode",
                                         "max_labels"], keep="last"))
    df.to_csv(csv, index=False, encoding="utf-8")

    test = df[df["split"] == "test"]
    print(f"\n{ckpt.name}  (test set, {splits})")
    if not test.empty:
        print(test[["threshold_mode", "threshold", "max_labels", *METRICS]]
              .round(4).to_string(index=False))
        best = test.loc[test["macro_f1"].idxmax()]
        print(f"\nbest: {best['threshold_mode']} thresholds, "
              f"max_labels={best['max_labels']} -> macro_f1 {best['macro_f1']:.4f}")
        print("Report the max_labels=2 row for comparison with the other models "
              "in results/;\nthe uncapped row is the tutorial's own behaviour.")
    print(f"appended to {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
