"""
Evaluate one checkpoint on TWO test sets, so a run can be read two ways.

WHY TWO
-------
A model trained on CODIPAS and scored on the CODIPAS test set answers "how well
did it learn this corpus". A model trained on CODIPAS and scored on the
**Annotated** test set answers "does that transfer to our real task". Those are
different questions, and the old experiment suite only ever asked the first —
which is why no two experiments in it could be compared.

Every run here is therefore scored on both:

**HOME**      the test set of whatever the model trained on.
              -> within-dataset performance.
**YARDSTICK** ``data/splits/test.csv`` — the 253 human-annotated rows, always
              the same 253, for every experiment.
              -> the only number comparable across experiments, across models,
                 and against the Month-1 baselines.

``transfer_gap = home − yardstick`` on the same metric. A large positive gap
means the model learned its own corpus but did not carry over.

When a model trains on the Annotated data itself, home == yardstick; the script
detects that and evaluates once, marking the row ``home_is_yardstick``.

THE PRECONDITION
----------------
The yardstick is only a fair exam if the model never studied it. This script
**refuses to run** if the training splits' train.csv contains any row whose text
appears in the yardstick's val or test — the exact defect measured in
``data/splits_combined`` (77% of the test set leaked). Use
``python -m src.make_splits_clean`` first, or pass ``--allow-leak`` to record a
deliberately contaminated number with a loud marker in the output.

Usage
-----
    python -m src.eval_two_exams --checkpoint checkpoints/foo --out results/exp2
    python -m src.eval_two_exams --checkpoint checkpoints/foo --out results/exp2 \\
        --home-splits data/splits_codipas_clean
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.make_splits_clean import audit

YARDSTICK = "data/splits"
METRICS = ["weighted_f1", "macro_f1", "macro_f1_10", "micro_f1",
           "positive_class_f1", "no_distortion_f1"]


def run_evaluate(checkpoint: Path, splits: str, out_dir: Path,
                 max_labels: int = 0) -> Path:
    """Shell out to src.evaluate — the one module allowed to read test.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "src.evaluate",
           "--checkpoint", str(checkpoint), "--splits", splits,
           "--out", str(out_dir), "--max-labels", str(max_labels)]
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"src.evaluate failed on {splits}:\n{r.stdout}\n{r.stderr}")
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")
    written = sorted(out_dir.glob("eval_*.json"), key=lambda p: p.stat().st_mtime)
    if not written:
        raise SystemExit(f"src.evaluate wrote no eval JSON into {out_dir}")
    return written[-1]


def read_metrics(eval_json: Path, split: str = "test") -> dict:
    d = json.loads(eval_json.read_text(encoding="utf-8"))
    block = d.get("splits", {}).get(split, {}).get("metrics", {})
    out = {}
    for m in METRICS:
        v = block.get(m, "")
        out[m] = float(v) if isinstance(v, (int, float)) and v != "" else None
    return out, d.get("meta", {})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="results dir for this experiment")
    ap.add_argument("--home-splits", default=None,
                    help="defaults to the splits recorded in the checkpoint's meta.json")
    ap.add_argument("--yardstick-splits", default=YARDSTICK)
    ap.add_argument("--max-labels", type=int, default=0,
                    help="multilabel prediction cap; 0 = uncapped")
    ap.add_argument("--tag", default=None, help="label for this run in the CSV")
    ap.add_argument("--allow-leak", action="store_true",
                    help="score anyway when the training data contaminates the "
                         "yardstick, marking the row leaked=True")
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint)
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
    home = args.home_splits or meta.get("splits", YARDSTICK)
    task = meta.get("task", "?")
    out_dir = Path(args.out)

    # --- the precondition ---
    leak = audit(home, args.yardstick_splits)
    contaminated = leak["train_rows_in_reference_val_or_test"] > 0
    if contaminated and not args.allow_leak:
        raise SystemExit(
            f"REFUSING TO SCORE.\n"
            f"  {home}/train.csv contains {leak['train_rows_in_reference_val_or_test']} "
            f"rows that also appear in {args.yardstick_splits} val/test.\n"
            f"  That is {leak['reference_test_pct_leaked']}% of the yardstick test "
            f"set — the model studied the exam, so the score is meaningless.\n"
            f"  Fix:  python -m src.make_splits_clean\n"
            f"  Then retrain on the *_clean dir. Use --allow-leak only to record "
            f"a knowingly contaminated number.")
    if contaminated:
        print(f"!! LEAKED: {leak['reference_test_pct_leaked']}% of the yardstick "
              f"test set is in {home}/train.csv. Row marked leaked=True.")

    home_is_yardstick = Path(home).resolve() == Path(args.yardstick_splits).resolve()
    rows = []

    print(f"\n[home]      {home}")
    hj = run_evaluate(ckpt, home, out_dir / "home", args.max_labels)
    hm, _ = read_metrics(hj)
    rows.append({"exam": "home", "test_set": home, **hm})

    if home_is_yardstick:
        print(f"[yardstick] same as home ({home}) — evaluated once")
        rows.append({"exam": "yardstick", "test_set": args.yardstick_splits, **hm})
    else:
        print(f"[yardstick] {args.yardstick_splits}")
        yj = run_evaluate(ckpt, args.yardstick_splits, out_dir / "yardstick",
                          args.max_labels)
        ym, _ = read_metrics(yj)
        rows.append({"exam": "yardstick", "test_set": args.yardstick_splits, **ym})

    tag = args.tag or ckpt.name
    for r in rows:
        r.update(tag=tag, checkpoint=str(ckpt), task=task,
                 model=meta.get("model", "?"), seed=meta.get("seed"),
                 trained_on=home, loss=meta.get("loss", "?"),
                 home_is_yardstick=home_is_yardstick, leaked=contaminated)

    df = pd.DataFrame(rows)[
        ["tag", "task", "model", "seed", "trained_on", "loss", "exam", "test_set",
         "home_is_yardstick", "leaked", *METRICS, "checkpoint"]]

    csv = out_dir / "two_exams.csv"
    if csv.exists():
        prev = pd.read_csv(csv)
        key = ["tag", "exam"]
        df = (pd.concat([prev, df], ignore_index=True)
                .drop_duplicates(subset=key, keep="last"))
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv, index=False, encoding="utf-8")

    headline = {"binary": "positive_class_f1", "multiclass": "macro_f1_10"}.get(
        task, "macro_f1")
    h = next(r for r in rows if r["exam"] == "home")[headline]
    y = next(r for r in rows if r["exam"] == "yardstick")[headline]
    print(f"\n  {headline}:  home {h if h is None else round(h, 3)}  |  "
          f"yardstick {y if y is None else round(y, 3)}", end="")
    if not home_is_yardstick and h is not None and y is not None:
        print(f"  |  transfer gap {h - y:+.3f}")
    else:
        print()
    print(f"  appended to {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
