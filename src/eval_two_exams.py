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
    python -m src.eval_two_exams --checkpoint checkpoints/foo --out results_RUN2/exp2
    python -m src.eval_two_exams --checkpoint checkpoints/foo --out results_RUN2/exp2 \\
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
from src.train_transformer import TEXT_COL

YARDSTICK = "data/splits"
METRICS = ["weighted_f1", "macro_f1", "macro_f1_10", "micro_f1",
           "positive_class_f1", "no_distortion_f1"]


def recalibrate_on(checkpoint: Path, splits: str):
    """Re-sweep this checkpoint's thresholds on ``splits``' VAL set.

    Writes them into meta.json so the next ``src.evaluate`` call picks them up,
    and stashes the originals under ``thresholds_home`` so
    :func:`restore_thresholds` can put the checkpoint back exactly as it was.

    Val only. Sweeping on test would fit the thresholds to the very rows the
    score is meant to be held out from.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Reuses evaluate.py's own predictor and train_transformer's tokenisation, so
    # the probabilities here are produced exactly as the scoring pass produces
    # them - a threshold swept on differently-tokenised probabilities would not
    # transfer to the pass that uses it.
    from src.evaluate import predict_logits
    from src.train_transformer import (encode_texts, get_labels, load_split,
                                       resolve_device, sweep_thresholds)

    meta_path = checkpoint / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("task") != "multilabel":
        return None

    device = resolve_device("auto")
    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))
    model.to(device).eval()

    val_df = load_split(splits, "val")
    y_val = get_labels(val_df, "multilabel")
    enc, _ = encode_texts(val_df[TEXT_COL], tok, meta.get("max_length", 512),
                          meta.get("truncation", "head"), meta.get("head_keep", 128))
    logits = predict_logits(model, enc, tok.pad_token_id, device)
    probs = 1 / (1 + np.exp(-logits))

    meta.setdefault("thresholds_home", meta.get("thresholds"))
    meta["thresholds"] = sweep_thresholds(probs, y_val)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return meta["thresholds"]


def restore_thresholds(checkpoint: Path) -> None:
    """Undo :func:`recalibrate_on` so the checkpoint matches its own meta."""
    meta_path = checkpoint / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if "thresholds_home" in meta:
        meta["thresholds"] = meta.pop("thresholds_home")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


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
    ap.add_argument("--no-recalibrate", action="store_true",
                    help="report only the zero-shot yardstick number (home "
                         "thresholds applied as-is) and skip re-sweeping on the "
                         "yardstick's val")
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
        # A multilabel checkpoint carries thresholds swept on its HOME val set.
        # Applying those to the yardstick means using cut points fitted to one
        # distribution on another — which understates transfer, because the
        # calibration is wrong rather than the model. So the yardstick pass is
        # scored two ways:
        #
        #   zero_shot   home thresholds applied as-is. "Does it work off the
        #               shelf on our task?" - the strict transfer number.
        #   calibrated  thresholds re-swept on the YARDSTICK's val (never its
        #               test). "Does it work once tuned for our task?" - the
        #               fair architecture comparison.
        #
        # Both are legitimate and they answer different questions, so reporting
        # only one would misrepresent the result either way.
        print(f"[yardstick] {args.yardstick_splits}  (zero-shot: home thresholds)")
        yj = run_evaluate(ckpt, args.yardstick_splits, out_dir / "yardstick",
                          args.max_labels)
        ym, _ = read_metrics(yj)
        rows.append({"exam": "yardstick_zero_shot",
                     "test_set": args.yardstick_splits, **ym})

        if task == "multilabel" and not args.no_recalibrate:
            recal = recalibrate_on(ckpt, args.yardstick_splits)
            if recal is not None:
                print(f"[yardstick] {args.yardstick_splits}  (calibrated: "
                      f"thresholds re-swept on its val)")
                yj2 = run_evaluate(ckpt, args.yardstick_splits,
                                   out_dir / "yardstick_calibrated", args.max_labels)
                ym2, _ = read_metrics(yj2)
                rows.append({"exam": "yardstick_calibrated",
                             "test_set": args.yardstick_splits, **ym2})
                restore_thresholds(ckpt)
        # The headline "yardstick" row is the calibrated one when it exists,
        # since that is the like-for-like comparison; zero-shot stays alongside.
        head = next((r for r in rows if r["exam"] == "yardstick_calibrated"),
                    next(r for r in rows if r["exam"] == "yardstick_zero_shot"))
        rows.append({**head, "exam": "yardstick"})

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
