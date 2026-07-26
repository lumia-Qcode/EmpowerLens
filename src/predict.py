"""
Prediction demo: show the model's output next to the ground truth.

Two modes:
  * default — take the first N rows of a split and print, per row, the text,
    the ground-truth label(s), the predicted label(s), and whether they match.
  * --text "..." — predict on one custom paragraph (no ground truth).

Needs a trained checkpoint dir (with meta.json). Inference is quick on CPU.

Usage
-----
    python -m src.predict --checkpoint checkpoints/multilabel_roberta-base_42 --rows 10
    python -m src.predict --checkpoint checkpoints/binary_roberta-base_42 --split test --rows 5
    python -m src.predict --checkpoint checkpoints/multiclass_roberta-base_42 \
        --text "The VC ghosted us so we're obviously not fundable and I'll never raise a round."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DISTORTIONS, MC_CLASSES
from src.metrics import BINARY_CLASSES
from src.evaluate import predict_logits, predictions_from_logits
from src.train_transformer import ML_COLS, TEXT_COL, encode_texts, resolve_device


def _softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def true_str(task, df, i):
    if task == "binary":
        return BINARY_CLASSES[int(df["y_bin"].iloc[i])]
    if task == "multiclass":
        return MC_CLASSES[int(df["y_mc"].iloc[i])]
    on = [DISTORTIONS[j] for j, c in enumerate(ML_COLS) if int(df[c].iloc[i]) == 1]
    return ", ".join(on) if on else "(none / no_distortion)"


def pred_str(task, logits_row, pred_row, thresholds):
    if task == "binary":
        p = _softmax(logits_row[None, :])[0]
        idx = int(np.argmax(logits_row))
        return f"{BINARY_CLASSES[idx]}  (conf {p[idx]:.2f})"
    if task == "multiclass":
        p = _softmax(logits_row[None, :])[0]
        idx = int(np.argmax(logits_row))
        return f"{MC_CLASSES[idx]}  (conf {p[idx]:.2f})"
    probs = 1 / (1 + np.exp(-logits_row))
    on = [(DISTORTIONS[j], probs[j]) for j in range(len(DISTORTIONS)) if pred_row[j] == 1]
    if not on:
        return "(none / no_distortion)"
    return ", ".join(f"{name} ({p:.2f})" for name, p in on)


def match_mark(task, df, i, pred_row, logits_row):
    if task == "binary":
        return "OK" if int(np.argmax(logits_row)) == int(df["y_bin"].iloc[i]) else "X"
    if task == "multiclass":
        return "OK" if int(np.argmax(logits_row)) == int(df["y_mc"].iloc[i]) else "X"
    truth = np.array([int(df[c].iloc[i]) for c in ML_COLS])
    return "OK" if np.array_equal(truth, pred_row) else "~"  # ~ = partial/mismatch


def main(argv=None):
    ap = argparse.ArgumentParser(description="Show model predictions vs ground truth.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--rows", type=int, default=5, help="how many rows of the split to show")
    ap.add_argument("--text", default=None, help="predict on a custom paragraph instead")
    ap.add_argument("--max-labels", type=int, default=2, help="multilabel cap (0 = no cap)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ckpt = Path(args.checkpoint)
    meta = json.loads((ckpt / "meta.json").read_text())
    task, max_length, truncation = meta["task"], meta["max_length"], meta["truncation"]
    thresholds = meta.get("thresholds")
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)

    if args.text:
        texts, df = [args.text], None
    else:
        df = pd.read_csv(f"{args.splits}/{args.split}.csv", encoding="utf-8-sig").head(args.rows)
        texts = df[TEXT_COL].tolist()

    enc, _ = encode_texts(texts, tokenizer, max_length, truncation)
    logits = predict_logits(model, enc, tokenizer.pad_token_id, device)
    preds = predictions_from_logits(logits, task, thresholds, max_labels=args.max_labels)

    print(f"\n=== {task} | checkpoint {ckpt.name} | device {device} ===")
    if task == "multilabel" and thresholds:
        print(f"per-class thresholds: {thresholds}  (predictions capped at {args.max_labels})")
    print()

    for i, text in enumerate(texts):
        snippet = " ".join(str(text).split())
        if len(snippet) > 240:
            snippet = snippet[:240] + " ..."
        print("-" * 80)
        print(f"[{i}] {snippet}")
        if df is not None:
            print(f"    GROUND TRUTH : {true_str(task, df, i)}")
        print(f"    PREDICTED    : {pred_str(task, logits[i], preds[i], thresholds)}")
        if df is not None:
            print(f"    MATCH        : {match_mark(task, df, i, preds[i], logits[i])}")
    print("-" * 80)
    if df is not None:
        print("MATCH legend: OK = exact match, X = wrong (binary/multiclass), "
              "~ = multilabel set not identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
