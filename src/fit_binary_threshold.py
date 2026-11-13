"""
Fit the binary decision threshold on val at a stated false-positive-rate target (task D13).

Picks the lowest threshold on P(distorted) whose val FPR is at or below the target: the
lowest such threshold keeps as much sensitivity as the false-positive budget allows. It is
written into the checkpoint's meta.json as ``binary_threshold``, and src/evaluate.py then
applies it unchanged to val and test. Only val.csv is read here; test.csv stays with
src/evaluate.py.

    python -m src.fit_binary_threshold --checkpoint checkpoints/binary_roberta-base_42 --target-fpr 0.20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.evaluate import load_split, predict_logits, true_labels
from src.train_transformer import build_inputs, encode_texts, resolve_device


def prob_distorted(logits):
    """Softmax probability of class 1 (distorted) from 2-column logits."""
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e[:, 1] / e.sum(axis=1)


def threshold_for_fpr(y_true, p, target_fpr):
    """Lowest threshold t with FPR(p >= t) <= target_fpr; returns (t, fpr, sensitivity)."""
    y_true, p = np.asarray(y_true), np.asarray(p)
    neg = y_true == 0
    if neg.sum() == 0:
        raise ValueError("val has no non-distorted rows; FPR is undefined")
    for t in np.unique(np.concatenate([p, [1.0 + 1e-6]])):
        fpr = float(((p >= t) & neg).sum() / neg.sum())
        if fpr <= target_fpr:
            sens = float(((p >= t) & ~neg).sum() / max((~neg).sum(), 1))
            return float(t), fpr, sens
    raise AssertionError("unreachable: a threshold above every score has FPR 0")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit the binary threshold on val at an FPR target.")
    ap.add_argument("--checkpoint", required=True, help="binary checkpoint dir with meta.json")
    ap.add_argument("--target-fpr", type=float, required=True, help="e.g. 0.20")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint)
    meta_path = ckpt / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["task"] != "binary":
        raise SystemExit(f"{ckpt} is a {meta['task']} checkpoint; this script is for binary only")

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)

    df = load_split(args.splits, "val")
    y_true = true_labels(df, "binary")
    texts, _ = build_inputs(df, meta.get("input_repr", "document"))
    enc, _ = encode_texts(texts, tokenizer, meta["max_length"], meta["truncation"], meta.get("head_keep", 128))
    p = prob_distorted(predict_logits(model, enc, tokenizer.pad_token_id, device, args.batch_size))

    t, fpr, sens = threshold_for_fpr(y_true, p, args.target_fpr)
    meta["binary_threshold"] = t
    meta["binary_threshold_criterion"] = {
        "rule": "lowest threshold on P(distorted) with val FPR <= target",
        "target_fpr": args.target_fpr,
        "val_fpr": round(fpr, 4),
        "val_sensitivity": round(sens, 4),
        "val_n_non_distorted": int((y_true == 0).sum()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[threshold] {meta['model']} seed={meta['seed']}: t={t:.4f} "
          f"val FPR={fpr:.3f} val sensitivity={sens:.3f} (target FPR <= {args.target_fpr})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
