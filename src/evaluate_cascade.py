"""
End-to-end cascade evaluation: Stage 1 (binary) -> Stage 2 (multilabel,
trained distorted-only) -> composed prediction, scored against the FULL
val/test set (including no_distortion rows).

This is the only honest way to know whether the cascade actually beats a
single flat multilabel model: scoring Stage 2 in isolation on distorted-only
inputs hides every false negative Stage 1 makes. Here, any row Stage 1 calls
"not distorted" gets an all-zero multilabel prediction and is scored against
its real labels like everything else — so Stage 1's error rate is baked
into the number you report, not hidden behind Stage 2's better-looking
isolated metrics.

--splits must point at a directory whose val.csv/test.csv are the FULL
splits (e.g. data/splits_combined or data/splits), NOT the Stage 2
distorted-only splits from make_splits_cascade.py — those have no
no_distortion rows left to catch Stage 1's misses against.

Usage
-----
    python -m src.evaluate_cascade \
        --stage1-checkpoint checkpoints/binary_mental-roberta-base_42 \
        --stage2-checkpoint checkpoints/multilabel_mental-roberta-base_42 \
        --splits data/splits_combined --out results_cascade
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import DISTORTIONS
from src.evaluate import load_split, predict_logits, predictions_from_logits
from src.metrics import metric_bundle, per_class_table, upsert_paper_comparison
from src.train_transformer import ML_COLS, TEXT_COL, encode_texts, resolve_device


def _load_model(ckpt_dir: str, device: str):
    ckpt = Path(ckpt_dir)
    meta = json.loads((ckpt / "meta.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)
    return model, tokenizer, meta


def cascade_predict(stage1_dir, stage2_dir, texts, device, batch_size=16, max_labels=2):
    """Returns (y_bin_pred, y_ml_pred, stage1_trunc_rate, stage2_trunc_rate, meta1, meta2)."""
    m1, tok1, meta1 = _load_model(stage1_dir, device)
    enc1, trunc1 = encode_texts(
        pd.Series(texts), tok1, meta1["max_length"], meta1["truncation"], meta1.get("head_keep", 128)
    )
    logits1 = predict_logits(m1, enc1, tok1.pad_token_id, device, batch_size)
    y_bin_pred = predictions_from_logits(logits1, "binary")

    m2, tok2, meta2 = _load_model(stage2_dir, device)
    thresholds = meta2.get("thresholds")
    ml_pred = np.zeros((len(texts), len(DISTORTIONS)), dtype=int)

    distorted_idx = np.where(y_bin_pred == 1)[0]
    trunc2 = 0.0
    if len(distorted_idx):
        sub_texts = [texts[i] for i in distorted_idx]
        enc2, trunc2 = encode_texts(
            pd.Series(sub_texts), tok2, meta2["max_length"], meta2["truncation"], meta2.get("head_keep", 128)
        )
        logits2 = predict_logits(m2, enc2, tok2.pad_token_id, device, batch_size)
        sub_pred = predictions_from_logits(logits2, "multilabel", thresholds, max_labels)
        ml_pred[distorted_idx] = sub_pred

    return y_bin_pred, ml_pred, trunc1, trunc2, meta1, meta2


def main(argv=None):
    ap = argparse.ArgumentParser(description="End-to-end cascade evaluation (Stage 1 -> Stage 2).")
    ap.add_argument("--stage1-checkpoint", required=True)
    ap.add_argument("--stage2-checkpoint", required=True)
    ap.add_argument("--splits", default="data/splits_combined",
                    help="dir with FULL val/test.csv (must include no_distortion rows)")
    ap.add_argument("--out", default="results_cascade")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-labels", type=int, default=2)
    args = ap.parse_args(argv)

    device = resolve_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    stage1_tag = Path(args.stage1_checkpoint).name
    stage2_tag = Path(args.stage2_checkpoint).name
    model_name = f"cascade[{stage1_tag}+{stage2_tag}]"

    rows = []
    eval_json = {
        "stage1_checkpoint": args.stage1_checkpoint,
        "stage2_checkpoint": args.stage2_checkpoint,
        "splits": {},
    }

    for split in ("val", "test"):
        df = load_split(args.splits, split)
        texts = df[TEXT_COL].tolist()
        y_bin_true = df["y_bin"].to_numpy().astype(int)
        y_ml_true = df[ML_COLS].to_numpy().astype(int)

        y_bin_pred, y_ml_pred, trunc1, trunc2, meta1, meta2 = cascade_predict(
            args.stage1_checkpoint, args.stage2_checkpoint, texts, device,
            args.batch_size, args.max_labels,
        )

        seed = meta2.get("seed", meta1.get("seed", ""))
        row_bin = metric_bundle("binary", y_bin_true, y_bin_pred, model_name, seed, split,
                                truncation_rate=round(trunc1, 4))
        row_ml = metric_bundle("multilabel", y_ml_true, y_ml_pred, model_name, seed, split,
                               truncation_rate=round(trunc2, 4))
        rows.extend([row_bin, row_ml])

        pc_ml = per_class_table("multilabel", y_ml_true, y_ml_pred)
        pc_path = out / f"per_class_cascade_multilabel_{stage1_tag}_{stage2_tag}_{split}.csv"
        pc_ml.to_csv(pc_path, index=False)

        eval_json["splits"][split] = {
            "binary_metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row_bin.items()},
            "multilabel_metrics": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in row_ml.items()},
            "multilabel_per_class": pc_ml.to_dict(orient="records"),
            "n_flagged_distorted_by_stage1": int(y_bin_pred.sum()),
            "n_total": int(len(df)),
        }

        print(
            f"[{split}] stage1 positive_class_f1={row_bin['positive_class_f1']:.3f}  "
            f"end-to-end multilabel weighted_f1={row_ml['weighted_f1']:.3f}  "
            f"macro_f1={row_ml['macro_f1']:.3f}"
        )

    upsert_paper_comparison(rows, out / "paper_comparison.csv")
    (out / f"eval_cascade_{stage1_tag}_{stage2_tag}.json").write_text(
        json.dumps(eval_json, indent=2), encoding="utf-8"
    )
    print(f"\nWrote cascade eval JSON, per-class CSVs, and paper_comparison.csv to {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
