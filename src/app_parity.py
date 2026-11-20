"""
D25 parity check: the evaluation code path's prediction for one reflection, to compare with the app.

Uses exactly what src/evaluate.py uses (encode_texts -> predict_logits -> predictions_from_logits with the
D13 binary threshold, the per-type multilabel thresholds and the 2-label cap), on the checkpoints the app
loads (the folder named in docs/model_selection.md, unzipped from the D25 Kaggle run).

    python -m src.app_parity --models ../empowerlens-demo/models/d12 --val-row 0
    python -m src.app_parity --models ../empowerlens-demo/models/d12 --text "One investor passed, so ..."

Only val.csv is read (never test). Prints JSON with the same fields the app's predictor returns.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.data import DISTORTIONS
from src.evaluate import predict_logits, predictions_from_logits
from src.train_transformer import TEXT_COL, encode_texts, load_split, resolve_device


def run(ckpt: Path, texts, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    meta = json.loads((ckpt / "meta.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)
    enc, _ = encode_texts(texts, tok, meta["max_length"], meta["truncation"], meta.get("head_keep", 128))
    return meta, predict_logits(model, enc, tok.pad_token_id, device)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluation-path prediction for the D25 app parity check")
    ap.add_argument("--models", required=True, help="folder with MODEL_CARD.json and the two checkpoints")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--val-row", type=int, help="row number in data/splits/val.csv")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    import pandas as pd
    models = Path(args.models)
    card = json.loads((models / "MODEL_CARD.json").read_text(encoding="utf-8"))
    if args.text is not None:
        text, source = args.text, "custom text"
    else:
        val = load_split("data/splits", "val")
        text, source = str(val[TEXT_COL].iloc[args.val_row]), f"val.csv row {args.val_row} (Id_Number {val['Id_Number'].iloc[args.val_row]})"
    texts = pd.Series([text])
    device = resolve_device(args.device)

    bmeta, blog = run(models / card["binary"]["checkpoint"], texts, device)
    z = blog - blog.max(axis=1, keepdims=True)
    p = float(np.exp(z[0, 1]) / np.exp(z[0]).sum())
    is_dist = bool(predictions_from_logits(blog, "binary", binary_threshold=bmeta["binary_threshold"])[0])

    mmeta, mlog = run(models / card["multilabel"]["checkpoint"], texts, device)
    probs = 1 / (1 + np.exp(-mlog[0]))
    preds = predictions_from_logits(mlog, "multilabel", mmeta["thresholds"], max_labels=card["multilabel"]["max_labels"])[0]

    out = {
        "source": source, "text": text, "model_version": card["model_version"],
        "is_distorted": is_dist, "p_distorted": round(p, 4), "binary_threshold": bmeta["binary_threshold"],
        "label_scores": {d: round(float(probs[j]), 4) for j, d in enumerate(DISTORTIONS)},
        "multilabel_predicted": [d for j, d in enumerate(DISTORTIONS) if preds[j]],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
