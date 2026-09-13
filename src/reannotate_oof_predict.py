"""
src/reannotate_oof_predict.py — Stage 1 of model-assisted re-annotation.

Produces OUT-OF-FOLD multilabel probabilities for every row in
Annotated_data.csv, using K-fold CV over the *entire* corpus (not the frozen
train/val/test split — this is a data-quality tool, not a benchmark run).

Why out-of-fold and not just "predict with the existing checkpoint":
A model scored on rows it trained on will trivially agree with its own
training labels (memorization), which would hide exactly the mislabeled rows
you're trying to find. Every row here is predicted by a model that never saw
it during training — same logic as Northcutt et al.'s confident learning
(2021), which explicitly requires out-of-sample predicted probabilities.

This does NOT produce a benchmark-quality model (few epochs, no threshold
sweep, no 3-seed averaging — that machinery is for src/train_transformer.py).
The only artifact that matters here is the probability matrix.

Usage
-----
    python -m src.reannotate_oof_predict --folds 5 --epochs 3 --seed 42
    # writes:
    #   results/reannotation/oof_probs.npy         (N, 10) float32
    #   results/reannotation/oof_meta.json         fold assignment + config
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import DISTORTIONS, load_raw, make_targets
from src.train_transformer import (
    TextDataset,
    WeightedTrainer,
    collate,
    encode_texts,
    pos_weights,
    resolve_device,
)

TEXT_COL = "Patient Question"


def main(argv=None):
    ap = argparse.ArgumentParser(description="K-fold OOF prediction for re-annotation triage.")
    ap.add_argument("--path", default="Annotated_data.csv")
    ap.add_argument("--model", default="mental/mental-roberta-base")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3, help="kept low — this is a triage signal, not a final model")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/reannotation")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    device = resolve_device(args.device)
    set_seed(args.seed)

    df = load_raw(args.path)
    _, _, y_ml = make_targets(df)
    n, n_classes = y_ml.shape
    assert n_classes == len(DISTORTIONS)

    texts = df[TEXT_COL].reset_index(drop=True)
    oof_probs = np.full((n, n_classes), np.nan, dtype=np.float32)
    fold_of_row = np.full(n, -1, dtype=int)

    kf = MultilabelStratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    X_dummy = np.zeros((n, 1))

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_dummy, y_ml)):
        print(f"\n=== Fold {fold + 1}/{args.folds} — train={len(train_idx)} held_out={len(val_idx)} ===")
        fold_of_row[val_idx] = fold

        y_tr = y_ml[train_idx]
        tr_enc, _ = encode_texts(texts.iloc[train_idx], tokenizer, args.max_length, "head", 128)
        va_enc, _ = encode_texts(texts.iloc[val_idx], tokenizer, args.max_length, "head", 128)
        train_ds = TextDataset(tr_enc, list(y_tr.astype(np.float32)))
        val_ds = TextDataset(va_enc, list(np.zeros((len(val_idx), n_classes), dtype=np.float32)))  # labels unused at predict time

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=n_classes, problem_type="multi_label_classification"
        ).to(device)

        pw = pos_weights(y_tr, device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        out_dir = Path(args.out) / f"fold{fold}_ckpt"
        targs = TrainingArguments(
            output_dir=str(out_dir), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr, warmup_ratio=0.1, weight_decay=0.01,
            eval_strategy="no", save_strategy="no", fp16=(device == "cuda"),
            logging_steps=50, report_to="none", seed=args.seed, use_cpu=(device == "cpu"),
        )
        trainer = WeightedTrainer(
            model=model, args=targs, train_dataset=train_ds,
            data_collator=functools.partial(collate, pad_id=tokenizer.pad_token_id, multilabel=True),
            processing_class=tokenizer, loss_fn=loss_fn,
        )
        trainer.train()

        with torch.no_grad():
            logits = trainer.predict(val_ds).predictions
            if isinstance(logits, tuple):
                logits = logits[0]
        probs = 1 / (1 + np.exp(-logits))
        oof_probs[val_idx] = probs.astype(np.float32)

        # free VRAM between folds
        del model, trainer
        torch.cuda.empty_cache() if device == "cuda" else None

    assert not np.isnan(oof_probs).any(), "every row must receive an out-of-fold prediction"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof_probs.npy", oof_probs)
    meta = {
        "model": args.model, "folds": args.folds, "epochs": args.epochs,
        "seed": args.seed, "n_rows": int(n), "distortions": DISTORTIONS,
        "note": "out-of-fold probs; each row predicted by a model that never trained on it",
    }
    (out_dir / "oof_meta.json").write_text(json.dumps(meta, indent=2))
    np.save(out_dir / "fold_of_row.npy", fold_of_row)
    print(f"\nWrote oof_probs.npy {oof_probs.shape} + oof_meta.json to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
