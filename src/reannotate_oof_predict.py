"""
src/reannotate_oof_predict.py — Stage 1 of model-assisted re-annotation.
(v3 — adds per-class calibration; v2 still over-predicted after its pos_weight fix)

Produces OUT-OF-FOLD multilabel probabilities for every row in
Annotated_data.csv, using K-fold CV over the *entire* corpus (not the frozen
train/val/test split — this is a data-quality tool, not a benchmark run).

Why out-of-fold and not just "predict with the existing checkpoint":
A model scored on rows it trained on will trivially agree with its own
training labels (memorization), which would hide exactly the mislabeled rows
you're trying to find. Every row here is predicted by a model that never saw
it during training — same logic as Northcutt et al.'s confident learning
(2021), which explicitly requires out-of-sample predicted probabilities.

WHAT CHANGED FROM v1 -> v2 (why the first run over-predicted ~4x):
1. `pos_weight` is now clamped (--max-pos-weight, default 10.0). Uncapped
   inverse-frequency weighting on rare classes (e.g. all_or_nothing at ~5%
   positive rate) produces weights near 19x, which combined with few epochs
   and zero monitoring pushed the model to "always guess positive."
2. Each fold now carves a small INTERNAL validation split out of its own
   TRAIN portion (--inner-val-frac, default 0.1) purely for early
   stopping / best-checkpoint selection. The true held-out outer fold
   (the rows we're generating OOF predictions for) is NEVER used for
   checkpoint selection — only for the final prediction pass. This matters:
   selecting a checkpoint based on how well it matches the labels of the
   exact rows you're about to score for "does the model disagree with this
   label" would quietly bias those rows toward looking more trustworthy
   than they are.
3. `eval_strategy="epoch"` + `load_best_model_at_end=True` on the inner
   split, so training that goes off the rails is visible in the printed
   per-epoch macro_f1 instead of only being caught after the fact.
4. Epochs default raised 3 -> 5 (still lower than nothing, but no longer the
   likely cause of undertraining on its own).
5. Optional `--use-distorted-part` concatenates the annotator-highlighted
   `Distorted part` span onto the input text when present. This is a
   genuinely more concentrated signal, but its PRESENCE correlates with
   "this row is distorted" almost by construction (no_distortion rows won't
   have a highlighted span) — so this is a second lever, not a bugfix.
   Keep it OFF (default) until the base fix is confirmed working via the
   smoke test, then A/B it as its own documented experiment.

WHAT CHANGED IN v3 (this version):
v2's fixes addressed genuine undertraining, but a run can still come back
with avg_pos_per_row well above the ~0.8 true rate (e.g. 2.61) even with a
clamped pos_weight and a healthy per-class std. That's because
BCEWithLogitsLoss(pos_weight=...) doesn't just help rare classes get
learned — it systematically shifts the RAW LOGITS upward for whichever
classes got the heaviest weight. The model's *ranking* of examples within a
class can be perfectly fine while its *absolute* probabilities are biased
high across the board — a calibration problem, not a "no signal learned"
problem, and the two failure modes look similar in the top-line sanity
numbers but need different fixes.

v3 fits a per-class 1-D logistic (Platt) recalibration on each fold's
INNER-VAL split (same split used for early stopping, never the outer fold
being scored — so no leakage) and applies it to the outer-fold logits before
converting them to probabilities. This corrects the systematic bias while
leaving the model's relative ranking (what self_confidence/entropy in
Stage 2 actually depend on) intact. Disable with --no-calibrate if you want
the raw sigmoid probs for comparison.

Usage
-----
    # 1. Smoke test first — 2 folds, 1 epoch, verify the sanity numbers
    #    printed at the end before committing to a full run:
    python -m src.reannotate_oof_predict --folds 2 --epochs 1

    # 2. Full run once the smoke test looks sane:
    python -m src.reannotate_oof_predict --folds 5 --epochs 5 --seed 42

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
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
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
    make_compute_metrics,
    pos_weights,
    resolve_device,
)

TEXT_COL = "Patient Question"
DISTORTED_PART_COL = "Distorted part"


def build_texts(df, use_distorted_part: bool):
    if not use_distorted_part:
        return df[TEXT_COL].reset_index(drop=True)
    base = df[TEXT_COL].astype(str)
    span = df[DISTORTED_PART_COL]
    combined = [
        f"{t} [SEP] Distorted part: {s}" if isinstance(s, str) and s.strip() else t
        for t, s in zip(base, span)
    ]
    return __import__("pandas").Series(combined).reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="K-fold OOF prediction for re-annotation triage.")
    ap.add_argument("--path", default="Annotated_data.csv")
    ap.add_argument("--model", default="mental/mental-roberta-base")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5, help="epochs for the inner-val monitored training")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/reannotation")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-pos-weight", type=float, default=10.0,
                     help="cap on BCEWithLogitsLoss pos_weight — prevents rare-class over-prediction")
    ap.add_argument("--inner-val-frac", type=float, default=0.10,
                     help="fraction of each fold's TRAIN portion held out for early stopping AND "
                          "per-class calibration (never used for the actual OOF prediction — avoids "
                          "checkpoint-selection / calibration leakage into the scored rows)")
    ap.add_argument("--use-distorted-part", action="store_true",
                     help="concatenate the annotator-highlighted 'Distorted part' span onto the input text. "
                          "OFF by default — see module docstring for why this is a separate experiment, "
                          "not a bugfix, and correlates with the label by construction.")
    ap.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                     help="disable per-class Platt-scaling calibration and use raw sigmoid probs instead "
                          "(calibration is ON by default — see 'WHAT CHANGED IN v3' in the module docstring)")
    ap.set_defaults(calibrate=True)
    args = ap.parse_args(argv)

    device = resolve_device(args.device)
    set_seed(args.seed)

    df = load_raw(args.path)
    _, _, y_ml = make_targets(df)
    n, n_classes = y_ml.shape
    assert n_classes == len(DISTORTIONS)

    texts = build_texts(df, args.use_distorted_part)
    oof_probs = np.full((n, n_classes), np.nan, dtype=np.float32)
    fold_of_row = np.full(n, -1, dtype=int)

    kf = MultilabelStratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    X_dummy = np.zeros((n, 1))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    compute_metrics = make_compute_metrics("multilabel")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_dummy, y_ml)):
        print(f"\n=== Fold {fold + 1}/{args.folds} — train={len(train_idx)} held_out={len(val_idx)} ===")
        fold_of_row[val_idx] = fold

        # Carve an INNER validation split out of TRAIN only, for early stopping
        # AND for calibration. The outer held-out fold (val_idx) is never
        # touched until the final predict + calibrate pass.
        y_train_full = y_ml[train_idx]
        inner = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=args.inner_val_frac, random_state=args.seed
        )
        inner_tr_local, inner_va_local = next(inner.split(np.zeros((len(train_idx), 1)), y_train_full))
        inner_tr_idx = train_idx[inner_tr_local]
        inner_va_idx = train_idx[inner_va_local]

        y_tr = y_ml[inner_tr_idx]
        y_inner_va = y_ml[inner_va_idx]

        tr_enc, _ = encode_texts(texts.iloc[inner_tr_idx], tokenizer, args.max_length, "head", 128)
        iva_enc, _ = encode_texts(texts.iloc[inner_va_idx], tokenizer, args.max_length, "head", 128)
        va_enc, _ = encode_texts(texts.iloc[val_idx], tokenizer, args.max_length, "head", 128)

        train_ds = TextDataset(tr_enc, list(y_tr.astype(np.float32)))
        inner_val_ds = TextDataset(iva_enc, list(y_inner_va.astype(np.float32)))
        predict_ds = TextDataset(va_enc, list(np.zeros((len(val_idx), n_classes), dtype=np.float32)))

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=n_classes, problem_type="multi_label_classification"
        ).to(device)

        pw = pos_weights(y_tr, device)
        pw = torch.clamp(pw, max=args.max_pos_weight)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

        out_dir = Path(args.out) / f"fold{fold}_ckpt"
        targs = TrainingArguments(
            output_dir=str(out_dir), num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr, warmup_ratio=0.1, weight_decay=0.01,
            eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
            metric_for_best_model="macro_f1", greater_is_better=True, save_total_limit=1,
            fp16=(device == "cuda"), logging_steps=50, report_to="none",
            seed=args.seed, use_cpu=(device == "cpu"),
        )
        trainer = WeightedTrainer(
            model=model, args=targs, train_dataset=train_ds, eval_dataset=inner_val_ds,
            data_collator=functools.partial(collate, pad_id=tokenizer.pad_token_id, multilabel=True),
            compute_metrics=compute_metrics, processing_class=tokenizer, loss_fn=loss_fn,
        )
        trainer.train()
        inner_metrics = trainer.evaluate()
        print(f"[fold {fold}] best inner-val macro_f1 = {inner_metrics.get('eval_macro_f1', float('nan')):.3f} "
              f"(watch this — near 0 or NaN means this fold didn't learn anything usable)")

        # --- per-class Platt-scale calibration -----------------------------
        # pos_weight-reweighted BCE systematically shifts RAW LOGITS upward
        # for heavily-weighted classes, which biases the outer-fold sigmoid
        # probs high even when the model's ranking is fine. Fit a 1-D
        # logistic recalibration per class on the INNER-VAL split (never the
        # outer fold being scored) and apply it to the outer-fold logits.
        with torch.no_grad():
            inner_logits = trainer.predict(inner_val_ds).predictions
            if isinstance(inner_logits, tuple):
                inner_logits = inner_logits[0]
            outer_logits = trainer.predict(predict_ds).predictions
            if isinstance(outer_logits, tuple):
                outer_logits = outer_logits[0]

        calibrated = np.zeros_like(outer_logits, dtype=np.float32)
        n_degenerate = 0
        for c in range(n_classes):
            y_c = y_inner_va[:, c]
            if args.calibrate and len(np.unique(y_c)) == 2:
                lr = LogisticRegression(C=1.0, solver="lbfgs")
                lr.fit(inner_logits[:, c].reshape(-1, 1), y_c)
                calibrated[:, c] = lr.predict_proba(outer_logits[:, c].reshape(-1, 1))[:, 1]
            else:
                if args.calibrate:
                    n_degenerate += 1
                # too few positives (or negatives) in the inner-val split to
                # fit a calibrator for this class this fold, or calibration
                # disabled entirely -> fall back to a plain sigmoid
                calibrated[:, c] = 1 / (1 + np.exp(-outer_logits[:, c]))
        if args.calibrate and n_degenerate:
            print(f"[fold {fold}] {n_degenerate}/{n_classes} classes had a single-label inner-val "
                  f"split (too few positives to calibrate) — used raw sigmoid for those")

        oof_probs[val_idx] = calibrated

        del model, trainer
        if device == "cuda":
            torch.cuda.empty_cache()

    assert not np.isnan(oof_probs).any(), "every row must receive an out-of-fold prediction"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "oof_probs.npy", oof_probs)
    meta = {
        "model": args.model, "folds": args.folds, "epochs": args.epochs,
        "seed": args.seed, "n_rows": int(n), "distortions": DISTORTIONS,
        "max_pos_weight": args.max_pos_weight, "inner_val_frac": args.inner_val_frac,
        "use_distorted_part": args.use_distorted_part,
        "calibrated": args.calibrate,
        "note": "out-of-fold probs; each row predicted by a model that never trained on it "
                "(inner-val split used for early stopping AND per-class Platt calibration is drawn "
                "from TRAIN only, never from the held-out fold being predicted)",
    }
    (out_dir / "oof_meta.json").write_text(json.dumps(meta, indent=2))
    np.save(out_dir / "fold_of_row.npy", fold_of_row)

    # --- sanity check printed immediately, so a bad run is visible before you even open triage.py ---
    avg_pos_per_row = (oof_probs > 0.5).sum(axis=1).mean()
    per_class_std = oof_probs.std(axis=0)
    print("\n" + "=" * 60)
    print("SANITY CHECK (compare against these before trusting the run):")
    print(f"  avg predicted positives/row @0.5 thr : {avg_pos_per_row:.2f}  "
          f"(true corpus average is ~0.8 — anything above ~2 means over-prediction, re-check pos_weight/epochs)")
    print(f"  per-class prob std (min / mean / max): "
          f"{per_class_std.min():.3f} / {per_class_std.mean():.3f} / {per_class_std.max():.3f}  "
          f"(values under ~0.15 mean that class barely varies row-to-row — no real signal learned)")
    print(f"  calibration                          : {'ON' if args.calibrate else 'OFF'}")
    print("=" * 60)
    print(f"\nWrote oof_probs.npy {oof_probs.shape} + oof_meta.json to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
