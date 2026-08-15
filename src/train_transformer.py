# =====================================================================
# PRODUCTION-READY: src/train_transformer.py
# =====================================================================
#
# CHANGES vs the original Month-1/2 version (see EmpowerLens_Report.pdf for
# the full before/after writeup):
#   - metric_for_best_model now uses macro_f1_10 for multiclass instead of
#     macro_f1, so checkpoint selection isn't partly rewarded for the easy
#     no_distortion class.
#   - --loss {bce,focal} for multilabel, --focal-gamma to tune it.
#   - --label-smoothing for binary/multiclass CrossEntropyLoss.
#   - --grad-accum for a larger effective batch size on a small physical one.
#   - --lr-scheduler {linear,cosine}.
#   - --dropout to override the classification head's dropout.
#   - --freeze-layers to freeze the bottom N encoder layers (coarse
#     regularization for backbones that overfit, e.g. DeBERTa-v3-base).
#   - --llrd / --llrd-decay for layer-wise learning-rate decay, a softer
#     alternative to --freeze-layers.
#   - --early-stopping-patience to stop wasting epochs once val stops
#     improving, instead of always training the full --epochs.
# All new flags default to the OLD behavior (off / unchanged) so existing
# commands from before this change still run identically unless you opt in.

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import functools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    # transformers 5.x dropped the group_by_length flag; the sampler it used is
    # still available and is wired up manually in WeightedTrainer below.
    TrainingArguments,
    set_seed,
)
from transformers.trainer_pt_utils import LengthGroupedSampler

from src.data import DISTORTIONS, MC_CLASSES
from src.losses import FocalLoss, build_llrd_optimizer, freeze_bottom_layers

TEXT_COL = "Patient Question"
ML_COLS = [f"ml_{d}" for d in DISTORTIONS]
TASK_NUM_LABELS = {"binary": 2, "multiclass": 11, "multilabel": 10}


def resolve_device(choice: str) -> str:
    if choice != "auto":
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_split(splits_dir: str, name: str) -> pd.DataFrame:
    return pd.read_csv(f"{splits_dir}/{name}.csv", encoding="utf-8-sig")


def get_labels(df: pd.DataFrame, task: str):
    if task == "binary":
        return df["y_bin"].to_numpy().astype(np.int64)
    if task == "multiclass":
        return df["y_mc"].to_numpy().astype(np.int64)
    return df[ML_COLS].to_numpy().astype(np.float32)


def _special_token_ids(tokenizer):
    cls = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    sep = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    prefix = [cls] if cls is not None else []
    suffix = [sep] if sep is not None else []
    return prefix, suffix


def encode_texts(texts, tokenizer, max_length: int, strategy: str, head_keep: int):
    prefix, suffix = _special_token_ids(tokenizer)
    budget = max_length - len(prefix) - len(suffix)

    do_head_tail = strategy == "head_tail" and budget > head_keep
    keep_tail = budget - head_keep
    encodings, truncated = [], 0

    string_list = texts.astype(str).tolist()
    all_ids = tokenizer(string_list, add_special_tokens=False)["input_ids"]

    for ids in all_ids:
        was_truncated = len(ids) > budget
        truncated += int(was_truncated)

        if was_truncated:
            if do_head_tail:
                ids = ids[:head_keep] + ids[-keep_tail:]
            else:
                ids = ids[:budget]

        input_ids = prefix + ids + suffix
        encodings.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})

    rate = truncated / max(len(texts), 1)
    return encodings, rate


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, i):
        item = dict(self.encodings[i])
        item["labels"] = self.labels[i]
        return item


def collate(batch, pad_id, multilabel):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, attn = [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        attn.append(b["attention_mask"] + [0] * pad)
    labels = np.array([b["labels"] for b in batch])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.float if multilabel else torch.long),
    }


def class_weights(y, num_labels, device):
    counts = np.bincount(y, minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    w = len(y) / (num_labels * counts)
    return torch.tensor(w, dtype=torch.float, device=device)


def pos_weights(y_ml, device):
    n_pos = y_ml.sum(axis=0)
    n_pos[n_pos == 0] = 1.0
    w = (len(y_ml) - n_pos) / n_pos
    return torch.tensor(w, dtype=torch.float, device=device)


class WeightedTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self, train_dataset=None):
        """Batch similar-length sequences together.

        Token lengths here are very skewed (median 138, p95 482, max 1396), so a
        randomly-drawn batch gets padded to its longest member and wastes roughly
        2.5x the compute on padding. Grouping by length removes most of that.

        transformers 5.x removed the ``group_by_length=True`` TrainingArguments
        flag, so the sampler is wired up by hand here. It only changes which
        examples share a batch — not what the model sees — and stays seeded, so
        runs remain reproducible.
        """
        ds = train_dataset if train_dataset is not None else self.train_dataset
        if ds is None:
            return None
        try:
            # encodings[i] is a dict — len() on it would count keys, not tokens.
            lengths = [len(ds.encodings[i]["input_ids"]) for i in range(len(ds))]
            return LengthGroupedSampler(
                batch_size=self.args.train_batch_size,
                dataset=ds,
                lengths=lengths,
                generator=torch.Generator().manual_seed(self.args.seed),
            )
        except Exception as e:  # noqa: BLE001 - never let an optimisation break training
            print(f"[warn] length-grouped sampling unavailable ({e}); using default sampler")
            return super()._get_train_sampler(train_dataset)


def make_compute_metrics(task):
    def compute(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        if task == "multilabel":
            preds = (1 / (1 + np.exp(-logits)) >= 0.5).astype(int)
            return {
                "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
                "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
            }
        preds = np.argmax(logits, axis=1)
        out = {"macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}
        if task == "multiclass":
            out["macro_f1_10"] = f1_score(labels, preds, labels=list(range(1, 11)), average="macro", zero_division=0)
        return out
    return compute


def sweep_thresholds(probs, y_true):
    grid = np.arange(0.05, 0.96, 0.05)
    thresholds = []
    for j in range(y_true.shape[1]):
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            f = f1_score(y_true[:, j], (probs[:, j] >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, float(t)
        thresholds.append(round(best_t, 2))
    return thresholds


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fine-tune a transformer baseline.")
    ap.add_argument("--task", required=True, choices=["binary", "multiclass", "multilabel"])
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--truncation", choices=["head", "head_tail"], default="head")
    ap.add_argument("--head-keep", type=int, default=128, help="Front tokens kept during head_tail truncation")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--smoke", action="store_true", help="100 rows execution check")
    ap.add_argument("--wandb", action="store_true", help="enable W&B")

    # --- new flags (all default to old behavior) -----------------------
    ap.add_argument("--loss", choices=["bce", "focal"], default="bce",
                    help="multilabel only: BCEWithLogitsLoss (default) or FocalLoss")
    ap.add_argument("--focal-gamma", type=float, default=2.0, help="focal loss gamma, only used with --loss focal")
    ap.add_argument("--label-smoothing", type=float, default=0.0,
                    help="binary/multiclass CrossEntropyLoss label smoothing, 0.0 = off (old behavior)")
    ap.add_argument("--grad-accum", type=int, default=1, help="gradient_accumulation_steps")
    ap.add_argument("--lr-scheduler", choices=["linear", "cosine"], default="linear")
    ap.add_argument("--dropout", type=float, default=None,
                    help="override classifier/hidden dropout prob; default None = model's own config default")
    ap.add_argument("--freeze-layers", type=int, default=0,
                    help="freeze the bottom N encoder layers, 0 = off (old behavior)")
    ap.add_argument("--llrd", action="store_true", help="use layer-wise LR decay optimizer instead of flat --lr")
    ap.add_argument("--llrd-decay", type=float, default=0.9, help="per-layer LR multiplier, only used with --llrd")
    ap.add_argument("--early-stopping-patience", type=int, default=0,
                    help="stop after N eval epochs with no improvement, 0 = off (old behavior: always run --epochs)")
    args = ap.parse_args(argv)

    if args.truncation == "head_tail" and args.max_length <= args.head_keep:
        raise ValueError(f"Configuration Error: --max-length ({args.max_length}) must be > --head-keep ({args.head_keep})")

    if args.wandb:
        if not os.environ.get("WANDB_API_KEY"):
            raise SystemExit("--wandb set but WANDB_API_KEY is not in the environment.")
        report_to = ["wandb"]
    else:
        os.environ["WANDB_DISABLED"] = "true"
        report_to = "none"

    device = resolve_device(args.device)
    set_seed(args.seed)
    num_labels = TASK_NUM_LABELS[args.task]
    multilabel = args.task == "multilabel"

    train_df = load_split(args.splits, "train")
    val_df = load_split(args.splits, "val")
    if args.smoke:
        train_df = train_df.head(100).reset_index(drop=True)
        val_df = val_df.head(100).reset_index(drop=True)
        args.epochs = 1

    y_train = get_labels(train_df, args.task)
    y_val = get_labels(val_df, args.task)

    print(f"[{args.task}] device={device} model={args.model} train={len(train_df)} val={len(val_df)} epochs={args.epochs}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_kwargs = {"num_labels": num_labels}
    if multilabel:
        model_kwargs["problem_type"] = "multi_label_classification"
    if args.dropout is not None:
        # Not every architecture recognizes both of these config fields, but
        # from_pretrained silently ignores kwargs a given config class doesn't
        # define, so passing both is safe across RoBERTa/MentalRoBERTa/DeBERTa.
        model_kwargs["classifier_dropout"] = args.dropout
        model_kwargs["hidden_dropout_prob"] = args.dropout

    model = AutoModelForSequenceClassification.from_pretrained(args.model, **model_kwargs)
    # Some HF checkpoints (DeBERTa-v3 included) ship a config.json torch_dtype hint that
    # loads weights as something other than float32. Under fp16 mixed precision, Trainer's
    # GradScaler expects fp32 master weights and raises "Attempting to unscale FP16
    # gradients" if it doesn't get them. Forcing fp32 here makes every --model behave the
    # same way regardless of what its checkpoint config requested.
    model = model.float()

    if args.freeze_layers > 0:
        n_total = freeze_bottom_layers(model, args.freeze_layers)
        print(f"[freeze] froze bottom {args.freeze_layers}/{n_total} encoder layers")

    model.to(device)

    tr_enc, _ = encode_texts(train_df[TEXT_COL], tokenizer, args.max_length, args.truncation, args.head_keep)
    va_enc, val_trunc_rate = encode_texts(val_df[TEXT_COL], tokenizer, args.max_length, args.truncation, args.head_keep)
    train_ds = TextDataset(tr_enc, list(y_train))
    val_ds = TextDataset(va_enc, list(y_val))

    if multilabel:
        pw = pos_weights(y_train, device)
        if args.loss == "focal":
            loss_fn = FocalLoss(pos_weight=pw, gamma=args.focal_gamma)
            print(f"[loss] FocalLoss(gamma={args.focal_gamma}) with pos_weight from class frequency")
        else:
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        cw = class_weights(y_train, num_labels, device)
        loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)

    run_name = f"{args.task}_{args.model.split('/')[-1]}_{args.seed}" + ("_smoke" if args.smoke else "")
    out_dir = Path(args.out) / run_name

    # Old default was hardcoded to macro_f1 for every task, which for
    # multiclass rewards the model partly for the easy no_distortion class.
    # macro_f1_10 (no_distortion excluded) is the honest selection metric.
    metric_key = "macro_f1_10" if args.task == "multiclass" else "macro_f1"

    targs = TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=args.epochs, per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, learning_rate=args.lr, warmup_ratio=0.1, weight_decay=0.01,
        gradient_accumulation_steps=args.grad_accum, lr_scheduler_type=args.lr_scheduler,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, metric_for_best_model=metric_key,
        greater_is_better=True, save_total_limit=1, fp16=(device == "cuda"), logging_steps=10, report_to=report_to,
        seed=args.seed, use_cpu=(device == "cpu"),
    )

    optimizers = (None, None)
    if args.llrd:
        optimizer = build_llrd_optimizer(model, base_lr=args.lr, decay=args.llrd_decay, head_lr=args.lr)
        optimizers = (optimizer, None)  # Trainer builds the matching scheduler itself
        print(f"[llrd] layer-wise LR decay enabled: base_lr={args.lr} decay={args.llrd_decay}")

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=functools.partial(collate, pad_id=tokenizer.pad_token_id, multilabel=multilabel),
        compute_metrics=make_compute_metrics(args.task), processing_class=tokenizer, loss_fn=loss_fn,
        optimizers=optimizers, callbacks=callbacks or None,
    )

    trainer.train()
    val_metrics = trainer.evaluate()

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    thresholds = None
    if multilabel:
        logits = trainer.predict(val_ds).predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = 1 / (1 + np.exp(-logits))
        thresholds = sweep_thresholds(probs, y_val)
        (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

        best_preds = (probs >= np.array(thresholds)).astype(int)
        val_metrics["eval_macro_f1"] = float(f1_score(y_val, best_preds, average="macro", zero_division=0))
        val_metrics["eval_micro_f1"] = float(f1_score(y_val, best_preds, average="micro", zero_division=0))

    meta = {
        # Record WHICH splits this was trained on. Without it there is no way to
        # tell a checkpoint trained on data/splits from one trained on a leaked
        # or combined dir after the fact — which is exactly the question that
        # matters when results look surprising.
        "splits": args.splits,
        "task": args.task, "model": args.model, "seed": args.seed, "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "max_length": args.max_length, "truncation": args.truncation,
        "head_keep": args.head_keep, "device": device, "smoke": args.smoke, "num_labels": num_labels,
        "loss": args.loss if multilabel else "weighted_ce", "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
        "label_smoothing": args.label_smoothing, "grad_accum": args.grad_accum, "lr_scheduler": args.lr_scheduler,
        "dropout": args.dropout, "freeze_layers": args.freeze_layers, "llrd": args.llrd, "llrd_decay": args.llrd_decay,
        "early_stopping_patience": args.early_stopping_patience,
        "val_truncation_rate": val_trunc_rate,
        "val_metrics": {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
        "thresholds": thresholds,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved to {out_dir}/")
    print(f"  val {metric_key} = {val_metrics.get('eval_' + metric_key, float('nan')):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())