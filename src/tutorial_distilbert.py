"""
EmpowerLens — the wellally.tech DistilBERT tutorial, run on the real dataset.

Source recipe
-------------
https://www.wellally.tech/blog/python-cognitive-distortion-transformer-tutorial

That tutorial fine-tunes ``distilbert-base-uncased`` as a **multi-label**
classifier over 10 cognitive distortions, with:

    learning_rate = 2e-5      per_device_train_batch_size = 8
    num_train_epochs = 10     weight_decay = 0.01
    problem_type = "multi_label_classification"
    eval each epoch, load_best_model_at_end
    sigmoid + fixed 0.5 threshold
    metrics: micro-F1, ROC-AUC, accuracy

Every one of those hyperparameters is reproduced here verbatim (they are the
defaults of this script's CLI). What is deliberately **not** reproduced:

1. **The data.** The tutorial trains on a hand-written 15-row toy CSV, which
   cannot produce a meaningful number. We train on ``Annotated_data.csv``
   through the project's frozen splits (``data/splits/{train,val,test}.csv``,
   2,024 / 253 / 253 rows). The multi-label target is the existing ``ml_*``
   columns: the union of the dominant and optional secondary distortion, with
   an all-zero row meaning "No Distortion".
2. **The split.** The tutorial calls ``dataset.train_test_split(test_size=0.2)``
   on the fly, which reshuffles on every run and makes results incomparable.
   Project rule: splits are immutable and read from disk. Model selection and
   every number this script prints come from **val**; ``test.csv`` is never
   opened here — run ``src.evaluate`` on the saved checkpoint for that.
3. **Dead API calls.** The tutorial targets transformers 4.x. On the installed
   5.x stack, ``evaluation_strategy`` is now ``eval_strategy``, ``tokenizer=``
   on Trainer is now ``processing_class=``, and ``return_all_scores=True`` on
   the pipeline is now ``top_k=None``. Fixed.
4. **ROC-AUC on binarized predictions.** The tutorial passes its 0/1
   thresholded predictions to ``roc_auc_score``, which throws away the ranking
   the metric exists to measure. We report ``roc_auc_micro`` on the sigmoid
   probabilities (the meaningful number) and keep the tutorial's binarized
   version alongside it as ``roc_auc_micro_tutorial`` so the two are
   comparable.
5. **"accuracy" as a headline.** ``accuracy_score`` on a multi-label matrix is
   *exact subset match* — every one of 10 columns correct — so it is reported
   as ``subset_accuracy`` and never as the headline. Headline is F1.

Loss and threshold: the two knobs, measured separately
------------------------------------------------------
The default ``--loss bce`` is plain ``BCEWithLogitsLoss``, exactly as the
tutorial has it — no ``pos_weight``. Every loss here is **sigmoid**-based and
scores the 10 labels independently; softmax would make them compete for one
probability budget, which is wrong when 329 of 2,024 training rows carry two
distortions at once. The model emits raw logits with no activation layer: the
sigmoid is fused inside ``BCEWithLogitsLoss`` during training (for numerical
stability) and applied explicitly at eval. Adding a ``nn.Sigmoid()`` to the
model would double-apply it and silently break training.

``--loss`` selects how the negatives are stopped from drowning the positives:

===========  ===============================================================
``bce``      unweighted. The tutorial's. On a 5%-positive class ~95% of the
             gradient says "no", so probabilities collapse below 0.5.
``pos_bce``  + ``pos_weight = negatives/positives`` per class (19.0x for
             all_or_nothing). What ``src.train_transformer`` uses.
``focal``    FocalLoss(gamma) on top of pos_weight — also damps *easy*
             examples, not just frequent ones.
``asl``      AsymmetricLoss: separate gammas for positives and negatives plus
             probability clipping. Replaces pos_weight rather than stacking.
===========  ===============================================================

Independently of the loss, every run is scored **twice on the same
probabilities**: at the tutorial's flat 0.5, and at per-class thresholds swept
on val by ``src.train_transformer.sweep_thresholds``. That separation matters —
a loss change alters what the model learned, a threshold change only moves the
decision line. ``--ablation`` runs all four losses and writes
``ablation_summary.csv`` with all eight rows.

Usage
-----
    venv\\Scripts\\python.exe -m src.tutorial_distilbert --smoke        # 2 min sanity check
    venv\\Scripts\\python.exe -m src.tutorial_distilbert                # tutorial as-is
    venv\\Scripts\\python.exe -m src.tutorial_distilbert --loss pos_bce
    venv\\Scripts\\python.exe -m src.tutorial_distilbert --ablation --seeds 42,1337,2024

Outputs land in ``results_tutorial_distilbert/`` (metrics JSON, per-class CSVs,
seed summary, learning curve, F1 bar chart, demo predictions) and checkpoints
in ``checkpoints/tutorial_<model>_<seed>/`` with a ``meta.json`` that
``src.evaluate`` can consume for the test-set numbers.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

# Windows + torch CPU hits an OpenMP double-load crash (0xC0000005) unless this
# is set before torch is imported. Same guard as src/train_transformer.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss,
                             precision_score, recall_score, roc_auc_score)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import DISTORTIONS, MC_CLASSES, TEXT_COL
from src.losses import AsymmetricLoss, FocalLoss, MaskedBCEWithLogitsLoss
from src.metrics import BINARY_CLASSES, per_class_table
# Imported, not reimplemented, so the threshold search here is bit-for-bit the
# same procedure the project pipeline uses — otherwise the comparison would be
# measuring two different sweeps as well as two different losses.
from src.train_transformer import sweep_thresholds

ML_COLS = [f"ml_{d}" for d in DISTORTIONS]

# The tutorial is a multi-label recipe, but the same backbone and the same
# frozen splits answer all three of the project's questions. Running them
# together is what lets one table say "this model, this split, this task".
TASK_NUM_LABELS = {"binary": 2, "multiclass": 11, "multilabel": 10}
TASK_LABEL_COL = {"binary": "y_bin", "multiclass": "y_mc", "multilabel": ML_COLS}

# Which imbalance treatments make sense per task. Multi-label uses independent
# sigmoids, so the knob is per-label pos_weight / focal / asymmetric. Binary and
# multiclass use a single softmax, so the equivalent knob is a class weight
# vector on the cross-entropy — there is no per-label threshold to sweep.
TASK_LOSSES = {
    "multilabel": ["bce", "pos_bce", "focal", "asl"],
    "binary": ["ce", "weighted_ce"],
    "multiclass": ["ce", "weighted_ce"],
}
# The loss each task's "as published" baseline uses.
TASK_DEFAULT_LOSS = {"multilabel": "bce", "binary": "ce", "multiclass": "ce"}

# A row carries a dominant distortion plus at most one optional secondary, so no
# row in the corpus has more than 2 labels. Measured on the frozen splits:
#   train  0:746  1:949  2:329      val  0:95  1:114  2:44
#   test   0:92   1:118  2:43
# src/evaluate.py already caps test predictions at 2 for this reason; the demo
# uses the same cap so what it prints matches what is actually scored.
MAX_LABELS_PER_ROW = 2

# id2label / label2id, exactly the mapping the tutorial builds from its CSV
# header — here it comes from the canonical DISTORTIONS order instead.
ID2LABEL = {i: name for i, name in enumerate(DISTORTIONS)}
LABEL2ID = {name: i for i, name in enumerate(DISTORTIONS)}

# The tutorial's three demo sentences, kept verbatim so the inference section
# reproduces its final output.
DEMO_TEXTS = [
    "I can't believe I made such a stupid mistake. I'm a complete failure.",
    "I'm sure they are all talking about how bad my presentation was.",
    "This is a great achievement, but I was just lucky.",
]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_split(splits_dir: str, name: str) -> pd.DataFrame:
    """Read one frozen split. Never generates or reshuffles anything."""
    return pd.read_csv(f"{splits_dir}/{name}.csv", encoding="utf-8-sig")


def get_labels(df: pd.DataFrame, task: str = "multilabel") -> np.ndarray:
    """Labels for one task.

    multilabel -> (N, 10) float matrix, the tutorial's ``df['labels']`` column
    binary     -> (N,) int, 1 if any distortion
    multiclass -> (N,) int in 0..10, index 0 is no_distortion
    """
    want = TASK_LABEL_COL[task]
    cols = want if isinstance(want, list) else [want]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Split file is missing column(s) {missing} needed for task "
            f"{task!r}. Regenerate with `python -m src.make_splits` (needs --force)."
        )
    if task == "multilabel":
        return df[ML_COLS].to_numpy().astype(np.float32)
    return df[want].to_numpy().astype(np.int64)


def encode_texts(texts, tokenizer, max_length: int):
    """Tokenize with plain head truncation — the tutorial's ``truncation=True``.

    Returns the encodings plus the fraction of rows that had to be cut, which
    is worth knowing: this corpus has a long right tail (p95 ~482 tokens).
    """
    ids_batch = tokenizer(texts.astype(str).tolist(), truncation=True,
                          max_length=max_length)["input_ids"]
    full = tokenizer(texts.astype(str).tolist(), add_special_tokens=True)["input_ids"]
    truncated = sum(1 for a, b in zip(ids_batch, full) if len(b) > len(a))
    encodings = [{"input_ids": ids, "attention_mask": [1] * len(ids)} for ids in ids_batch]
    return encodings, truncated / max(len(ids_batch), 1)


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


def collate(batch, pad_id, multilabel: bool = True):
    """Dynamic padding to the longest member of the batch.

    Written by hand rather than using ``DataCollatorWithPadding`` because that
    collator routes ``labels`` through ``tokenizer.pad``, which does not know
    what to do with a 10-wide float vector per row.

    Label dtype is task-dependent and not optional: BCE wants float targets,
    CrossEntropy wants int64 class indices and raises on floats.
    """
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, attn, labels = [], [], []
    for b in batch:
        pad = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        attn.append(b["attention_mask"] + [0] * pad)
        labels.append(b["labels"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(np.array(labels),
                               dtype=torch.float if multilabel else torch.long),
    }


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def sigmoid(x):
    """NumPy sigmoid, for turning eval logits into per-label probabilities.

    Sigmoid, never softmax. Softmax would force the ten distortion scores to
    sum to 1, i.e. make the labels compete — but a row here can legitimately
    carry two distortions at once (329 of 2,024 training rows do), and 746 carry
    none. Sigmoid scores each label independently on 0..1, which is what
    multi-label needs. The model itself emits raw logits with no activation
    layer; the sigmoid lives in the loss during training (BCEWithLogitsLoss
    fuses it for numerical stability) and here at inference.
    """
    return 1.0 / (1.0 + np.exp(-x))


def pos_weights(y_ml: np.ndarray) -> np.ndarray:
    """Per-class ``negatives / positives`` ratio, computed on TRAIN only.

    This is the number BCEWithLogitsLoss multiplies the positive term by, so a
    class the corpus rarely marks still contributes gradient comparable to the
    flood of negatives. all_or_nothing has 101 positives in 2,024 train rows,
    so its weight is 1923/101 = 19.0 -- one missed positive costs as much as 19
    false alarms.
    """
    n_pos = y_ml.sum(axis=0)
    n_pos[n_pos == 0] = 1.0          # never divide by zero on an absent label
    return (len(y_ml) - n_pos) / n_pos


def class_weights(y: np.ndarray, num_labels: int) -> np.ndarray:
    """Inverse-frequency weight per class, for single-label cross-entropy.

    The softmax analogue of pos_weight. ``n / (k * count_c)`` gives every class
    the same total say regardless of how many rows carry it; a class holding
    exactly 1/k of the data gets weight 1.0. Computed on TRAIN only.
    """
    counts = np.bincount(y, minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    return len(y) / (num_labels * counts)


def build_loss(name: str, y_train: np.ndarray, device: str, focal_gamma: float,
               task: str = "multilabel"):
    """Map ``--loss`` onto a module, plus a one-line description for the log.

    Multi-label losses are sigmoid-based and score 10 independent labels; they
    differ only in how they stop the negatives from drowning the positives.
    Binary/multiclass losses are softmax-based over mutually exclusive classes,
    where the equivalent knob is a per-class weight on the cross-entropy.
    """
    if task != "multilabel":
        k = TASK_NUM_LABELS[task]
        if name == "ce":
            return nn.CrossEntropyLoss(), "plain CrossEntropyLoss, unweighted"
        if name == "weighted_ce":
            cw = class_weights(y_train, k)
            return (nn.CrossEntropyLoss(
                        weight=torch.tensor(cw, dtype=torch.float, device=device)),
                    f"CrossEntropyLoss + class weights {np.round(cw, 2).tolist()}")
        raise ValueError(f"loss {name!r} is not defined for task {task!r} "
                         f"(valid: {', '.join(TASK_LOSSES[task])})")

    pw = torch.tensor(pos_weights(y_train), dtype=torch.float, device=device)
    if name == "bce":
        # Mathematically identical to what HF applies internally for
        # problem_type="multi_label_classification" -- verified in tests. Routed
        # through the same class as the others so the ablation differs in one
        # thing only: the loss argument, not the training loop.
        return MaskedBCEWithLogitsLoss(), "plain BCEWithLogitsLoss, unweighted (the tutorial's loss)"
    if name == "pos_bce":
        return (MaskedBCEWithLogitsLoss(pos_weight=pw),
                f"BCEWithLogitsLoss + pos_weight {np.round(pos_weights(y_train), 1).tolist()}")
    if name == "focal":
        return (FocalLoss(pos_weight=pw, gamma=focal_gamma),
                f"FocalLoss(gamma={focal_gamma}) on top of pos_weight")
    if name == "asl":
        return AsymmetricLoss(), "AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05), no pos_weight"
    raise ValueError(f"unknown loss {name!r}")


class CustomLossTrainer(Trainer):
    """Trainer that applies our own loss instead of the model's built-in one."""

    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def _safe_auc(y_true, score, **kw):
    """roc_auc_score raises if a column is single-class; return NaN instead."""
    try:
        return float(roc_auc_score(y_true, score, **kw))
    except ValueError:
        return float("nan")


def softmax(x):
    """Row-wise softmax, for the two single-label tasks.

    Correct HERE and wrong for multilabel: softmax forces one row's class scores
    to sum to 1, i.e. the classes compete for a fixed budget. That is exactly
    right for binary (distorted / not) and for the 11-way multiclass, where a
    row has precisely one dominant label — and exactly wrong for multilabel,
    where a row can carry two distortions at once.
    """
    z = x - x.max(axis=1, keepdims=True)      # shift for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def singlelabel_metrics(task: str, y_true, probs) -> dict:
    """Metric bundle for binary / multiclass.

    Shares the keys ``macro_f1``, ``micro_f1``, ``weighted_f1``, ``accuracy``
    and ``roc_auc`` with :func:`multilabel_metrics` so one results table can
    hold all three tasks, then adds what only makes sense per task.
    """
    y_pred = probs.argmax(axis=1)
    out = {
        # Precision and recall are reported alongside F1, not folded into it:
        # F1 alone cannot distinguish "predicts almost nothing, but correctly"
        # from "fires constantly and is often wrong", and those need opposite
        # fixes. On this corpus the first is the actual failure mode.
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if task == "binary":
        out["positive_class_f1"] = float(
            f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0))
        out["roc_auc"] = _safe_auc(y_true, probs[:, 1])
    else:
        # macro over the ten DISTORTION classes only, no_distortion dropped —
        # the honest headline, since no_distortion is the easy majority class.
        out["macro_f1_10"] = float(f1_score(
            y_true, y_pred, labels=list(range(1, 11)), average="macro", zero_division=0))
        per = f1_score(y_true, y_pred, labels=list(range(11)),
                       average=None, zero_division=0)
        out["no_distortion_f1"] = float(per[0])
        out["roc_auc"] = _safe_auc(y_true, probs, multi_class="ovr", average="macro")
    return out


def multilabel_metrics(y_true, probs, threshold: float) -> dict:
    """The tutorial's metric set, plus the ones the project actually reports."""
    y_pred = (probs >= threshold).astype(int)
    return {
        # Precision/recall before F1: under-firing (high precision, near-zero
        # recall) and spraying (the reverse) both land at a low F1, but need
        # opposite fixes. The pair says which one is happening.
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        # headline
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "samples_f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        # ranking quality, independent of where the threshold sits
        "roc_auc_micro": _safe_auc(y_true, probs, average="micro"),
        "roc_auc_macro": _safe_auc(y_true, probs, average="macro"),
        # the tutorial's own (weaker) version: AUC over binarized predictions
        "roc_auc_micro_tutorial": _safe_auc(y_true, y_pred, average="micro"),
        # accuracy_score on a label matrix = all 10 columns right at once
        "subset_accuracy": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        # sanity: how many labels the model is willing to fire at all
        "mean_labels_predicted": float(y_pred.sum(axis=1).mean()),
        "mean_labels_true": float(y_true.sum(axis=1).mean()),
        # Aliases so binary / multiclass / multilabel rows share column names in
        # the results table. On a label matrix, accuracy IS exact subset match.
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": _safe_auc(y_true, probs, average="micro"),
    }


def make_compute_metrics(threshold: float, task: str = "multilabel"):
    """Per-epoch metrics for the Trainer (drives load_best_model_at_end)."""
    def compute(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        if task == "multilabel":
            m = multilabel_metrics(labels, sigmoid(logits), threshold)
            keys = ("micro_f1", "macro_f1", "roc_auc_micro", "subset_accuracy")
        else:
            m = singlelabel_metrics(task, labels, softmax(logits))
            keys = (("micro_f1", "macro_f1", "accuracy", "positive_class_f1")
                    if task == "binary"
                    else ("micro_f1", "macro_f1", "accuracy", "macro_f1_10"))
        return {k: m[k] for k in keys if k in m}
    return compute


def selection_metric(task: str) -> str:
    """Which val metric picks the best epoch.

    Not micro-F1 for every task. On multiclass, micro-F1 equals accuracy and is
    dominated by ``no_distortion`` (36.9% of rows), so it would reward the model
    for the one class that is easy; ``macro_f1_10`` excludes it. On binary the
    minority-but-important question is "did it catch the distortion", so the
    positive class F1 leads.
    """
    return {"multilabel": "micro_f1",
            "binary": "positive_class_f1",
            "multiclass": "macro_f1_10"}[task]


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------

def run_one_seed(args, seed: int, out_root: Path) -> dict:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    train_df = load_split(args.splits, "train")
    val_df = load_split(args.splits, "val")
    epochs = args.epochs
    if args.smoke:
        train_df = train_df.head(64).reset_index(drop=True)
        val_df = val_df.head(64).reset_index(drop=True)
        epochs = 1

    task = args.task
    multilabel = task == "multilabel"
    num_labels = TASK_NUM_LABELS[task]
    y_train, y_val = get_labels(train_df, task), get_labels(val_df, task)

    print(f"\n{'=' * 70}\n[tutorial-distilbert] task={task} seed={seed} "
          f"device={device} model={args.model} loss={args.loss}\n"
          f"  splits={args.splits} train={len(train_df)} val={len(val_df)} "
          f"classes={num_labels} epochs={epochs} lr={args.lr} "
          f"bs={args.batch_size} max_len={args.max_length}"
          f"{f' threshold={args.threshold}' if multilabel else ''}\n{'=' * 70}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    id2label = (ID2LABEL if multilabel else
                dict(enumerate(MC_CLASSES if task == "multiclass"
                               else ["no_distortion", "distorted"])))
    model_kwargs = {"num_labels": num_labels,
                    "id2label": id2label,
                    "label2id": {v: k for k, v in id2label.items()}}
    if multilabel:
        # Selects HF's BCEWithLogitsLoss path. Omitted for binary/multiclass,
        # where the head is a single softmax over mutually exclusive classes.
        model_kwargs["problem_type"] = "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, **model_kwargs).float()
    model.to(device)

    tr_enc, _ = encode_texts(train_df[TEXT_COL], tokenizer, args.max_length)
    va_enc, val_trunc_rate = encode_texts(val_df[TEXT_COL], tokenizer, args.max_length)
    train_ds = TextDataset(tr_enc, list(y_train))
    val_ds = TextDataset(va_enc, list(y_val))
    print(f"  val rows truncated at {args.max_length} tokens: {val_trunc_rate:.1%}")

    loss_fn, loss_desc = build_loss(args.loss, y_train, device, args.focal_gamma, task)
    print(f"  loss: {loss_desc}")

    # Task and loss are both part of the run identity: without them an ablation's
    # arms — and the three tasks — would overwrite each other's checkpoints.
    suffix = ("" if args.loss == TASK_DEFAULT_LOSS[task] else f"_{args.loss}") \
             + ("_smoke" if args.smoke else "")
    run_name = f"tutorial_{task}_{args.model.split('/')[-1]}_{seed}{suffix}"
    ckpt_dir = Path(args.checkpoints) / run_name

    targs = TrainingArguments(
        output_dir=str(ckpt_dir),
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",          # transformers 5.x name
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=selection_metric(task),
        greater_is_better=True,
        save_total_limit=1,
        fp16=(device == "cuda"),
        use_cpu=(device == "cpu"),
        # Per-step bars are the bulk of the output; epoch metrics survive both
        # ways. logging_steps=0 is rejected by TrainingArguments, so quiet mode
        # switches the strategy to per-epoch rather than trying to disable it.
        disable_tqdm=args.quiet,
        logging_strategy=("epoch" if args.quiet else "steps"),
        logging_steps=25,
        report_to="none",
        seed=seed,
    )

    trainer = CustomLossTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=functools.partial(collate, pad_id=tokenizer.pad_token_id,
                                        multilabel=multilabel),
        compute_metrics=make_compute_metrics(args.threshold, task),
        processing_class=tokenizer,     # transformers 5.x name for tokenizer=
        loss_fn=loss_fn,
    )

    trainer.train()

    # Final val pass with the best-epoch weights restored.
    logits = trainer.predict(val_ds).predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    out_root.mkdir(parents=True, exist_ok=True)
    tag = f"{task}_{args.model.split('/')[-1]}_{seed}{suffix}"

    if multilabel:
        probs = sigmoid(logits)
        # Score the SAME probabilities two ways, so loss and threshold can be
        # told apart. "fixed" is the tutorial's flat 0.5; "tuned" is a per-class
        # cut point chosen on val. No retraining — only where the line sits.
        metrics = multilabel_metrics(y_val, probs, args.threshold)
        tuned_thresholds = sweep_thresholds(probs, y_val)
        metrics_tuned = multilabel_metrics(y_val, probs, np.array(tuned_thresholds))
        y_pred = (probs >= args.threshold).astype(int)

        pc_tuned = per_class_table(task, y_val,
                                   (probs >= np.array(tuned_thresholds)).astype(int))
        pc_tuned.insert(0, "seed", seed)
        pc_tuned.insert(1, "threshold", tuned_thresholds)
        pc_tuned.to_csv(out_root / f"per_class_val_tuned_{tag}.csv", index=False)
    else:
        # Single-label: argmax picks the winner, so there is no threshold to
        # tune and no second scoring. metrics_tuned stays None rather than
        # silently duplicating the fixed row.
        probs = softmax(logits)
        metrics = singlelabel_metrics(task, y_val, probs)
        metrics_tuned, tuned_thresholds = None, None
        y_pred = probs.argmax(axis=1)

    per_class = per_class_table(task, y_val, y_pred)
    per_class.insert(0, "seed", seed)
    per_class.to_csv(out_root / f"per_class_val_{tag}.csv", index=False)

    # Per-epoch learning curve, so a run that peaked at epoch 3 and then
    # overfit for seven more is visible rather than hidden behind one number.
    sel = f"eval_{selection_metric(task)}"
    history = [h for h in trainer.state.log_history if sel in h]
    pd.DataFrame(history).to_csv(out_root / f"epoch_history_{tag}.csv", index=False)

    record = {
        "model": args.model, "seed": seed, "split": "val", "task": task,
        "tag": tag, "suffix": suffix, "splits": args.splits,
        "num_labels": num_labels, "selection_metric": selection_metric(task),
        "source": "wellally_tutorial", "epochs": epochs, "lr": args.lr,
        "batch_size": args.batch_size, "weight_decay": args.weight_decay,
        "max_length": args.max_length, "threshold": args.threshold,
        "loss": args.loss, "loss_desc": loss_desc,
        "device": device, "smoke": args.smoke,
        "val_truncation_rate": val_trunc_rate,
        "n_train": len(train_df), "n_val": len(val_df),
        "metrics": metrics,                     # at the flat 0.5 threshold
        "metrics_tuned": metrics_tuned,         # per-class thresholds, swept on val
        "tuned_thresholds": tuned_thresholds,
    }
    (out_root / f"val_metrics_{tag}.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")

    # Save weights + a meta.json shaped for src/evaluate.py, so the test-set
    # numbers come out of the one module allowed to read test.csv.
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))
    (ckpt_dir / "meta.json").write_text(json.dumps({
        "splits": args.splits, "task": task,
        "model": (f"tutorial-{args.model.split('/')[-1]}"
                  + ("" if args.loss == TASK_DEFAULT_LOSS[task] else f"-{args.loss}")),
        "init_from": args.model,
        "seed": seed, "epochs": epochs, "lr": args.lr,
        "batch_size": args.batch_size, "max_length": args.max_length,
        "truncation": "head", "head_keep": 128, "device": device,
        "smoke": args.smoke, "num_labels": num_labels,
        "loss": args.loss, "val_truncation_rate": val_trunc_rate,
        "val_metrics": {f"eval_{k}": v for k, v in metrics.items()},
        # What src/evaluate.py will apply on test. Default "fixed" keeps the
        # tutorial honest (flat 0.5, no tuning); "tuned" uses the per-class cut
        # points swept on val, which is what you would actually deploy.
        # None for binary/multiclass: argmax has no threshold, and evaluate.py
        # only consults this field for multilabel checkpoints.
        "thresholds": (None if not multilabel else
                       (tuned_thresholds if args.meta_thresholds == "tuned"
                        else [args.threshold] * num_labels)),
        "thresholds_fixed": [args.threshold] * num_labels if multilabel else None,
        "thresholds_tuned": tuned_thresholds,
    }, indent=2), encoding="utf-8")

    print(f"\n  [{task} seed {seed}] loss={args.loss}")
    if multilabel:
        print(f"    @ fixed {args.threshold}  micro-F1 {metrics['micro_f1']:.3f} | "
              f"macro-F1 {metrics['macro_f1']:.3f} | "
              f"ROC-AUC {metrics['roc_auc']:.3f} | "
              f"labels/row {metrics['mean_labels_predicted']:.2f}")
        print(f"    @ tuned      micro-F1 {metrics_tuned['micro_f1']:.3f} | "
              f"macro-F1 {metrics_tuned['macro_f1']:.3f} | "
              f"labels/row {metrics_tuned['mean_labels_predicted']:.2f}")
        print(f"    tuned thresholds: {tuned_thresholds}")
    else:
        extra = ("positive-class F1 " + f"{metrics['positive_class_f1']:.3f}"
                 if task == "binary" else
                 "macro-F1(10) " + f"{metrics['macro_f1_10']:.3f}")
        print(f"    macro-F1 {metrics['macro_f1']:.3f} | "
              f"weighted-F1 {metrics['weighted_f1']:.3f} | "
              f"accuracy {metrics['accuracy']:.3f} | "
              f"ROC-AUC {metrics['roc_auc']:.3f} | {extra}")
    print(f"  checkpoint -> {ckpt_dir}")

    del trainer, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return record


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _summary_frame(rows) -> pd.DataFrame:
    """mean ± std across seeds (project convention #6)."""
    df = pd.DataFrame(rows).set_index("seed")
    summary = pd.DataFrame({"mean": df.mean(), "std": df.std(ddof=0)})
    summary["mean_pm_std"] = [f"{m:.3f} +/- {s:.3f}" for m, s in
                              zip(summary["mean"], summary["std"])]
    summary.index.name = "metric"
    return df, summary


def class_names(task: str):
    """Row order for the per-class table, per task."""
    return {"multilabel": DISTORTIONS, "multiclass": MC_CLASSES,
            "binary": BINARY_CLASSES}[task]


def summarize(records, out_root: Path, threshold: float) -> pd.DataFrame:
    """Roll one (task, loss)'s seeds into summary CSVs.

    File names carry BOTH the task and the loss, so three tasks x four losses
    land side by side in one results dir instead of overwriting each other.
    Multi-label additionally gets the swept-threshold scoring; binary and
    multiclass have no threshold to sweep, so those files are simply absent.
    """
    task = records[0]["task"]
    key = f"{task}{records[0]['suffix']}"

    df, summary = _summary_frame([{"seed": r["seed"], **r["metrics"]} for r in records])
    df.to_csv(out_root / f"val_metrics_per_seed_{key}.csv")
    summary.to_csv(out_root / f"val_summary_{key}.csv")

    if records[0].get("metrics_tuned"):
        df_t, summary_t = _summary_frame(
            [{"seed": r["seed"], **r["metrics_tuned"]} for r in records])
        df_t.to_csv(out_root / f"val_metrics_per_seed_tuned_{key}.csv")
        summary_t.to_csv(out_root / f"val_summary_tuned_{key}.csv")

    # Per-class F1 averaged over seeds — where the model actually fails. Read
    # the exact files this run produced rather than globbing, so a directory
    # holding several tasks and losses cannot cross-contaminate the average.
    pcs = [pd.read_csv(out_root / f"per_class_val_{r['tag']}.csv") for r in records]
    pc = pd.concat(pcs)
    pc_mean = (pc.groupby("class")[["precision", "recall", "f1", "support"]]
               .mean().reindex(class_names(task)))
    pc_mean.to_csv(out_root / f"per_class_val_mean_{key}.csv")
    _plot(pc_mean, out_root, threshold, key, records[0]["loss"], task)

    # The same table at the swept thresholds. Without this, the only per-class
    # view is the flat-0.5 one, which on an under-firing model is all zeros —
    # and it looks like the threshold sweep never happened, when in fact it did
    # and its numbers were simply never averaged or shown.
    tuned_files = [out_root / f"per_class_val_tuned_{r['tag']}.csv" for r in records]
    if all(f.exists() for f in tuned_files):
        pct = pd.concat([pd.read_csv(f) for f in tuned_files])
        cols = ["precision", "recall", "f1", "support", "threshold"]
        pct_mean = (pct.groupby("class")[cols].mean().reindex(class_names(task)))
        pct_mean.to_csv(out_root / f"per_class_val_mean_tuned_{key}.csv")
    return summary


def _plot(pc_mean: pd.DataFrame, out_root: Path, threshold: float,
          key: str = "multilabel", loss: str = "bce",
          task: str = "multilabel") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    order = pc_mean.sort_values("f1", ascending=False)
    ax.bar(range(len(order)), order["f1"], color="#4C72B0")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("val F1")
    ax.set_ylim(0, 1)
    thr = f"threshold {threshold}" if task == "multilabel" else "argmax"
    ax.set_title(f"DistilBERT {task}, loss={loss} — per-class val F1 "
                 f"({thr}, mean over seeds)")
    for i, (v, s) in enumerate(zip(order["f1"], order["support"])):
        ax.text(i, v + 0.02, f"n={int(s)}", ha="center", fontsize=7, color="#555")
    fig.tight_layout()
    fig.savefig(out_root / f"per_class_val_f1_{key}.png", dpi=150)
    plt.close(fig)


def demo(checkpoint: Path, threshold: float, out_root: Path,
         top_k: int = MAX_LABELS_PER_ROW) -> pd.DataFrame:
    """The tutorial's predict.py step, on the checkpoint we just trained.

    Shows the top ``top_k`` labels rather than only those above the threshold.
    Two reasons:

    * **The data caps at 2.** A row carries at most a dominant plus one optional
      secondary distortion — measured max is 2 in train, val and test alike, and
      ``src/evaluate.py`` already defaults to ``--max-labels 2``. Printing more
      than two would show the model doing something the labels never do.
    * **The tutorial prints nothing when the model under-fires.** Its version
      lists only labels above 0.5; on a model whose probabilities collapse below
      that (the exact failure this project measures) it emits a blank result that
      reads like a crash. Ranking is informative even when confidence is not, so
      the ranked top-2 is always shown, with the threshold marked separately.
    """
    from transformers import pipeline
    clf = pipeline("text-classification", model=str(checkpoint),
                   tokenizer=str(checkpoint), top_k=None,  # 5.x for return_all_scores
                   device=0 if torch.cuda.is_available() else -1)
    # The per-label thresholds swept on val, saved alongside the weights. The
    # demo shows BOTH cut points because on an under-firing model the flat 0.5
    # column is empty, and "nothing fired" invites the wrong conclusion — the
    # model has an opinion, it is the threshold that is refusing to report it.
    tuned = None
    meta_path = checkpoint / "meta.json"
    if meta_path.exists():
        tuned = json.loads(meta_path.read_text(encoding="utf-8")).get("thresholds_tuned")

    rows = []
    print(f"\n--- demo predictions: top {top_k} labels per text ---")
    print(f"    (the data allows at most {MAX_LABELS_PER_ROW} labels per row)")
    if tuned:
        print(f"    @{threshold} = the tutorial's flat threshold | "
              f"@tuned = per-label cut point swept on val")
    for text in DEMO_TEXTS:
        scores = sorted(clf(text, truncation=True)[0],
                        key=lambda d: d["score"], reverse=True)
        print(f"\nText: {text!r}")
        for rank, s in enumerate(scores[:top_k], start=1):
            fixed_hit = s["score"] > threshold
            line = (f"  {rank}. {s['label']:<22} {s['score']:.4f}   "
                    f"@{threshold} {'FIRES' if fixed_hit else '  -  '}")
            if tuned and s["label"] in DISTORTIONS:
                t = tuned[DISTORTIONS.index(s["label"])]
                line += f"   @tuned {t:.2f} {'FIRES' if s['score'] >= t else '  -  '}"
            print(line)
        if tuned and not any(s["score"] > threshold for s in scores[:top_k]):
            fired_tuned = [s for s in scores[:top_k] if s["label"] in DISTORTIONS
                           and s["score"] >= tuned[DISTORTIONS.index(s["label"])]]
            if fired_tuned:
                print(f"     -> nothing at {threshold}, but "
                      f"{len(fired_tuned)} label(s) fire at the tuned cut points. "
                      f"Same model,\n        same probabilities — only the line moved.")
            else:
                print(f"     -> nothing fires at either threshold.")
        for rank, s in enumerate(scores, start=1):
            r = {"text": text, "rank": rank, "label": s["label"],
                 "score": s["score"], "in_top_k": rank <= top_k,
                 "fired_fixed": s["score"] > threshold}
            if tuned and s["label"] in DISTORTIONS:
                t = tuned[DISTORTIONS.index(s["label"])]
                r.update(tuned_threshold=t, fired_tuned=s["score"] >= t)
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "demo_predictions.csv", index=False, encoding="utf-8")
    return df


# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Run the wellally.tech DistilBERT tutorial recipe on the "
                    "frozen EmpowerLens splits.")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--task", default="multilabel",
                    choices=["binary", "multiclass", "multilabel"],
                    help="multilabel = the tutorial's; binary = distorted or not; "
                         "multiclass = 11 classes incl. no_distortion")
    ap.add_argument("--tasks", default=None,
                    help="comma-separated, e.g. binary,multiclass,multilabel — "
                         "runs each in turn. Overrides --task.")
    ap.add_argument("--seeds", default="42",
                    help="comma-separated, e.g. 42,1337,2024 for the 3-seed protocol")
    # --- the tutorial's hyperparameters, as defaults ---
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--max-length", type=int, default=512)
    # --- class-imbalance ablation (all four are sigmoid + binary losses) ---
    ap.add_argument("--loss", default=None,
                    choices=["bce", "pos_bce", "focal", "asl", "ce", "weighted_ce"],
                    help="multilabel: bce (the tutorial's unweighted "
                         "BCEWithLogitsLoss) | pos_bce (+ per-class pos_weight) | "
                         "focal | asl. binary/multiclass: ce | weighted_ce. "
                         "Defaults to the task's unweighted baseline.")
    ap.add_argument("--focal-gamma", type=float, default=2.0,
                    help="only used with --loss focal")
    ap.add_argument("--ablation", action="store_true",
                    help="train every loss valid for the task in turn and write "
                         "ablation_summary.csv comparing them")
    ap.add_argument("--ablation-losses", default=None,
                    help="override the per-task loss list used by --ablation")
    ap.add_argument("--meta-thresholds", default="fixed", choices=["fixed", "tuned"],
                    help="which thresholds go into the checkpoint's meta.json, "
                         "i.e. what src/evaluate.py applies on test")
    # --- plumbing ---
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="results_tutorial_distilbert")
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="64 rows / 1 epoch - checks the code runs, not the science")
    ap.add_argument("--quiet", action="store_true",
                    help="drop the per-step progress bars and the HF load "
                         "reports; keeps the per-epoch metrics. A 12-run "
                         "ablation emits tens of thousands of tqdm lines "
                         "otherwise, which can hang a notebook front-end.")
    ap.add_argument("--no-demo", action="store_true",
                    help="skip the inference demo on the tutorial's three sentences")
    args = ap.parse_args(argv)

    if args.quiet:
        # The "LOAD REPORT" table transformers prints on every from_pretrained
        # is ~15 lines; across a 12-run ablation that alone is 180 lines of noise
        # before any training output. Errors and warnings still surface.
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tasks = ([t.strip() for t in args.tasks.split(",") if t.strip()]
             if args.tasks else [args.task])
    unknown = [t for t in tasks if t not in TASK_NUM_LABELS]
    if unknown:
        raise SystemExit(f"--tasks: unknown task(s) {unknown}. "
                         f"Valid: {', '.join(TASK_NUM_LABELS)}")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    plan = []      # (task, loss) pairs, in run order
    for task in tasks:
        if args.ablation:
            losses = ([s.strip() for s in args.ablation_losses.split(",") if s.strip()]
                      if args.ablation_losses else TASK_LOSSES[task])
        else:
            losses = [args.loss or TASK_DEFAULT_LOSS[task]]
        bad = [l for l in losses if l not in TASK_LOSSES[task]]
        if bad:
            raise SystemExit(f"loss(es) {bad} are not valid for task {task!r}. "
                             f"Valid: {', '.join(TASK_LOSSES[task])}")
        plan += [(task, l) for l in losses]

    if len(plan) > 1:
        print(f"[plan] {len(plan)} configuration(s) x {len(seeds)} seed(s) = "
              f"{len(plan) * len(seeds)} training runs")
        for t, l in plan:
            print(f"    {t:<11} loss={l}")

    all_records, ablation_rows = [], []
    for task, loss_name in plan:
        args.task, args.loss = task, loss_name
        records = [run_one_seed(args, seed, out_root) for seed in seeds]
        summary = summarize(records, out_root, args.threshold)
        all_records.extend(records)
        key = f"{task}{records[0]['suffix']}"

        print(f"\n{'=' * 70}\nVAL results - DistilBERT {task}, loss={loss_name}, "
              f"{len(seeds)} seed(s) {seeds}\n{'=' * 70}")
        print(summary[["mean_pm_std"]].to_string())
        thr = f"threshold {args.threshold}" if task == "multilabel" else "argmax"
        print(f"\nPer-class val F1 (mean over seeds, {thr}):")
        print(pd.read_csv(out_root / f"per_class_val_mean_{key}.csv")
                .round(3).to_string(index=False))

        tuned_pc = out_root / f"per_class_val_mean_tuned_{key}.csv"
        if tuned_pc.exists():
            print(f"\nPer-class val F1 at the SWEPT thresholds (same model, same "
                  f"probabilities,\nonly the cut point moved — the column shows "
                  f"where it landed per class):")
            print(pd.read_csv(tuned_pc).round(3).to_string(index=False))

        # Multi-label gets two scorings of the same probabilities; single-label
        # has only one, because argmax has no threshold to move.
        modes = ([("fixed 0.5", "metrics"), ("tuned", "metrics_tuned")]
                 if task == "multilabel" else [("argmax", "metrics")])
        for mode, mkey in modes:
            vals = pd.DataFrame([r[mkey] for r in records])
            row = {"task": task, "loss": loss_name, "threshold_mode": mode,
                   "seeds": len(seeds)}
            for m in ("macro_precision", "macro_recall", "macro_f1", "micro_f1",
                      "weighted_f1", "accuracy", "roc_auc"):
                if m in vals:
                    row[m] = vals[m].mean()
                    row[f"{m}_std"] = vals[m].std(ddof=0)
            for m in ("positive_class_f1", "macro_f1_10", "no_distortion_f1",
                      "mean_labels_predicted"):
                if m in vals:
                    row[m] = vals[m].mean()
            ablation_rows.append(row)

    abl = pd.DataFrame(ablation_rows)
    abl.to_csv(out_root / "ablation_summary.csv", index=False)
    if len(plan) > 1:
        print(f"\n{'=' * 70}\nTASK x LOSS x THRESHOLD ablation (val, mean over "
              f"{len(seeds)} seed(s))\n{'=' * 70}")
        print(abl.round(3).to_string(index=False))
        print("\nRead it as: loss changes what the model learns; threshold_mode "
              "changes only\nwhere the decision line sits on the same "
              "probabilities (no retraining).")
        print("Do NOT compare macro_f1 across tasks - 2, 11 and 10 classes are "
              "three different exams.")
        print("CAVEAT: tuned thresholds are swept ON val, so tuned val numbers "
              "are optimistic.\nThe honest read is the test-set pass.")

    if not args.no_demo and any(t == "multilabel" for t, _ in plan):
        last = next(r for r in reversed(all_records) if r["task"] == "multilabel")
        demo(Path(args.checkpoints) / f"tutorial_{last['tag']}",
             args.threshold, out_root)

    print(f"\nWrote {out_root}/ - val_summary_*.csv, val_metrics_per_seed_*.csv, "
          f"per_class_val_mean_*.csv, per_class_val_f1_*.png, ablation_summary.csv")
    print("For TEST-set numbers (the only module allowed to read test.csv):")
    for r in all_records:
        ck = Path(args.checkpoints) / f"tutorial_{r['tag']}"
        # 2 = the corpus cap and evaluate.py's default, so the number is
        # comparable with every other model in results/. The tutorial is
        # uncapped; use --max-labels 0 to reproduce that exactly.
        cap = " --max-labels 2" if r["task"] == "multilabel" else ""
        print(f"  python -m src.evaluate --checkpoint {ck}{cap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
