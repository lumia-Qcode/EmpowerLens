"""
EmpowerLens — Flat Mental-RoBERTa experiment suite (Experiments 1-8).

SCOPE NOTE
----------
Cascade architecture is deliberately OUT OF SCOPE here — that track is
on Kaggle separately and it is not yet stable (GPU deadlock failures). Every
experiment in this file, including Experiment 7, uses the existing FLAT
architecture only (single Mental-RoBERTa head per task, exactly as trained by
``src/train_transformer.py``). Experiment 7 therefore reports flat-model
numbers (3-seed mean +/- SD, val + test, macro-F1 and per-label F1) in the
same shape a future flat-vs-cascade table would need — it does not attempt to
train or evaluate a cascade.

This script is additive: it does not modify ``src/train_transformer.py``,
``src/evaluate.py``, ``src/metrics.py`` or ``src/data.py``. It imports and
reuses them wherever their behaviour is already correct (tokenisation,
collation, the metric bundle, checkpoint evaluation, threshold sweeping) and
only adds new code where the eight experiments need something the existing
scripts don't do: configurable losses, configurable samplers, data audits,
dataset ablation orchestration, and sequence-length analysis.

Usage (single seed, single run)
--------------------------------
    python -m src.experiments_flat_mentalroberta --experiment 1 \
        --splits data/splits_combined --out results/exp1

    python -m src.experiments_flat_mentalroberta --experiment 3 \
        --task multiclass --loss weighted_ce --seed 42 \
        --splits data/splits_combined --out results/exp3

Usage (orchestrate all 3 seeds — spawns one subprocess per seed, matching
the deadlock-safe pattern already used in notebooks/kaggle_runner.ipynb)
--------------------------------------------------------------------------
    python -m src.experiments_flat_mentalroberta --experiment 3 \
        --task multiclass --loss weighted_ce --run-all-seeds \
        --splits data/splits_combined --out results/exp3

Each experiment writes its own CSV/MD report under ``--out`` and never
touches ``test.csv`` outside of ``src/evaluate.py`` (the one script allowed
to read it), preserving the project's no-leakage discipline.
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data import DISTORTIONS, MC_CLASSES
from src.metrics import BINARY_CLASSES, metric_bundle, per_class_table
from src.train_transformer import (
    ML_COLS,
    TASK_NUM_LABELS,
    TEXT_COL,
    TextDataset,
    class_weights,
    collate,
    encode_texts,
    get_labels,
    load_split,
    pos_weights,
    resolve_device,
    sweep_thresholds,
)

DEFAULT_MODEL = "mental/mental-roberta-base"
DEFAULT_SEEDS = [42, 1337, 2024]

# Which --task values each experiment is valid for, and what --task it
# should default to when the person doesn't pass one explicitly. This is
# what makes "--experiment 3 --task binary" run the binary CE-vs-weighted-CE
# comparison, "--experiment 3 --task multiclass" run the multiclass version,
# and "--experiment 3 --task multilabel" refuse with a pointer to Exp 6 —
# 3/4/5 are class-imbalance-via-loss experiments for single-label heads
# (binary or multiclass, both softmax+CE-family); 6/7 are multilabel-only
# (sigmoid+BCE-family); 1/2/8 apply to any task.
EXPERIMENT_ALLOWED_TASKS = {
    1: {"binary", "multiclass", "multilabel"},
    2: {"binary", "multiclass", "multilabel"},
    3: {"binary", "multiclass"},
    4: {"binary", "multiclass"},
    5: {"binary", "multiclass"},
    6: {"multilabel"},
    7: {"multilabel"},
    8: {"binary", "multiclass", "multilabel"},
}
EXPERIMENT_DEFAULT_TASK = {
    1: "multiclass", 2: "multilabel", 3: "multiclass", 4: "multiclass",
    5: "multiclass", 6: "multilabel", 7: "multilabel", 8: "multiclass",
}
EXPERIMENT_REDIRECT_HINT = {
    3: "For multilabel class imbalance, use Experiment 6 (label-wise loss weighting) instead.",
    4: "For multilabel class imbalance, use Experiment 6 (label-wise loss weighting) instead.",
    5: "For multilabel class imbalance, use Experiment 6 (label-wise loss weighting) instead.",
    6: "For binary/multiclass class imbalance, use Experiment 3 (CE vs weighted CE), "
       "Experiment 4 (focal vs class-balanced), or Experiment 5 (weighted sampling) instead.",
    7: "Experiment 7 reports the flat MULTILABEL model only. For binary/multiclass results, "
       "run Experiment 2 (dataset ablation) or 3/4/5 (imbalance) with the task you need.",
}


def validate_task_for_experiment(experiment: int, task: str) -> None:
    """Enforce that --task matches what the experiment actually measures,
    so 'put --task binary and binary-relevant experiments run, put
    --task multiclass and multiclass-relevant ones run, etc.' — mismatches
    fail loudly with a pointer to the right experiment number instead of
    silently training the wrong head."""
    allowed = EXPERIMENT_ALLOWED_TASKS[experiment]
    if task in allowed:
        return
    hint = EXPERIMENT_REDIRECT_HINT.get(experiment, "")
    raise SystemExit(
        f"Experiment {experiment} only supports --task {sorted(allowed)}, got --task {task!r}. "
        + hint
    )


# =====================================================================
# 0. Losses (Experiments 3, 4, 6)
# =====================================================================

class FocalLoss(nn.Module):
    """Multiclass focal loss (Lin et al. 2017). alpha is an optional
    per-class weight tensor (e.g. from class_weights() or class-balanced
    weighting); gamma controls down-weighting of easy examples."""

    def __init__(self, alpha: torch.Tensor | None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class MultilabelFocalLoss(nn.Module):
    """Per-label sigmoid focal loss. pos_weight is an optional per-label
    tensor (e.g. from pos_weights()) applied inside the BCE term, so rare
    labels get both the focal down-weighting of easy negatives AND the
    positive-class boost — this is the "label-wise loss weighting" asked
    for in Experiment 6."""

    def __init__(self, pos_weight: torch.Tensor | None, gamma: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        p = torch.sigmoid(logits)
        pt = torch.where(target == 1, p, 1 - p)
        loss = ((1 - pt) ** self.gamma) * bce
        return loss.mean()


def class_balanced_weights(y: np.ndarray, num_labels: int, beta: float = 0.999,
                            device: str = "cpu") -> torch.Tensor:
    """Cui et al. (2019) "effective number of samples" class weights.
    Distinct from src.train_transformer.class_weights (plain inverse
    frequency, used for the "weighted_ce" condition in Experiment 3);
    this is the "class_balanced" condition in Experiment 4."""
    counts = np.bincount(y, minlength=num_labels).astype(float)
    counts[counts == 0] = 1.0
    eff_num = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / eff_num
    w = w / w.sum() * num_labels
    return torch.tensor(w, dtype=torch.float, device=device)


def build_loss(task: str, loss_name: str, y_train: np.ndarray, num_labels: int,
                device: str, gamma: float, cb_beta: float):
    """loss_name in {ce, weighted_ce, focal, class_balanced} for
    binary/multiclass, or {bce, weighted_bce, focal} for multilabel."""
    if task == "multilabel":
        if loss_name in ("weighted_bce", "weighted_ce"):
            pw = pos_weights(y_train, device)
            return nn.BCEWithLogitsLoss(pos_weight=pw)
        if loss_name == "focal":
            pw = pos_weights(y_train, device)
            return MultilabelFocalLoss(pos_weight=pw, gamma=gamma)
        if loss_name == "bce" or loss_name == "ce":
            return nn.BCEWithLogitsLoss()
        raise ValueError(f"unknown multilabel loss {loss_name!r}")

    # binary / multiclass
    if loss_name == "ce":
        return nn.CrossEntropyLoss()
    if loss_name == "weighted_ce":
        cw = class_weights(y_train, num_labels, device)
        return nn.CrossEntropyLoss(weight=cw)
    if loss_name == "class_balanced":
        cw = class_balanced_weights(y_train, num_labels, cb_beta, device)
        return nn.CrossEntropyLoss(weight=cw)
    if loss_name == "focal":
        cw = class_weights(y_train, num_labels, device)
        return FocalLoss(alpha=cw, gamma=gamma)
    raise ValueError(f"unknown loss {loss_name!r}")


# =====================================================================
# 1. Sampler (Experiment 5)
# =====================================================================

def build_sampler(task: str, y_train: np.ndarray, sampler_name: str):
    """Inverse-frequency WeightedRandomSampler. For multiclass/binary the
    weight is 1/freq(class). For multilabel there's no single class per
    row, so each row's weight is the max inverse-frequency over the
    labels it carries (an all-zero / no_distortion row gets the inverse
    frequency of the no-distortion "class", i.e. the complement)."""
    if sampler_name == "none":
        return None
    if sampler_name != "weighted":
        raise ValueError(f"unknown sampler {sampler_name!r}")

    if task in ("binary", "multiclass"):
        counts = np.bincount(y_train)
        counts[counts == 0] = 1
        inv_freq = 1.0 / counts
        weights = inv_freq[y_train]
    else:  # multilabel
        n = y_train.shape[0]
        pos_counts = y_train.sum(axis=0)
        pos_counts[pos_counts == 0] = 1
        inv_freq = 1.0 / pos_counts
        neg_rows = (y_train.sum(axis=1) == 0)
        neg_weight = 1.0 / max(neg_rows.sum(), 1)
        weights = np.full(n, neg_weight, dtype=float)
        has_label = y_train.sum(axis=1) > 0
        weights[has_label] = (y_train[has_label] * inv_freq).max(axis=1)

    weights = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class FlatTrainer(Trainer):
    """Trainer subclass supporting a custom loss_fn (all experiments) and
    an optional custom train sampler (Experiment 5). Mirrors
    src.train_transformer.WeightedTrainer but adds the sampler hook,
    which HF's default Trainer does not expose a simple kwarg for."""

    def __init__(self, *args, loss_fn=None, custom_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn
        self.custom_sampler = custom_sampler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self, train_dataset=None):
        if self.custom_sampler is not None:
            return self.custom_sampler
        return super()._get_train_sampler(train_dataset)


# =====================================================================
# 2. Core training routine (shared by Experiments 2-7)
# =====================================================================

def run_training(task: str, model_name: str, seed: int, splits_dir: str, out_dir: str,
                  loss_name: str, sampler_name: str = "none", epochs: int = 4, lr: float = 2e-5,
                  batch_size: int = 16, max_length: int = 512, truncation: str = "head",
                  head_keep: int = 128, device_choice: str = "auto", gamma: float = 2.0,
                  cb_beta: float = 0.999, run_tag: str = "run", smoke: bool = False) -> Path:
    """Train one flat Mental-RoBERTa run with a configurable loss/sampler
    and save a checkpoint whose meta.json is fully compatible with
    src/evaluate.py (task, model, seed, max_length, truncation, head_keep,
    thresholds), so evaluation always goes through the existing script."""
    device = resolve_device(device_choice)
    set_seed(seed)
    num_labels = TASK_NUM_LABELS[task]
    multilabel = task == "multilabel"

    train_df = load_split(splits_dir, "train")
    val_df = load_split(splits_dir, "val")
    if smoke:
        train_df = train_df.head(100).reset_index(drop=True)
        val_df = val_df.head(100).reset_index(drop=True)
        epochs = 1

    y_train = get_labels(train_df, task)
    y_val = get_labels(val_df, task)

    print(f"[{run_tag}] task={task} model={model_name} seed={seed} loss={loss_name} "
          f"sampler={sampler_name} device={device} train={len(train_df)} val={len(val_df)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {"num_labels": num_labels}
    if multilabel:
        model_kwargs["problem_type"] = "multi_label_classification"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs).to(device)

    tr_enc, _ = encode_texts(train_df[TEXT_COL], tokenizer, max_length, truncation, head_keep)
    va_enc, val_trunc_rate = encode_texts(val_df[TEXT_COL], tokenizer, max_length, truncation, head_keep)
    train_ds = TextDataset(tr_enc, list(y_train))
    val_ds = TextDataset(va_enc, list(y_val))

    loss_fn = build_loss(task, loss_name, y_train, num_labels, device, gamma, cb_beta)
    sampler = build_sampler(task, y_train, sampler_name)

    out_name = f"{run_tag}_{task}_{model_name.split('/')[-1]}_{loss_name}_{sampler_name}_{seed}"
    ckpt_dir = Path(out_dir) / "checkpoints" / out_name

    targs = TrainingArguments(
        output_dir=str(ckpt_dir), num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, learning_rate=lr, warmup_ratio=0.1, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True, save_total_limit=1,
        fp16=(device == "cuda"), logging_steps=10, report_to="none", seed=seed,
        use_cpu=(device == "cpu"),
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        if multilabel:
            preds = (1 / (1 + np.exp(-logits)) >= 0.5).astype(int)
            return {"macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}
        preds = np.argmax(logits, axis=1)
        return {"macro_f1": f1_score(labels, preds, average="macro", zero_division=0)}

    trainer = FlatTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=functools.partial(collate, pad_id=tokenizer.pad_token_id, multilabel=multilabel),
        compute_metrics=compute_metrics, processing_class=tokenizer,
        loss_fn=loss_fn, custom_sampler=sampler,
    )
    trainer.train()
    val_metrics = trainer.evaluate()

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    thresholds = None
    if multilabel:
        logits = trainer.predict(val_ds).predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = 1 / (1 + np.exp(-logits))
        thresholds = sweep_thresholds(probs, y_val)
        (ckpt_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    meta = {
        "task": task, "model": model_name, "seed": seed, "epochs": epochs, "lr": lr,
        "batch_size": batch_size, "max_length": max_length, "truncation": truncation,
        "head_keep": head_keep, "device": device, "smoke": smoke, "num_labels": num_labels,
        "val_truncation_rate": val_trunc_rate,
        "loss": loss_name, "sampler": sampler_name, "gamma": gamma, "cb_beta": cb_beta,
        "val_metrics": {k: float(v) for k, v in val_metrics.items() if isinstance(v, (int, float))},
        "thresholds": thresholds,
    }
    (ckpt_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"  -> saved {ckpt_dir} (val macro_f1={val_metrics.get('eval_macro_f1', float('nan')):.3f})")
    return ckpt_dir


def run_evaluation(ckpt_dir: Path, splits_dir: str, out_dir: str, max_labels: int = 2) -> dict:
    """Delegate to src/evaluate.py via subprocess — the only script allowed
    to read test.csv — then parse the eval_*.json it writes. Isolating this
    in its own process also avoids compounding CUDA state across repeated
    runs in one Kaggle session (a known T4x2 deadlock risk)."""
    cmd = [
        sys.executable, "-m", "src.evaluate",
        "--checkpoint", str(ckpt_dir), "--splits", splits_dir, "--out", out_dir,
        "--max-labels", str(max_labels),
    ]
    subprocess.run(cmd, check=True)
    meta = json.loads((ckpt_dir / "meta.json").read_text())
    tag = meta["model"].split("/")[-1]
    eval_path = Path(out_dir) / f"eval_{tag}_{meta['task']}_{meta['seed']}.json"
    return json.loads(eval_path.read_text())


def run_single_seed(experiment_tag: str, task: str, model_name: str, seed: int, splits_dir: str,
                     out_dir: str, loss_name: str, sampler_name: str, args) -> dict:
    ckpt = run_training(
        task=task, model_name=model_name, seed=seed, splits_dir=splits_dir, out_dir=out_dir,
        loss_name=loss_name, sampler_name=sampler_name, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, max_length=args.max_length, truncation=args.truncation,
        head_keep=args.head_keep, device_choice=args.device, gamma=args.gamma, cb_beta=args.cb_beta,
        run_tag=experiment_tag, smoke=args.smoke,
    )
    return run_evaluation(ckpt, splits_dir, out_dir, max_labels=args.max_labels)


def orchestrate_all_seeds(experiment_num: int, seeds: list[int], base_argv: list[str]) -> None:
    """Spawn one subprocess per seed (same pattern as
    notebooks/kaggle_runner.ipynb's `for seed in (42, 1337, 2024): !python ...`
    loop) so each seed gets a fresh CUDA context."""
    for seed in seeds:
        cmd = [sys.executable, "-m", "src.experiments_flat_mentalroberta",
               *base_argv, "--seed", str(seed)]
        print(f"\n=== orchestrating seed {seed}: {' '.join(cmd)} ===")
        subprocess.run(cmd, check=True)


def _mean_std_table(rows: list[dict], metric_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    agg = {}
    for c in metric_cols:
        if c in df.columns:
            agg[f"{c}_mean"] = df[c].mean()
            agg[f"{c}_std"] = df[c].std(ddof=0)
    return pd.DataFrame([agg])


# =====================================================================
# Experiment 1 — Data / label audit
# =====================================================================

def experiment1_audit(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    splits = {n: load_split(args.splits, n) for n in ("train", "val", "test")}

    # --- class frequencies per split (multiclass, 11 classes) -----------
    freq_rows = []
    for name, df in splits.items():
        vc = df["y_mc"].value_counts().reindex(range(11), fill_value=0)
        for idx, count in vc.items():
            freq_rows.append({"split": name, "class": MC_CLASSES[idx], "count": int(count),
                              "pct_of_split": round(100 * count / len(df), 2)})
    freq_df = pd.DataFrame(freq_rows)
    freq_df.to_csv(out / "exp1_multiclass_frequencies.csv", index=False)
    print("\n=== Multiclass frequencies by split ===")
    print(freq_df.pivot(index="class", columns="split", values="count").reindex(MC_CLASSES))

    # --- multilabel prevalence per split ---------------------------------
    prev_rows = []
    for name, df in splits.items():
        for d in DISTORTIONS:
            col = f"ml_{d}"
            pos = int(df[col].sum())
            prev_rows.append({"split": name, "label": d, "positives": pos,
                              "prevalence_pct": round(100 * pos / len(df), 2)})
    prev_df = pd.DataFrame(prev_rows)
    prev_df.to_csv(out / "exp1_multilabel_prevalence.csv", index=False)
    print("\n=== Multilabel prevalence by split ===")
    print(prev_df.pivot(index="label", columns="split", values="prevalence_pct").reindex(DISTORTIONS))

    # --- label co-occurrence (train split, 10x10) -------------------------
    ml_cols = [f"ml_{d}" for d in DISTORTIONS]
    train_ml = splits["train"][ml_cols].to_numpy()
    cooc = train_ml.T @ train_ml  # raw co-occurrence counts
    cooc_df = pd.DataFrame(cooc.astype(int), index=DISTORTIONS, columns=DISTORTIONS)
    cooc_df.to_csv(out / "exp1_label_cooccurrence_counts_train.csv")
    # normalized (Jaccard-style: co-occur / union) for readability
    diag = np.diag(cooc).astype(float)
    union = diag[:, None] + diag[None, :] - cooc
    with np.errstate(divide="ignore", invalid="ignore"):
        jacc = np.where(union > 0, cooc / union, 0.0)
    pd.DataFrame(jacc, index=DISTORTIONS, columns=DISTORTIONS).round(3).to_csv(
        out / "exp1_label_cooccurrence_jaccard_train.csv"
    )
    print("\n=== Label co-occurrence (train, raw counts) ===")
    print(cooc_df)

    # --- split-imbalance flags -------------------------------------------
    warn_lines = ["# Experiment 1 — split imbalance flags\n"]
    for d in DISTORTIONS:
        counts = {n: int(splits[n][f"ml_{d}"].sum()) for n in splits}
        if min(counts.values()) < 15:
            warn_lines.append(
                f"- **{d}**: low support in at least one split -> {counts} "
                f"(rare-class risk for multilabel/multiclass F1 variance)\n"
            )
    (out / "exp1_imbalance_flags.md").write_text("".join(warn_lines), encoding="utf-8")

    # --- optional: model-based per-class P/R/F1 + confusion matrix -------
    # (delegates to src.evaluate so test.csv is still only read there)
    for ckpt_arg, task in ((args.checkpoint_mc, "multiclass"), (args.checkpoint_ml, "multilabel")):
        if not ckpt_arg:
            continue
        print(f"\n[exp1] evaluating provided {task} checkpoint {ckpt_arg} ...")
        eval_json = run_evaluation(Path(ckpt_arg), args.splits, str(out), max_labels=args.max_labels)
        pc_test = pd.DataFrame(eval_json["splits"]["test"]["per_class"])
        pc_test.to_csv(out / f"exp1_{task}_per_class_test.csv", index=False)
        print(pc_test.to_string(index=False))
        if task == "multiclass":
            print("(row-normalised confusion matrix PNG written by src.evaluate to the same --out dir)")

    print(f"\nExperiment 1 artifacts written to {out}/")


# =====================================================================
# Experiment 2 — Dataset ablation (Annotated vs CODIPAS vs combined)
# =====================================================================

def experiment2_dataset_ablation(args):
    """Same backbone + eval protocol across three data configurations.
    Requires the three split directories to already exist (each with its
    own frozen train/val/test.csv + split_manifest.json, produced the same
    way as data/splits / data/splits_combined). This script does not
    generate splits — see src/make_splits.py and CLAUDE.md's no-leakage
    rules for why splits are frozen and version-controlled."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    configs = {
        "annotated_only": args.annotated_splits,
        "codipas_only": args.codipas_splits,
        "annotated_plus_codipas": args.combined_splits,
    }
    missing = {k: v for k, v in configs.items() if not v or not Path(v).exists()}
    if missing:
        raise SystemExit(
            f"Experiment 2 needs all three split dirs to exist. Missing/unset: {missing}. "
            f"Pass --annotated-splits / --codipas-splits / --combined-splits."
        )

    if args.aggregate_only:
        _aggregate_experiment2(args)
        return

    if args.seed is None:
        # orchestration mode: spawn one subprocess per (config, seed).
        # Respects --only-config so a single call trains exactly one dataset
        # at a time (all 3 of ITS seeds), never more than that.
        configs_to_orchestrate = (
            {args.only_config: configs[args.only_config]} if args.only_config else configs
        )
        for name, splits_dir in configs_to_orchestrate.items():
            base_argv = ["--experiment", "2", "--task", args.task, "--out", args.out,
                         "--annotated-splits", args.annotated_splits,
                         "--codipas-splits", args.codipas_splits,
                         "--combined-splits", args.combined_splits,
                         "--only-config", name]
            orchestrate_all_seeds(2, args.seeds, base_argv)
        if not args.only_config:
            _aggregate_experiment2(args)
        return

    # single-seed leaf run for one config (called by the orchestration above,
    # or directly if the caller only wants one config/seed). Loss is chosen
    # by task: multilabel needs weighted BCE, binary/multiclass need
    # weighted CE — passing "weighted_ce" for a multilabel task would be
    # silently wrong (build_loss would reject it), so this must branch.
    loss_name = "weighted_bce" if args.task == "multilabel" else "weighted_ce"
    configs_to_run = {args.only_config: configs[args.only_config]} if args.only_config else configs
    for name, splits_dir in configs_to_run.items():
        eval_json = run_single_seed(
            experiment_tag=f"exp2_{name}", task=args.task, model_name=args.model, seed=args.seed,
            splits_dir=splits_dir, out_dir=str(out / name), loss_name=loss_name,
            sampler_name="none", args=args,
        )
        row = {"config": name, "seed": args.seed,
               **{f"val_{k}": v for k, v in eval_json["splits"]["val"]["metrics"].items()},
               **{f"test_{k}": v for k, v in eval_json["splits"]["test"]["metrics"].items()}}
        _append_row_csv(out / "exp2_all_seed_results.csv", row)


def _append_row_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        df_new = pd.concat([old, df_new], ignore_index=True)
    df_new.to_csv(path, index=False)


def _aggregate_experiment2(args):
    out = Path(args.out)
    path = out / "exp2_all_seed_results.csv"
    if not path.exists():
        print("No exp2_all_seed_results.csv found yet — nothing to aggregate.")
        return
    df = pd.read_csv(path)
    metric_cols = [c for c in df.columns if c.startswith("val_") or c.startswith("test_")]
    metric_cols = [c for c in metric_cols if pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby("config")[metric_cols].agg(["mean", "std"])
    summary.to_csv(out / "exp2_mean_std_by_config.csv")
    print("\n=== Experiment 2 — individual seed results ===")
    print(df.to_string(index=False))
    print("\n=== Experiment 2 — mean +/- std by config ===")
    print(summary)
    print(f"\nWrote {out/'exp2_all_seed_results.csv'} and {out/'exp2_mean_std_by_config.csv'}")


# =====================================================================
# Experiments 3, 4, 5 — loss / sampler comparisons (multiclass, by default)
# =====================================================================

def _loss_sampler_experiment(args, experiment_tag: str, loss_name: str, sampler_name: str,
                              summary_prefix: str):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.seed is None:
        base_argv = ["--experiment", args.experiment_num_str, "--task", args.task,
                     "--splits", args.splits, "--out", args.out, "--loss", loss_name,
                     "--sampler", sampler_name, "--gamma", str(args.gamma), "--cb-beta", str(args.cb_beta)]
        orchestrate_all_seeds(int(args.experiment_num_str), args.seeds, base_argv)
        _aggregate_generic(out, f"{summary_prefix}_all_seed_results.csv",
                           f"{summary_prefix}_mean_std.csv", group_cols=["loss", "sampler"])
        return

    eval_json = run_single_seed(
        experiment_tag=experiment_tag, task=args.task, model_name=args.model, seed=args.seed,
        splits_dir=args.splits, out_dir=str(out), loss_name=loss_name, sampler_name=sampler_name,
        args=args,
    )
    row = {"loss": loss_name, "sampler": sampler_name, "seed": args.seed,
           **{f"val_{k}": v for k, v in eval_json["splits"]["val"]["metrics"].items()},
           **{f"test_{k}": v for k, v in eval_json["splits"]["test"]["metrics"].items()}}
    _append_row_csv(out / f"{summary_prefix}_all_seed_results.csv", row)


def _aggregate_generic(out: Path, in_name: str, out_name: str, group_cols: list[str]):
    path = out / in_name
    if not path.exists():
        print(f"No {in_name} found yet — nothing to aggregate.")
        return
    df = pd.read_csv(path)
    metric_cols = [c for c in df.columns if (c.startswith("val_") or c.startswith("test_"))
                   and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby(group_cols)[metric_cols].agg(["mean", "std"])
    summary.to_csv(out / out_name)
    print(f"\n=== {in_name} — individual results ===")
    print(df.to_string(index=False))
    print(f"\n=== mean +/- std by {group_cols} ===")
    print(summary)


def experiment3_ce_vs_weighted_ce(args):
    """Standard CE vs class-weighted CE, binary or multiclass (--task),
    same backbone/splits. Run this only if Experiment 1's audit shows
    meaningful imbalance (e.g. no_distortion / all_or_nothing support gap
    for multiclass, or the Distorted/Non-Distorted split for binary)."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.seed is None:
        for loss_name in ("ce", "weighted_ce"):
            base_argv = ["--experiment", "3", "--task", args.task, "--splits", args.splits,
                         "--out", args.out, "--loss", loss_name]
            orchestrate_all_seeds(3, args.seeds, base_argv)
        _aggregate_generic(out, "exp3_all_seed_results.csv", "exp3_mean_std.csv", ["loss"])
        return
    _loss_sampler_experiment(args, "exp3", args.loss, "none", "exp3")


def experiment4_focal_vs_class_balanced(args):
    """Best of {weighted_ce baseline} vs focal vs class_balanced, binary or
    multiclass (--task). Run only if Experiment 3's weighted CE improvement
    over plain CE is insufficient (per the FYP's stated decision rule)."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.seed is None:
        for loss_name in ("weighted_ce", "focal", "class_balanced"):
            base_argv = ["--experiment", "4", "--task", args.task, "--splits", args.splits,
                         "--out", args.out, "--loss", loss_name, "--gamma", str(args.gamma),
                         "--cb-beta", str(args.cb_beta)]
            orchestrate_all_seeds(4, args.seeds, base_argv)
        _aggregate_generic(out, "exp4_all_seed_results.csv", "exp4_mean_std.csv", ["loss"])
        return
    _loss_sampler_experiment(args, "exp4", args.loss, "none", "exp4")


def experiment5_weighted_sampling(args):
    """Weighted sampling vs the best loss-based approach from Exp3/4
    (pass --best-loss to name it), binary or multiclass (--task), all
    other settings fixed."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    best_loss = args.best_loss or "weighted_ce"
    if args.seed is None:
        for sampler_name in ("none", "weighted"):
            base_argv = ["--experiment", "5", "--task", args.task, "--splits", args.splits,
                         "--out", args.out, "--loss", best_loss, "--sampler", sampler_name]
            orchestrate_all_seeds(5, args.seeds, base_argv)
        _aggregate_generic(out, "exp5_all_seed_results.csv", "exp5_mean_std.csv", ["loss", "sampler"])
        return
    _loss_sampler_experiment(args, "exp5", best_loss, args.sampler, "exp5")


# =====================================================================
# Experiment 6 — Multilabel per-label performance + threshold re-check
# =====================================================================

def experiment6_multilabel(args):
    """Trains/evaluates the multilabel head with label-wise loss weighting
    (pos_weight and/or focal), reports per-label F1, and re-runs threshold
    optimisation strictly on val (src.train_transformer.sweep_thresholds,
    already val-only) with those thresholds then fixed for test — this
    re-uses the exact mechanism train_transformer.py/evaluate.py already
    implement, so this experiment mainly changes the loss and reports the
    per-label breakdown explicitly."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    loss_name = args.loss if args.loss != "ce" else "weighted_bce"
    if args.seed is None:
        base_argv = ["--experiment", "6", "--task", args.task, "--splits", args.splits,
                     "--out", args.out, "--loss", loss_name, "--gamma", str(args.gamma)]
        orchestrate_all_seeds(6, args.seeds, base_argv)
        _report_experiment6(out)
        return

    eval_json = run_single_seed(
        experiment_tag="exp6", task=args.task, model_name=args.model, seed=args.seed,
        splits_dir=args.splits, out_dir=str(out), loss_name=loss_name, sampler_name="none", args=args,
    )
    pc_test = pd.DataFrame(eval_json["splits"]["test"]["per_class"])
    pc_test["seed"] = args.seed
    pc_test["loss"] = loss_name
    path = out / "exp6_per_label_all_seeds.csv"
    if path.exists():
        pc_test = pd.concat([pd.read_csv(path), pc_test], ignore_index=True)
    pc_test.to_csv(path, index=False)

    row = {"loss": loss_name, "seed": args.seed,
           **{f"val_{k}": v for k, v in eval_json["splits"]["val"]["metrics"].items()},
           **{f"test_{k}": v for k, v in eval_json["splits"]["test"]["metrics"].items()}}
    _append_row_csv(out / "exp6_all_seed_results.csv", row)


def _report_experiment6(out: Path):
    pl_path = out / "exp6_per_label_all_seeds.csv"
    if pl_path.exists():
        pl = pd.read_csv(pl_path)
        summary = pl.groupby(["loss", "class"])["f1"].agg(["mean", "std"]).round(3)
        summary.to_csv(out / "exp6_per_label_mean_std.csv")
        print("\n=== Experiment 6 — per-label F1 (mean +/- std across seeds, test) ===")
        print(summary)
    _aggregate_generic(out, "exp6_all_seed_results.csv", "exp6_mean_std.csv", ["loss"])


# =====================================================================
# Experiment 7 — Flat model report (cascade explicitly excluded)
# =====================================================================

def experiment7_flat_report(args):
    """Reports the flat multilabel model's macro-F1 and per-label F1 across
    3 seeds. Cascade is NOT run or compared here — see module docstring.
    Use this table as the flat side of a future flat-vs-cascade comparison
    once Izza's cascade track is stable; do not fabricate cascade numbers."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print("NOTE: Experiment 7 in this script reports FLAT architecture only. "
          "Cascade (Stage 1 binary gate -> Stage 2 multilabel) is owned by "
          "Izza's separate Kaggle track and is intentionally not run here.")
    loss_name = args.loss if args.loss != "ce" else "weighted_bce"
    if args.seed is None:
        base_argv = ["--experiment", "7", "--task", args.task, "--splits", args.splits,
                     "--out", args.out, "--loss", loss_name]
        orchestrate_all_seeds(7, args.seeds, base_argv)
        _report_experiment7(out)
        return

    eval_json = run_single_seed(
        experiment_tag="exp7_flat", task=args.task, model_name=args.model, seed=args.seed,
        splits_dir=args.splits, out_dir=str(out), loss_name=loss_name, sampler_name="none", args=args,
    )
    pc_test = pd.DataFrame(eval_json["splits"]["test"]["per_class"])
    pc_test["seed"] = args.seed
    path = out / "exp7_flat_per_label_all_seeds.csv"
    if path.exists():
        pc_test = pd.concat([pd.read_csv(path), pc_test], ignore_index=True)
    pc_test.to_csv(path, index=False)

    row = {"architecture": "flat", "seed": args.seed,
           **{f"val_{k}": v for k, v in eval_json["splits"]["val"]["metrics"].items()},
           **{f"test_{k}": v for k, v in eval_json["splits"]["test"]["metrics"].items()}}
    _append_row_csv(out / "exp7_flat_all_seed_results.csv", row)


def _report_experiment7(out: Path):
    pl_path = out / "exp7_flat_per_label_all_seeds.csv"
    if pl_path.exists():
        pl = pd.read_csv(pl_path)
        summary = pl.groupby("class")["f1"].agg(["mean", "std"]).round(3)
        summary.to_csv(out / "exp7_flat_per_label_mean_std.csv")
        print("\n=== Experiment 7 — FLAT per-label F1 (mean +/- std, test) ===")
        print(summary)
    res_path = out / "exp7_flat_all_seed_results.csv"
    if res_path.exists():
        df = pd.read_csv(res_path)
        print("\n=== Experiment 7 — FLAT individual seed results ===")
        print(df.to_string(index=False))
        macro_cols = [c for c in df.columns if "macro_f1" in c]
        print("\n=== Experiment 7 — FLAT mean +/- std (macro_f1) ===")
        for c in macro_cols:
            print(f"  {c}: {df[c].mean():.3f} +/- {df[c].std(ddof=0):.3f}")
    print("\nCascade comparison intentionally omitted — see Izza's cascade track "
          "(currently blocked on Kaggle GPU deadlocks per project notes).")


# =====================================================================
# Experiment 8 — Sequence length / truncation, optional long-context model
# =====================================================================

def experiment8_sequence_length(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_lengths = []
    for split_name in ("train", "val", "test"):
        df = load_split(args.splits, split_name)
        ids = tokenizer(df[TEXT_COL].astype(str).tolist(), add_special_tokens=False)["input_ids"]
        lengths = np.array([len(x) for x in ids])
        all_lengths.append(lengths)
        for max_len in (128, 256, 512):
            trunc_rate = float((lengths > max_len - 2).mean())  # -2 for CLS/SEP
            print(f"[{split_name}] max_length={max_len}: truncation_rate={trunc_rate:.4f}")

    lengths = np.concatenate(all_lengths)
    stats = {
        "n": int(len(lengths)),
        "median": float(np.median(lengths)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "max": int(lengths.max()),
    }
    for max_len in (128, 256, 512, 1024):
        stats[f"pct_truncated_at_{max_len}"] = round(100 * float((lengths > max_len - 2).mean()), 2)

    (out / "exp8_sequence_length_stats.json").write_text(json.dumps(stats, indent=2))
    print("\n=== Experiment 8 — sequence length stats (all splits pooled) ===")
    print(json.dumps(stats, indent=2))

    substantial = stats["pct_truncated_at_512"] > args.truncation_threshold_pct
    print(f"\n>512-token truncation rate = {stats['pct_truncated_at_512']}% "
          f"(threshold for triggering a long-context test: {args.truncation_threshold_pct}%)")

    if not substantial:
        print("Truncation at max_length=512 is below threshold — skipping the "
              "Longformer-type comparison run per the experiment's own trigger condition.")
        return

    if not args.run_longformer:
        print("Truncation IS substantial, but --run-longformer was not passed. "
              "Re-run with --run-longformer --longformer-model <hf-id> "
              "(e.g. allenai/longformer-base-4096) to launch the comparison training run.")
        return

    print(f"Launching long-context comparison run with {args.longformer_model} ...")
    eval_json = run_single_seed(
        experiment_tag="exp8_longformer", task=args.task, model_name=args.longformer_model,
        seed=args.seed or DEFAULT_SEEDS[0], splits_dir=args.splits, out_dir=str(out),
        loss_name=args.loss, sampler_name="none",
        args=args,
    )
    (out / "exp8_longformer_eval.json").write_text(json.dumps(eval_json, indent=2))
    print("Wrote exp8_longformer_eval.json — compare its test macro_f1 against the "
          "Mental-RoBERTa flat baseline at the same task/splits/seed.")


# =====================================================================
# CLI
# =====================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EmpowerLens flat Mental-RoBERTa experiment suite (1-8).")
    ap.add_argument("--experiment", type=int, required=True, choices=range(1, 9))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--task", default=None,
                    choices=["binary", "multiclass", "multilabel"],
                    help="defaults to whichever task each experiment measures if omitted "
                         "(see EXPERIMENT_DEFAULT_TASK) — e.g. Experiment 6/7 default to "
                         "multilabel, Experiment 3/4/5 default to multiclass")
    ap.add_argument("--splits", default="data/splits_combined")
    ap.add_argument("--out", default="results/experiments")
    ap.add_argument("--seed", type=int, default=None, help="omit to orchestrate all --seeds via subprocess")
    ap.add_argument("--seeds", default="42,1337,2024")
    ap.add_argument("--loss", default="weighted_ce",
                    choices=["ce", "weighted_ce", "focal", "class_balanced", "bce", "weighted_bce"])
    ap.add_argument("--sampler", default="none", choices=["none", "weighted"])
    ap.add_argument("--best-loss", default=None, help="Experiment 5: loss config to pair with sampling")
    ap.add_argument("--gamma", type=float, default=2.0, help="focal loss gamma")
    ap.add_argument("--cb-beta", type=float, default=0.999, help="class-balanced loss beta")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--truncation", choices=["head", "head_tail"], default="head")
    ap.add_argument("--head-keep", type=int, default=128)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-labels", type=int, default=2, help="multilabel prediction cap, passed to src.evaluate")
    ap.add_argument("--smoke", action="store_true")
    # Experiment 1
    ap.add_argument("--checkpoint-mc", default=None, help="optional trained multiclass checkpoint for audit")
    ap.add_argument("--checkpoint-ml", default=None, help="optional trained multilabel checkpoint for audit")
    # Experiment 2
    ap.add_argument("--annotated-splits", default="data/splits")
    ap.add_argument("--codipas-splits", default="data/splits_codipas_cls")
    ap.add_argument("--combined-splits", default="data/splits_combined")
    ap.add_argument("--only-config", default=None, help=argparse.SUPPRESS)  # internal, used by orchestration
    ap.add_argument("--aggregate-only", action="store_true",
                    help="skip training entirely; just re-read existing *_all_seed_results.csv "
                         "and rewrite the mean/std summary (use after running each dataset "
                         "config separately with --only-config)")
    # Experiment 8
    ap.add_argument("--run-longformer", action="store_true")
    ap.add_argument("--longformer-model", default="allenai/longformer-base-4096")
    ap.add_argument("--truncation-threshold-pct", type=float, default=10.0,
                    help="trigger the long-context run only above this %% truncated at 512 tokens")
    return ap


def main(argv=None):
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    args.seeds = [int(s) for s in args.seeds.split(",")]
    args.experiment_num_str = str(args.experiment)

    if args.task is None:
        args.task = EXPERIMENT_DEFAULT_TASK[args.experiment]
    validate_task_for_experiment(args.experiment, args.task)

    dispatch = {
        1: experiment1_audit,
        2: experiment2_dataset_ablation,
        3: experiment3_ce_vs_weighted_ce,
        4: experiment4_focal_vs_class_balanced,
        5: experiment5_weighted_sampling,
        6: experiment6_multilabel,
        7: experiment7_flat_report,
        8: experiment8_sequence_length,
    }
    dispatch[args.experiment](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
