"""
Evaluate a trained checkpoint — **the only script allowed to read test.csv**.

Loads a checkpoint (and, for multilabel, its ``thresholds.json``), reconstructs
the exact tokenization from the checkpoint's ``meta.json``, scores **val and
test**, and writes to ``results/``:

  (a) per_class_<model>_<task>_<seed>.csv  — per-class P/R/F1/support +
      weighted_contribution, for the test split. Prints the no_distortion row's
      contribution and its share of the weighted-F1 to stdout.
  (b) paper_comparison.csv                 — one row per (model, task, seed,
      split), merged in via the shared upsert. ``--reference`` appends the two
      literature rows (source=paper) with their uncertainty markers.
  (c) confusion_<model>_<task>_<seed>.png  — row-normalised confusion matrix
      (binary/multiclass); the 11-class task also gets a *_no_nd.png with
      no_distortion dropped so distortion-vs-distortion structure is legible.
  (d) eval_<model>_<task>_<seed>.json      — full machine-readable bundle for
      both splits; aggregate.py consumes these.

Usage
-----
    python -m src.evaluate --checkpoint checkpoints/binary_roberta-base_42
    python -m src.evaluate --checkpoint <dir> --reference
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import DISTORTIONS, MC_CLASSES
from src.metrics import (
    BINARY_CLASSES,
    PAPER_COMPARISON_COLUMNS,
    metric_bundle,
    per_class_table,
    upsert_paper_comparison,
)
from src.train_transformer import (
    ML_COLS,
    TEXT_COL,
    TextDataset,
    collate,
    encode_texts,
    resolve_device,
)


def load_split(splits_dir, name):
    return pd.read_csv(f"{splits_dir}/{name}.csv", encoding="utf-8-sig")


def true_labels(df, task):
    if task == "binary":
        return df["y_bin"].to_numpy().astype(int)
    if task == "multiclass":
        return df["y_mc"].to_numpy().astype(int)
    return df[ML_COLS].to_numpy().astype(int)


@torch.no_grad()
def predict_logits(model, encodings, pad_id, device, batch_size=16):
    model.eval()
    ds = TextDataset(encodings, [0] * len(encodings))  # dummy labels, dropped below
    loader = DataLoader(
        ds, batch_size=batch_size,
        collate_fn=functools.partial(collate, pad_id=pad_id, multilabel=False),
    )
    chunks = []
    for batch in loader:
        batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        chunks.append(model(**batch).logits.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def predictions_from_logits(logits, task, thresholds=None):
    if task == "multilabel":
        probs = 1 / (1 + np.exp(-logits))
        t = np.array(thresholds if thresholds is not None else [0.5] * probs.shape[1])
        return (probs >= t).astype(int)
    return np.argmax(logits, axis=1)


def save_confusion(y_true, y_pred, labels, names, path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    size = max(4.0, len(names) * 0.7)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)
    thresh = 0.5
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on val + test.")
    ap.add_argument("--checkpoint", required=True, help="checkpoint dir with meta.json")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--reference", action="store_true",
                    help="append the Shreevastava2021 literature rows (source=paper)")
    args = ap.parse_args(argv)

    ckpt = Path(args.checkpoint)
    meta = json.loads((ckpt / "meta.json").read_text())
    task = meta["task"]
    model_name = meta["model"]
    seed = meta["seed"]
    max_length = meta["max_length"]
    truncation = meta["truncation"]
    thresholds = meta.get("thresholds")
    tag = model_name.split("/")[-1]

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSequenceClassification.from_pretrained(str(ckpt)).to(device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[eval] {task} {tag} seed={seed} device={device} "
          f"(max_length={max_length}, truncation={truncation})")

    rows, eval_json = [], {"meta": meta, "splits": {}}
    per_class_by_split = {}

    for split in ("val", "test"):
        df = load_split(args.splits, split)
        y_true = true_labels(df, task)
        enc, trunc_rate = encode_texts(df[TEXT_COL], tokenizer, max_length, truncation)
        logits = predict_logits(model, enc, tokenizer.pad_token_id, device, args.batch_size)
        y_pred = predictions_from_logits(logits, task, thresholds)

        row = metric_bundle(task, y_true, y_pred, model_name, seed, split,
                            truncation_rate=round(trunc_rate, 4))
        rows.append(row)
        pc = per_class_table(task, y_true, y_pred)
        per_class_by_split[split] = pc
        eval_json["splits"][split] = {
            "metrics": {k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in row.items()},
            "per_class": pc.to_dict(orient="records"),
            "truncation_rate": round(trunc_rate, 4),
        }

    # (b) merge into paper_comparison.csv
    if args.reference:
        rows.extend(_reference_rows())
    upsert_paper_comparison(rows, out / "paper_comparison.csv")

    # (a) per-class CSV for the test split (the headline evaluation)
    pc_test = per_class_by_split["test"]
    pc_path = out / f"per_class_{tag}_{task}_{seed}.csv"
    pc_test.to_csv(pc_path, index=False)

    # (d) full machine-readable bundle
    (out / f"eval_{tag}_{task}_{seed}.json").write_text(
        json.dumps(eval_json, indent=2), encoding="utf-8")

    # (c) confusion matrices (binary / multiclass only)
    if task in ("binary", "multiclass"):
        df_test = load_split(args.splits, "test")
        yt = true_labels(df_test, task)
        enc, _ = encode_texts(df_test[TEXT_COL], tokenizer, max_length, truncation)
        logits = predict_logits(model, enc, tokenizer.pad_token_id, device, args.batch_size)
        yp = predictions_from_logits(logits, task)
        if task == "binary":
            save_confusion(yt, yp, [0, 1], BINARY_CLASSES,
                           out / f"confusion_{tag}_{task}_{seed}.png",
                           f"{tag} binary (test, row-normalised)")
        else:
            save_confusion(yt, yp, list(range(11)), MC_CLASSES,
                           out / f"confusion_{tag}_{task}_{seed}.png",
                           f"{tag} 11-class (test, row-normalised)")
            # second version with no_distortion dropped
            save_confusion(yt, yp, list(range(1, 11)), DISTORTIONS,
                           out / f"confusion_{tag}_{task}_{seed}_no_nd.png",
                           f"{tag} 10 distortions only (test, row-normalised)")

    # stdout summary: the no_distortion contribution question
    if task == "multiclass":
        nd = pc_test.loc[pc_test["class"] == "no_distortion"].iloc[0]
        wf1 = float([r for r in rows if r["split"] == "test"][0]["weighted_f1"])
        share = (nd["weighted_contribution"] / wf1 * 100) if wf1 else float("nan")
        print("-" * 68)
        print("NO_DISTORTION on TEST (11-class):")
        print(f"  f1={nd['f1']:.3f}  support={int(nd['support'])}  "
              f"weighted_contribution={nd['weighted_contribution']:.3f}")
        print(f"  = {share:.1f}% of the test weighted_f1 ({wf1:.3f})")
        print("-" * 68)

    print(f"Wrote per-class, paper_comparison, eval JSON, and confusion PNG(s) to {out}/")
    return 0


def _reference_rows():
    """The two literature rows, marked source=paper, carrying uncertainty markers."""
    base = {c: "" for c in PAPER_COMPARISON_COLUMNS}
    binary = {**base, "model": "Shreevastava2021", "task": "binary", "seed": "",
              "split": "paper", "n_classes": 2, "weighted_f1": 0.79,
              "source": "paper", "averaging": "UNSPECIFIED"}
    multi = {**base, "model": "Shreevastava2021", "task": "multiclass", "seed": "",
             "split": "paper", "n_classes": "UNSTATED", "weighted_f1": 0.30,
             "source": "paper", "averaging": "weighted"}
    return [binary, multi]


if __name__ == "__main__":
    raise SystemExit(main())
