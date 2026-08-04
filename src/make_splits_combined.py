"""
Merge the Annotated_data.csv train split with the CODIPAS message-level
train split into one larger training set — **train only**.

Why train-only merging: val.csv and test.csv are left as the frozen
Annotated_data.csv splits (data/splits/{val,test}.csv), unchanged. CODIPAS's
y_mc is a *derived* label (mode of non-"no_distortion" span dominants per
message — see src/make_splits_codipas_classification.py), not a directly
hand-annotated single-label, so folding CODIPAS rows into val/test would
quietly change what "correct" means for the benchmark you're comparing
against roberta-base / mentalbert / Shreevastava2021 (0.30 weighted_f1).
Folding it into TRAIN only is safe: more gradient signal, same yardstick.

Column contract: only the columns train_transformer.py actually reads are
kept — "Patient Question", y_bin, y_mc, ml_<distortion> x10 — so this run
directory is a drop-in --splits target for the existing pipeline with zero
code changes.

Usage
-----
    python -m src.make_splits_combined \
        --annotated data/splits --codipas data/splits_codipas_cls \
        --out data/splits_combined --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DISTORTIONS, MC_CLASSES, TEXT_COL

SPLIT_NAMES = ("train", "val", "test")
KEEP_COLS = [TEXT_COL, "y_bin", "y_mc"] + [f"ml_{d}" for d in DISTORTIONS]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_keep_cols(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return df[KEEP_COLS].copy()


def _per_class_counts(df: pd.DataFrame) -> dict:
    return {c: int((df["y_mc"] == i).sum()) for i, c in enumerate(MC_CLASSES)}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Merge Annotated_data train + CODIPAS-cls train; "
                    "val/test stay the frozen Annotated_data split."
    )
    ap.add_argument("--annotated", default="data/splits",
                    help="dir with Annotated_data train/val/test.csv")
    ap.add_argument("--codipas", default="data/splits_codipas_cls",
                    help="dir with CODIPAS message-level train/val/test.csv")
    ap.add_argument("--out", default="data/splits_combined")
    ap.add_argument("--seed", type=int, default=42, help="shuffle seed for the merged train set")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    ann_dir, cod_dir, out_dir = Path(args.annotated), Path(args.codipas), Path(args.out)
    targets = [out_dir / f"{n}.csv" for n in SPLIT_NAMES]
    manifest_path = out_dir / "split_manifest.json"
    existing = [p for p in [*targets, manifest_path] if p.exists()]
    if existing and not args.force:
        print("REFUSING to overwrite existing split files (pass --force):\n  "
              + "\n  ".join(str(p) for p in existing), file=sys.stderr)
        return 1

    for d in (ann_dir, cod_dir):
        for n in SPLIT_NAMES:
            if not (d / f"{n}.csv").exists():
                raise FileNotFoundError(f"missing {d / f'{n}.csv'} — run the upstream split script first")

    ann_train = _load_keep_cols(ann_dir / "train.csv")
    cod_train = _load_keep_cols(cod_dir / "train.csv")
    ann_val = _load_keep_cols(ann_dir / "val.csv")
    ann_test = _load_keep_cols(ann_dir / "test.csv")

    rng = np.random.default_rng(args.seed)
    combined_train = pd.concat([ann_train, cod_train], ignore_index=True)
    combined_train = combined_train.iloc[rng.permutation(len(combined_train))].reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    combined_train.to_csv(out_dir / "train.csv", index=False)
    ann_val.to_csv(out_dir / "val.csv", index=False)
    ann_test.to_csv(out_dir / "test.csv", index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "train = Annotated_data.csv train + CODIPAS-cls train (concatenated, shuffled); "
                   "val/test = Annotated_data.csv only (frozen benchmark, unchanged)",
        "annotated_source_dir": str(ann_dir),
        "codipas_source_dir": str(cod_dir),
        "shuffle_seed": args.seed,
        "n_train_annotated": int(len(ann_train)),
        "n_train_codipas": int(len(cod_train)),
        "n_train_combined": int(len(combined_train)),
        "n_val": int(len(ann_val)),
        "n_test": int(len(ann_test)),
        "train_per_class_annotated": _per_class_counts(ann_train),
        "train_per_class_codipas": _per_class_counts(cod_train),
        "train_per_class_combined": _per_class_counts(combined_train),
        "val_per_class": _per_class_counts(ann_val),
        "test_per_class": _per_class_counts(ann_test),
        "source_file_sha256": {
            "annotated_train": _sha256_file(ann_dir / "train.csv"),
            "codipas_train": _sha256_file(cod_dir / "train.csv"),
            "annotated_val": _sha256_file(ann_dir / "val.csv"),
            "annotated_test": _sha256_file(ann_dir / "test.csv"),
        },
    }

    # Sanity checks before declaring victory.
    assert len(combined_train) == len(ann_train) + len(cod_train), "row count mismatch after concat"
    assert set(combined_train.columns) == set(KEEP_COLS)
    for name in ("val", "test"):
        present = set(pd.read_csv(out_dir / f"{name}.csv")["y_mc"].unique())
        missing = set(range(11)) - present
        if missing:
            print(f"[warn] {name} split missing y_mc classes: {[MC_CLASSES[i] for i in missing]}")

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(
        f"\nWrote train.csv ({len(combined_train)} rows = "
        f"{len(ann_train)} Annotated + {len(cod_train)} CODIPAS), "
        f"val.csv ({len(ann_val)}), test.csv ({len(ann_test)}) to {out_dir}/"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
