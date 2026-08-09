"""
Derive Stage 2 (distorted-only) train/val/test splits from an existing,
already-frozen splits directory (e.g. data/splits_combined) — the second
half of the cascade architecture:

    Input -> Stage 1: binary model (trained on the FULL splits dir, unchanged)
          -> if distorted -> Stage 2: multi-label head (trained ONLY on
             distorted rows, produced by this script)

Why this is safe and not a new leakage risk: it does not re-split anything.
Every row here already belongs to train/val/test exactly as make_splits.py
(or make_splits_combined.py) decided; this script only FILTERS each of those
three files down to y_bin == 1 and re-writes them under a new directory. The
row-level train/val/test boundary from the source dir is inherited unchanged.

Column contract matches make_splits_combined.py's: only the columns
train_transformer.py reads (Patient Question, y_bin, y_mc, ml_<distortion> x10)
are kept, so this is a drop-in --splits target with zero code changes.

Usage
-----
    python -m src.make_splits_cascade --source data/splits_combined \
        --out data/splits_stage2 --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data import DISTORTIONS, TEXT_COL

SPLIT_NAMES = ("train", "val", "test")
KEEP_COLS = [TEXT_COL, "y_bin", "y_mc"] + [f"ml_{d}" for d in DISTORTIONS]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Filter an existing splits dir down to distorted-only rows for Stage 2."
    )
    ap.add_argument("--source", default="data/splits_combined",
                    help="parent splits dir (already frozen; train/val/test.csv)")
    ap.add_argument("--out", default="data/splits_stage2")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    src_dir, out_dir = Path(args.source), Path(args.out)
    targets = [out_dir / f"{n}.csv" for n in SPLIT_NAMES]
    manifest_path = out_dir / "split_manifest.json"
    existing = [p for p in [*targets, manifest_path] if p.exists()]
    if existing and not args.force:
        print(
            "REFUSING to overwrite existing split files (pass --force to override):\n  "
            + "\n  ".join(str(p) for p in existing),
            file=sys.stderr,
        )
        return 1

    for n in SPLIT_NAMES:
        if not (src_dir / f"{n}.csv").exists():
            raise FileNotFoundError(f"missing {src_dir / f'{n}.csv'} — point --source at a real splits dir")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Stage 2 splits = --source splits filtered to y_bin==1 (distorted only). "
                   "No re-splitting: row-level train/val/test boundaries are inherited "
                   "unchanged from --source.",
        "source_dir": str(src_dir),
        "n_per_split": {},
        "n_removed_no_distortion": {},
        "per_class_train": {},
        "source_file_sha256": {},
    }

    for n in SPLIT_NAMES:
        df = pd.read_csv(src_dir / f"{n}.csv", encoding="utf-8-sig")
        missing = [c for c in KEEP_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{src_dir / f'{n}.csv'} is missing required columns: {missing}")

        n_before = len(df)
        distorted = df[df["y_bin"] == 1][KEEP_COLS].reset_index(drop=True)
        distorted.to_csv(out_dir / f"{n}.csv", index=False)

        manifest["n_per_split"][n] = int(len(distorted))
        manifest["n_removed_no_distortion"][n] = int(n_before - len(distorted))
        manifest["source_file_sha256"][n] = _sha256_file(src_dir / f"{n}.csv")

        if n == "train":
            for j, d in enumerate(DISTORTIONS):
                manifest["per_class_train"][d] = int(distorted[f"ml_{d}"].sum())

    # Sanity check: every one of the 10 distortion columns must have at least
    # one positive row in train, or Stage 2 can never learn that class at all.
    dead_classes = [d for d, n in manifest["per_class_train"].items() if n == 0]
    if dead_classes:
        print(f"[warn] Stage 2 train has ZERO positive rows for: {dead_classes}", file=sys.stderr)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(
        f"\nWrote Stage 2 (distorted-only) train/val/test to {out_dir}/ "
        f"(train={manifest['n_per_split']['train']}, "
        f"val={manifest['n_per_split']['val']}, "
        f"test={manifest['n_per_split']['test']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
