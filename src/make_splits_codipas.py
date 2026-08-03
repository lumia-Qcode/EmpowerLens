"""
Generate frozen train/val/test splits for CODIPAS.json — **once**, mirroring
the discipline of ``src/make_splits.py``, but split at the **group level**
(``Id_Patient_Question`` — the source message), never the row level.

Why group-level: CODIPAS annotates multiple spans per source message (3,277
unique messages -> 5,055 span rows). Splitting rows independently would put
near-duplicate or literally identical source text in both train and test —
a leakage bug worse than the one already fixed in cd_pipeline.py, since here
it would leak the *exact same input text* across splits, just with a
different span highlighted. Every row sharing a group_id goes to the same
split, full stop.

Stratification: since MultilabelStratifiedShuffleSplit needs one label
vector per unit-being-split, we stratify on the **group-level** label matrix
from ``data_codipas.group_label_matrix`` (10 distortion columns, unioned
across a group's spans, +1 binary column) — analogous to
``src/make_splits.py``'s 11-column stratify matrix, just aggregated to the
group first.

Usage
-----
    python -m src.make_splits_codipas --path CODIPAS.json \
        --out data/splits_codipas --seed 42 --force
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
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.data import DISTORTIONS, MC_CLASSES
from src.data_codipas import group_label_matrix, load_raw

SPLIT_NAMES = ("train", "val", "test")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_group_split_indices(group_matrix: pd.DataFrame, seed: int = 42):
    """Return (train_groups, val_groups, test_groups) — arrays of group_id,
    80/10/10, multilabel-stratified on the group-level distortion matrix."""
    L = group_matrix[[*DISTORTIONS, "y_bin"]].to_numpy()
    X = np.zeros((len(group_matrix), 1))

    s1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_i, temp_i = next(s1.split(X, L))

    s2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_local, test_local = next(s2.split(X[temp_i], L[temp_i]))
    val_i, test_i = temp_i[val_local], temp_i[test_local]

    ids = group_matrix["group_id"].to_numpy()
    return ids[train_i], ids[val_i], ids[test_i]


def _rows_for_groups(df: pd.DataFrame, group_ids) -> pd.DataFrame:
    mask = df["group_id"].isin(set(group_ids))
    return df[mask].reset_index(drop=True)


def build_manifest(df, split_frames, source_path, seed) -> dict:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_json": str(source_path),
        "source_sha256": _sha256(source_path),
        "split_unit": "group_id (Id_Patient_Question) — NOT row-level",
        "random_state": seed,
        "proportions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "n_total_rows": int(len(df)),
        "n_total_groups": int(df["group_id"].nunique()),
        "splits": {},
    }
    for name, sub in split_frames.items():
        counts = {c: int((sub["dominant"] == c).sum()) for c in MC_CLASSES}
        manifest["splits"][name] = {
            "n_rows": int(len(sub)),
            "n_groups": int(sub["group_id"].nunique()),
            "n_distorted_rows": int((sub["y_bin"] == 1).sum()),
            "per_class_row_counts": counts,
            "match_type_counts": sub["match_type"].value_counts().to_dict(),
        }
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate frozen group-level CODIPAS splits.")
    ap.add_argument("--path", default="CODIPAS.json")
    ap.add_argument("--out", default="data/splits_codipas")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    targets = [out_dir / f"{n}.csv" for n in SPLIT_NAMES]
    manifest_path = out_dir / "split_manifest.json"
    existing = [p for p in [*targets, manifest_path] if p.exists()]
    if existing and not args.force:
        print("REFUSING to overwrite existing split files (pass --force):\n  "
              + "\n  ".join(str(p) for p in existing), file=sys.stderr)
        return 1

    df = load_raw(args.path)
    gm = group_label_matrix(df)
    train_g, val_g, test_g = make_group_split_indices(gm, seed=args.seed)

    # Hard invariants before writing anything.
    all_g = np.concatenate([train_g, val_g, test_g])
    assert len(set(all_g.tolist())) == len(all_g), "group split indices overlap"
    assert set(all_g.tolist()) == set(gm["group_id"].tolist()), "not every group was assigned a split"

    split_frames = {
        "train": _rows_for_groups(df, train_g),
        "val": _rows_for_groups(df, val_g),
        "test": _rows_for_groups(df, test_g),
    }

    # No group leaks across splits — the whole point of this splitter.
    id_sets = {n: set(f["group_id"]) for n, f in split_frames.items()}
    assert id_sets["train"].isdisjoint(id_sets["val"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["val"].isdisjoint(id_sets["test"])
    assert sum(len(s) for s in id_sets.values()) == df["group_id"].nunique()
    assert sum(len(f) for f in split_frames.values()) == len(df), "row count mismatch after group split"

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sub in split_frames.items():
        sub.to_csv(out_dir / f"{name}.csv", index=False)

    manifest = build_manifest(df, split_frames, args.path, args.seed)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {SPLIT_NAMES} to {out_dir}/ "
          f"({[len(f) for f in split_frames.values()]} rows) + split_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
