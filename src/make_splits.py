"""
Generate the frozen 80/10/10 train/val/test splits — **once**.

The splits are the backbone of the whole project: every model reads these exact
files, so they are generated a single time, written to ``data/splits/`` as CSV,
committed to git, and never regenerated casually. Re-running this script refuses
to overwrite existing split files unless ``--force`` is passed.

Stratification uses ``MultilabelStratifiedShuffleSplit`` over the 10-column
multi-label matrix (``y_ml``) with the binary label appended as an 11th column,
so that (a) no rare distortion lands entirely in one split and (b) the
distorted / non-distorted ratio is balanced across splits. The 80/10/10 target
is reached with two nested splits: 80/20, then the 20 halved into 10/10.

Usage
-----
    python -m src.make_splits [--path Annotated_data.csv]
                              [--out data/splits] [--seed 42] [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

from src.data import (
    DISTORTIONS,
    ID_COL,
    MC_CLASSES,
    load_raw,
    make_targets,
)

SPLIT_NAMES = ("train", "val", "test")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pkg_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _stratify_matrix(y_bin: np.ndarray, y_ml: np.ndarray) -> np.ndarray:
    """10 distortion columns + the binary label as an 11th column."""
    return np.hstack([y_ml, y_bin.reshape(-1, 1)])


def make_split_indices(y_bin, y_ml, seed=42):
    """
    Return ``(train_idx, val_idx, test_idx)`` for an 80/10/10 multi-label
    stratified split. Deterministic given ``seed``.
    """
    n = len(y_bin)
    L = _stratify_matrix(y_bin, y_ml)
    X = np.zeros((n, 1))  # features are irrelevant to the stratifier

    # 80 / 20
    s1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, temp_idx = next(s1.split(X, L))

    # split the 20 into 10 / 10
    s2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_local, test_local = next(s2.split(X[temp_idx], L[temp_idx]))
    val_idx, test_idx = temp_idx[val_local], temp_idx[test_local]

    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _per_class_counts(y_mc_subset: np.ndarray) -> dict:
    """Count of each of the 11 multi-class labels present in a split."""
    counts = {name: 0 for name in MC_CLASSES}
    for idx, c in zip(*np.unique(y_mc_subset, return_counts=True)):
        counts[MC_CLASSES[int(idx)]] = int(c)
    return counts


def build_manifest(df, splits, y_bin, y_mc, y_ml, source_path, seed) -> dict:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(source_path),
        "source_sha256": _sha256(source_path),
        "encoding": "utf-8-sig",
        "random_state": seed,
        "proportions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "n_total": int(len(df)),
        "splits": {},
        "library_versions": {
            "python": platform.python_version(),
            "numpy": _pkg_version("numpy"),
            "pandas": _pkg_version("pandas"),
            "scikit-learn": _pkg_version("scikit-learn"),
            "iterative-stratification": _pkg_version("iterative-stratification"),
        },
    }
    for name, idx in zip(SPLIT_NAMES, splits):
        manifest["splits"][name] = {
            "n": int(len(idx)),
            "n_distorted": int(y_bin[idx].sum()),
            "n_non_distorted": int((y_bin[idx] == 0).sum()),
            "per_class": _per_class_counts(y_mc[idx]),
            "ml_positive_per_distortion": {
                d: int(y_ml[idx, j].sum()) for j, d in enumerate(DISTORTIONS)
            },
        }
    return manifest


def _split_frame(df, idx, y_bin, y_mc, y_ml) -> pd.DataFrame:
    """Original columns plus the derived targets, for one split."""
    out = df.iloc[idx].copy().reset_index(drop=True)
    out["y_bin"] = y_bin[idx]
    out["y_mc"] = y_mc[idx]
    for j, d in enumerate(DISTORTIONS):
        out[f"ml_{d}"] = y_ml[idx, j].astype(int)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate frozen train/val/test splits.")
    ap.add_argument("--path", default="Annotated_data.csv", help="source CSV")
    ap.add_argument("--out", default="data/splits", help="output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite existing splits")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
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

    df = load_raw(args.path)
    y_bin, y_mc, y_ml = make_targets(df)
    splits = make_split_indices(y_bin, y_ml, seed=args.seed)
    train_idx, val_idx, test_idx = splits

    # Hard invariants before we write anything.
    all_idx = np.concatenate(splits)
    assert len(all_idx) == len(df), "splits do not cover every row exactly once"
    assert len(set(all_idx.tolist())) == len(df), "splits overlap"
    ids = df[ID_COL].to_numpy()
    id_sets = [set(ids[idx].tolist()) for idx in splits]
    assert id_sets[0].isdisjoint(id_sets[1]) and id_sets[0].isdisjoint(id_sets[2]) \
        and id_sets[1].isdisjoint(id_sets[2]), "Id_Number overlaps across splits"
    for name, idx in zip(SPLIT_NAMES, splits):
        present = set(np.unique(y_mc[idx]).tolist())
        assert present == set(range(11)), f"{name} is missing classes: {set(range(11)) - present}"

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, idx, target in zip(SPLIT_NAMES, splits, targets):
        _split_frame(df, idx, y_bin, y_mc, y_ml).to_csv(target, index=False)

    manifest = build_manifest(df, splits, y_bin, y_mc, y_ml, args.path, args.seed)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(json.dumps(manifest, indent=2))
    print(
        f"\nWrote {SPLIT_NAMES} to {out_dir}/ "
        f"({len(train_idx)}/{len(val_idx)}/{len(test_idx)}) + split_manifest.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
