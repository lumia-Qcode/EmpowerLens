"""S08: training splits with the synthetic rows added to TRAIN only.

    data/splits_synth/train.csv  = data/splits/train.csv + data/synthetic/synthetic_train_en_final.jsonl
                                   (mapped to the benchmark columns, shuffled, seed 42)
    data/splits_synth/val.csv    = data/splits/val.csv, byte-for-byte copy
    data/splits_synth/split_manifest.json

No test.csv is written: every evaluation uses `src.evaluate --splits data/splits`, so test stays read only
by src/evaluate.py. Val is unchanged, so epoch/threshold selection is on the same rows as the
train-only condition. Synthetic rows never enter val or test.

    venv\\Scripts\\python.exe -m src.make_splits_synth
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data import DISTORTIONS, DOMINANT_COL, ID_COL, MC_INDEX, SECONDARY_COL, TEXT_COL, _canon
from src.make_splits_combined import _per_class_counts, _sha256_file

SPLITS = Path("data/splits")
SYN = Path("data/synthetic/synthetic_train_en_final.jsonl")
OUT = Path("data/splits_synth")
SPAN_COL = "Distorted part"


def synthetic_rows(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    """Map synthetic rows onto the benchmark train.csv columns (y_bin, y_mc, ml_* from dominant + secondary)."""
    out = []
    for r in rows:
        dom = _canon(r["dominant_distortion"])
        sec = _canon(r["secondary_distortion"]) if r.get("secondary_distortion") else None
        row = {c: "" for c in columns}
        row.update({ID_COL: r["id"], TEXT_COL: r["text"], SPAN_COL: r.get("distorted_part") or "",
                    DOMINANT_COL: r["dominant_distortion"], SECONDARY_COL: r.get("secondary_distortion") or "",
                    "y_bin": int(dom != "no_distortion"), "y_mc": MC_INDEX[dom]})
        for d in DISTORTIONS:
            row[f"ml_{d}"] = int(d in (dom, sec))
        out.append(row)
    return pd.DataFrame(out, columns=columns)


def build(splits: Path = SPLITS, syn: Path = SYN, out: Path = OUT, seed: int = 42) -> dict:
    train = pd.read_csv(splits / "train.csv", encoding="utf-8-sig")
    rows = [json.loads(line) for line in syn.read_text(encoding="utf-8").splitlines() if line.strip()]
    s = synthetic_rows(rows, list(train.columns))
    merged = pd.concat([train, s], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "train.csv", index=False, encoding="utf-8")
    shutil.copyfile(splits / "val.csv", out / "val.csv")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "S08: benchmark train + final synthetic rows (train only); val = data/splits/val.csv unchanged; "
                   "no test.csv (evaluate with --splits data/splits)",
        "benchmark_dir": str(splits), "synthetic_file": str(syn), "shuffle_seed": seed,
        "n_train_benchmark": len(train), "n_train_synthetic": len(s), "n_train": len(merged),
        "n_val": int(len(pd.read_csv(out / "val.csv", encoding="utf-8-sig"))),
        "train_per_class_benchmark": _per_class_counts(train),
        "train_per_class_synthetic": _per_class_counts(s),
        "sha256": {"benchmark_train": _sha256_file(splits / "train.csv"), "synthetic": _sha256_file(syn),
                   "val": _sha256_file(out / "val.csv"), "train": _sha256_file(out / "train.csv")},
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S08: benchmark train + synthetic rows, train only")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    m = build(out=Path(args.out))
    print(json.dumps({k: m[k] for k in ("n_train_benchmark", "n_train_synthetic", "n_train", "n_val")}, indent=1))
    print("synthetic per class:", m["train_per_class_synthetic"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
