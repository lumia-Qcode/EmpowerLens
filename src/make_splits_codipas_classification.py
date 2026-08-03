"""
Aggregate CODIPAS.json's span-level rows up to ONE ROW PER SOURCE MESSAGE,
for plain whole-message classification (binary / multiclass / multilabel) —
the same task shape as Annotated_data.csv, reusing train_transformer.py and
evaluate.py completely unmodified.

Why aggregate at all: CODIPAS annotates possibly-several distortion spans
per message, each with its own dominant/secondary label (714 / 3,277
messages have 2+ distinct dominant types across their spans). Feeding the
same message text into a whole-message classifier multiple times, once per
span, would train it against contradictory single-message labels. Instead,
each message becomes exactly one training example whose targets are derived
from the UNION (and, for y_mc, the MODE) of that message's span labels.

Target derivation, mirroring src/data.py's make_targets() semantics:
  y_bin : 1 if ANY span in the message is not "no_distortion".
  y_mc  : the message's single "most representative" distortion — the most
          frequent non-"no_distortion" dominant label among its spans (ties
          broken by DISTORTIONS order for determinism), or "no_distortion"
          if every span in the message is "no_distortion". This is a
          methodological simplification worth stating plainly: CODIPAS does
          not label one dominant type per *message*, only per *span*, so
          "the message's dominant type" is a derived quantity, not a
          measured one.
  y_ml  : 10-column multi-hot union of every non-"no_distortion" dominant
          AND secondary label appearing anywhere among the message's spans
          (same union semantics as src/data.py's y_ml for Annotated_data.csv,
          just unioned over spans instead of over one row's
          dominant+secondary pair).

Output columns match train_transformer.py's expectations exactly:
"Patient Question" (text), "y_bin", "y_mc", "ml_<distortion>" x10 — so
train_transformer.py / evaluate.py run against this data with NO code
changes, only `--splits data/splits_codipas_cls`.

Usage
-----
    python -m src.make_splits_codipas_classification --path CODIPAS.json \
        --out data/splits_codipas_cls --seed 42 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DISTORTIONS, MC_CLASSES
from src.data_codipas import group_label_matrix, load_raw
from src.make_splits_codipas import _sha256, make_group_split_indices

SPLIT_NAMES = ("train", "val", "test")
TEXT_COL = "Patient Question"  # matches train_transformer.py's TEXT_COL exactly


def aggregate_to_message_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per group_id (source message), targets derived per the
    module docstring above. Uses ALL rows regardless of match_type — label
    correctness doesn't depend on whether the char-offset span was locatable,
    only the (not-yet-needed-here) token alignment does."""
    dist_index = {d: i for i, d in enumerate(DISTORTIONS)}
    rows = []
    for gid, sub in df.groupby("group_id"):
        text = sub["text"].iloc[0]
        assert (sub["text"] == text).all(), f"group {gid} has inconsistent text across rows"

        non_nd = [d for d in sub["dominant"] if d != "no_distortion"]
        y_bin = int(len(non_nd) > 0)

        if non_nd:
            counts = Counter(non_nd)
            top_count = max(counts.values())
            # deterministic tie-break: earliest in canonical DISTORTIONS order
            dominant_msg = min(
                (d for d, c in counts.items() if c == top_count),
                key=lambda d: dist_index[d],
            )
        else:
            dominant_msg = "no_distortion"

        ml = [0] * len(DISTORTIONS)
        for d in list(sub["dominant"]) + [s for s in sub["secondary"] if pd.notna(s)]:
            if d != "no_distortion":
                ml[dist_index[d]] = 1

        row = {
            "group_id": gid,
            TEXT_COL: text,
            "dominant_message_level": dominant_msg,
            "y_bin": y_bin,
            "y_mc": MC_CLASSES.index(dominant_msg),
            "n_spans": len(sub),
        }
        for d, v in zip(DISTORTIONS, ml):
            row[f"ml_{d}"] = v
        rows.append(row)

    return pd.DataFrame(rows)


def build_manifest(df, agg, split_frames, source_path, seed) -> dict:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_json": str(source_path),
        "source_sha256": _sha256(source_path),
        "split_unit": "group_id (Id_Patient_Question), aggregated to one row per message",
        "random_state": seed,
        "n_total_span_rows": int(len(df)),
        "n_total_messages": int(len(agg)),
        "y_mc_derivation": "mode of non-no_distortion dominant labels per message, "
                            "tie-broken by DISTORTIONS order; no_distortion if the "
                            "message has zero distorted spans",
        "splits": {},
    }
    for name, sub in split_frames.items():
        manifest["splits"][name] = {
            "n_messages": int(len(sub)),
            "n_distorted": int(sub["y_bin"].sum()),
            "per_class_y_mc": {c: int((sub["y_mc"] == i).sum()) for i, c in enumerate(MC_CLASSES)},
            "ml_positive_per_distortion": {
                d: int(sub[f"ml_{d}"].sum()) for d in DISTORTIONS
            },
        }
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Aggregate CODIPAS to message-level classification splits "
                    "(same column shape as data/splits/*.csv)."
    )
    ap.add_argument("--path", default="CODIPAS.json")
    ap.add_argument("--out", default="data/splits_codipas_cls")
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
    agg = aggregate_to_message_level(df)

    # Reuse the SAME group-level split assignment as make_splits_codipas.py
    # (same seed, same stratify matrix) so the classification and future
    # span-extraction splits partition messages identically.
    gm = group_label_matrix(df)
    train_g, val_g, test_g = make_group_split_indices(gm, seed=args.seed)

    split_frames = {
        "train": agg[agg["group_id"].isin(set(train_g))].reset_index(drop=True),
        "val": agg[agg["group_id"].isin(set(val_g))].reset_index(drop=True),
        "test": agg[agg["group_id"].isin(set(test_g))].reset_index(drop=True),
    }

    # Hard invariants before writing anything.
    id_sets = {n: set(f["group_id"]) for n, f in split_frames.items()}
    assert id_sets["train"].isdisjoint(id_sets["val"])
    assert id_sets["train"].isdisjoint(id_sets["test"])
    assert id_sets["val"].isdisjoint(id_sets["test"])
    assert sum(len(f) for f in split_frames.values()) == len(agg), "message count mismatch after split"
    for name, sub in split_frames.items():
        present = set(sub["y_mc"].unique())
        missing = set(range(11)) - present
        if missing:
            print(f"[warn] {name} split is missing y_mc classes: "
                  f"{[MC_CLASSES[i] for i in missing]} (small-support class, expected on some seeds)")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, sub in split_frames.items():
        sub.to_csv(out_dir / f"{name}.csv", index=False)

    manifest = build_manifest(df, agg, split_frames, args.path, args.seed)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {SPLIT_NAMES} to {out_dir}/ "
          f"({[len(f) for f in split_frames.values()]} messages) + split_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
