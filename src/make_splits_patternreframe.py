"""
Convert PatternReframe (Maddela et al., ACL 2023) into this repo's column contract.

PatternReframe is ~9,688 crowdsourced "unhelpful thoughts", each written to exhibit
a given pattern, conditioned on a persona. Download (the parl.ai URL redirects
twice; this is the final one, 2.4 MB, no ParlAI install needed):

    https://dl.fbaipublicfiles.com/parlai/reframe_thoughts/reframe_thoughts_v0.1.tar.gz
    sha256 bfbfc61c26341dd64b59945c3d290caba67fa2db435fb01ac309cef295222c99

Source schema — JSONL, one object per line, in train.txt / valid.txt / test.txt:

    persona          the persona the thought was written for
    pattern          THE primary pattern (one of 10)
    pattern_def      its definition
    thought          the text                                   <- what we train on
    marked_patterns  {pattern: intensity "0".."4"} over 12 keys  <- multi-label signal
    reframes         list of positive reframes (unused here)

What this script does and does NOT give you
-------------------------------------------
* **Stage 2 only.** Every row is distorted — there are no No-Distortion examples —
  so this is useless for the binary task but an exact structural match for the
  cascade's Stage 2, which trains on distorted rows across these same 10 classes.
* **Emotional Reasoning gets nothing.** The key exists in ``marked_patterns`` but
  its intensity is 0 in all 9,688 rows. Nine of the ten classes get augmented and
  one does not, so that class becomes relatively rarer than it is today.
* **Length mismatch is the main risk.** These thoughts are a median of 17 words;
  Annotated_data.csv reflections are a median of 129. Training on one-liners and
  testing on paragraphs is a real distribution shift, and it is the most likely
  reason this fails to help. Report that if it does.
* The official train/valid/test split is test-heavy (1,920 / 961 / 6,807) and is
  **ignored** — everything is emitted as training data. Evaluation stays on the
  untouched Annotated test set ("train augmented, test natural").

The 2.4 MB tarball is COMMITTED at data/patternreframe/, so none of the below needs
network access. It is extracted on demand to data/patternreframe/extracted/, which
is gitignored — 23 MB of derived text does not belong in git.

Usage
-----
    python -m src.make_splits_patternreframe --out data/splits_patternreframe

    # append to an existing Stage 2 training set (val/test copied through unchanged)
    python -m src.make_splits_patternreframe \\
        --merge-into data/splits_stage2 --out data/splits_stage2_pr --force

    # or point at an already-extracted dir (e.g. a fresh download on Kaggle)
    python -m src.make_splits_patternreframe --source /kaggle/working/reframe_thoughts_dataset ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data import DISTORTIONS, MC_CLASSES, TEXT_COL

ML_COLS = [f"ml_{d}" for d in DISTORTIONS]
KEEP_COLS = [TEXT_COL, "y_bin", "y_mc"] + ML_COLS

# PatternReframe pattern -> this repo's canonical label.
# Nine map cleanly. Two PatternReframe keys have no counterpart here:
#   "Discounting the positive" — close to mental_filter but a distinct CBT category,
#      so it is DROPPED rather than silently merged (that would corrupt mental_filter).
#   "None" — a placeholder, never used.
# And our emotional_reasoning has no PatternReframe counterpart (all-zero intensity).
PATTERN_MAP = {
    "Catastrophizing": "magnification",
    "Overgeneralization": "overgeneralization",
    "Personalization": "personalization",
    "Black-and-white or polarized thinking / All or nothing thinking": "all_or_nothing",
    "Mental filtering": "mental_filter",
    "Jumping to conclusions: mind reading": "mind_reading",
    "Jumping to conclusions: Fortune-telling": "fortune_telling",
    "Should statements": "should_statements",
    "Labeling and mislabeling": "labeling",
}
DROPPED = {"Discounting the positive", "Emotional reasoning", "None"}


def _load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# The tarball is committed (2.4 MB) so this runs offline and reproducibly. The
# EXTRACTED text is 23 MB and is gitignored — it is regenerated from the tarball
# on demand, never committed.
DEFAULT_SOURCE = Path("data/patternreframe/reframe_thoughts_v0.1.tar.gz")
SOURCE_SHA256 = "bfbfc61c26341dd64b59945c3d290caba67fa2db435fb01ac309cef295222c99"


def _resolve_source(src: Path) -> Path:
    """Accept either an extracted dir or the .tar.gz, and return the dir.

    Extracting on demand keeps the repo small while still making the pipeline
    runnable with no network: the archive is the committed artifact, the text
    files are derived.
    """
    if src.is_dir():
        return src
    if not src.exists():
        raise FileNotFoundError(
            f"{src} not found. Either pass --source <extracted dir>, or place the "
            f"2.4 MB tarball at {DEFAULT_SOURCE} (see the module docstring for the URL)."
        )

    digest = hashlib.sha256(src.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        # A mismatch means a different release or a corrupt download — refuse
        # rather than silently building splits from unknown data.
        raise ValueError(
            f"{src} sha256 mismatch:\n  expected {SOURCE_SHA256}\n  got      {digest}"
        )

    dest = src.parent / "extracted"
    marker = dest / "train.txt"
    if not marker.exists():
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(src) as t:
            # filter="data" (3.12+) blocks absolute paths, .. traversal, symlinks
            # and device files. Guarded because it is not in every version.
            try:
                t.extractall(dest, filter="data")
            except TypeError:
                t.extractall(dest)
        if not marker.exists():          # the archive nests one directory deep
            for cand in dest.rglob("train.txt"):
                dest = cand.parent
                break
        print(f"[extract] {src} -> {dest}")
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description="PatternReframe -> repo column contract.")
    ap.add_argument("--source", default=str(DEFAULT_SOURCE),
                    help="the committed .tar.gz (default, extracted on demand) OR a "
                         "dir already containing train.txt / valid.txt / test.txt")
    ap.add_argument("--out", default="data/splits_patternreframe")
    ap.add_argument("--merge-into", default=None,
                    help="existing splits dir; its train is prepended and its "
                         "val/test are copied through UNCHANGED")
    # marked_patterns intensities run 0-5. The threshold controls how dense the
    # multi-label targets are, and it must be matched to the target data or the
    # label structure itself becomes a distribution shift:
    #     >=1 -> 3.55 labels/row   >=2 -> 1.83   >=3 -> 1.05   >=4 -> 0.61
    # Annotated_data.csv averages 1.3 (dominant + optional secondary, capped at 2),
    # so 3 is the closest match. The primary label is forced on regardless, so no
    # row can end up with zero labels.
    ap.add_argument("--min-intensity", type=int, default=3,
                    help="marked_patterns intensity at or above which a pattern counts "
                         "as present (0-5 scale; default 3 ~= 1.3 labels/row, matching "
                         "Annotated_data.csv)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"REFUSING to overwrite {out} (pass --force)", file=sys.stderr)
        return 1
    src = _resolve_source(Path(args.source))

    rows, skipped = [], 0
    for name in ("train", "valid", "test"):
        p = src / f"{name}.txt"
        if not p.exists():
            raise FileNotFoundError(f"missing {p} — point --source at the extracted dir")
        for r in _load_jsonl(p):
            canon = PATTERN_MAP.get(r["pattern"])
            if canon is None:            # "Discounting the positive" etc.
                skipped += 1
                continue
            ml = {c: 0 for c in ML_COLS}
            # Multi-label from marked_patterns, not just the primary label: 93% of
            # thoughts carry 2+ patterns with a graded intensity, which matches the
            # multilabel task far better than a single label does.
            for k, v in (r.get("marked_patterns") or {}).items():
                m = PATTERN_MAP.get(k)
                if m and str(v).isdigit() and int(v) >= args.min_intensity:
                    ml[f"ml_{m}"] = 1
            ml[f"ml_{canon}"] = 1        # the primary is always present
            rows.append({
                TEXT_COL: r["thought"],
                "y_bin": 1,               # every PatternReframe row is distorted
                "y_mc": MC_CLASSES.index(canon),
                **ml,
            })

    df = pd.DataFrame(rows)[KEEP_COLS].drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(src),
        "purpose": "PatternReframe as Stage-2 TRAINING augmentation. No No-Distortion "
                   "rows exist, so this cannot be used for the binary task. Official "
                   "train/valid/test split is ignored; everything becomes train.",
        "n_rows": int(len(df)),
        "n_skipped_unmappable_pattern": skipped,
        "per_class": {c.replace("ml_", ""): int(df[c].sum()) for c in ML_COLS},
        "median_words": int(df[TEXT_COL].str.split().str.len().median()),
        "caveats": [
            "emotional_reasoning has ZERO coverage (intensity 0 in every source row)",
            "'Discounting the positive' dropped — no counterpart, merging would corrupt mental_filter",
            "median 17 words vs ~129 in Annotated_data.csv — a real distribution shift",
        ],
    }

    if args.merge_into:
        base = Path(args.merge_into)
        base_train = pd.read_csv(base / "train.csv", encoding="utf-8-sig")
        missing = [c for c in KEEP_COLS if c not in base_train.columns]
        if missing:
            raise ValueError(f"{base}/train.csv is missing {missing}")
        merged = pd.concat([base_train[KEEP_COLS], df], ignore_index=True)
        merged = merged.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
        merged.to_csv(out / "train.csv", index=False)
        # val/test pass through untouched — augment train only, never the eval sets.
        for n in ("val", "test"):
            pd.read_csv(base / f"{n}.csv", encoding="utf-8-sig")[KEEP_COLS].to_csv(
                out / f"{n}.csv", index=False)
        manifest.update({
            "merged_into": str(base),
            "n_train_base": int(len(base_train)),
            "n_train_patternreframe": int(len(df)),
            "n_train_total": int(len(merged)),
            "note": "val/test copied unchanged from --merge-into; only train is augmented",
        })
        print(f"train {len(base_train)} + {len(df)} -> {len(merged)} (after dedup)")
    else:
        df.to_csv(out / "train.csv", index=False)
        print(f"Wrote {len(df)} rows (train only — no val/test, this is augmentation data)")

    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if skipped:
        print(f"\n[note] skipped {skipped} rows whose pattern has no counterpart here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
