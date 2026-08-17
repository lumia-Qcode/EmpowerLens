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
* The official train/valid/test split is test-heavy (1,920 / 961 / 6,807). By
  default everything becomes training data and evaluation stays on the untouched
  Annotated test set ("train augmented, test natural").

  The official splits ARE meaningful in one respect: they are **persona-disjoint**
  (231 / 115 / 812 personas, zero pairwise overlap). Since each thought is written
  *from* its persona, any in-domain holdout must respect that boundary or the model
  can match persona-specific phrasing instead of distortion structure. Hence
  ``--holdout official-valid``, which reserves their valid split (961 rows) and
  costs only those rows; reserving their *test* split would cost 6,807 and leave
  less training data than Annotated already provides.

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
    # For SEQUENTIAL fine-tuning (train on PatternReframe, then continue on
    # Annotated) stage 1 must train on PatternReframe ALONE — merging defeats the
    # point. But it still needs a val set for epoch selection and early stopping,
    # and PatternReframe's own official split is not usable here (it is test-heavy
    # and out of domain). So: take val/test from the target dataset, take train
    # from PatternReframe only.
    ap.add_argument("--eval-from", default=None,
                    help="like --merge-into but does NOT prepend that dir's train: "
                         "train is PatternReframe alone, val/test are copied from "
                         "here. For stage 1 of sequential fine-tuning.")
    # marked_patterns intensities run 0-5. The threshold controls how dense the
    # multi-label targets are, and it must be matched to the target data or the
    # label structure itself becomes a distribution shift:
    #     >=1 -> 3.55 labels/row   >=2 -> 1.83   >=3 -> 1.05   >=4 -> 0.61
    # Annotated_data.csv averages 1.3 (dominant + optional secondary, capped at 2),
    # so 3 is the closest match. The primary label is forced on regardless, so no
    # row can end up with zero labels.
    # An IN-DOMAIN diagnostic set. Without one, stage A is scored only on the
    # Annotated test set, and a low number is ambiguous: stage A may have learned
    # nothing, or it may have learned PatternReframe well and simply not transferred.
    # Those call for opposite responses (fix the run vs report a negative result).
    #
    # The official split is NOT used for this: it is test-heavy (1,920/961/6,807),
    # so honouring it would cut training data from 8,712 to 1,920 and destroy the
    # premise of the experiment. A random slice costs ~870 training rows instead.
    # The official splits are PERSONA-DISJOINT — 231/115/812 personas with zero
    # pairwise overlap — and each thought is written FROM its persona, so a random
    # holdout puts the same persona on both sides and lets the model match
    # persona-specific phrasing rather than distortion structure. Default respects
    # that boundary.
    ap.add_argument("--holdout", choices=["none", "official-valid", "random"],
                    default="none",
                    help="in-domain diagnostic set. 'official-valid' uses the authors' "
                         "valid split (961 rows, persona-disjoint, the correct choice); "
                         "'random' takes a stratified --holdout-frac slice and is NOT "
                         "persona-disjoint, so it reads optimistically.")
    ap.add_argument("--holdout-frac", type=float, default=0.1,
                    help="fraction used only by --holdout random")
    ap.add_argument("--holdout-out", default=None,
                    help="where to write the in-domain diagnostic dir "
                         "(default: <out>_holdout)")
    ap.add_argument("--holdout-seed", type=int, default=42)
    # "Discounting the positive" is a whole class of 970 rows with no counterpart in
    # this taxonomy. Merging it into mental_filter is ABLATION-ONLY, off by default:
    # mental_filter has 936 rows, so the merge doubles it to 1,906 and makes it twice
    # the size of every other class, with half of it a different CBT concept. See the
    # note in PATTERN_MAP.
    ap.add_argument("--merge-discounting", action="store_true",
                    help="ABLATION: map 'Discounting the positive' onto mental_filter "
                         "instead of dropping it. Broadens the class definition away "
                         "from the one the test set is annotated with.")
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

    pattern_map = dict(PATTERN_MAP)
    if args.merge_discounting:
        pattern_map["Discounting the positive"] = "mental_filter"
        print("[ablation] 'Discounting the positive' -> mental_filter "
              "(broadens the class beyond the test set's annotation)")

    rows, skipped = [], 0
    for name in ("train", "valid", "test"):
        p = src / f"{name}.txt"
        if not p.exists():
            raise FileNotFoundError(f"missing {p} — point --source at the extracted dir")
        for r in _load_jsonl(p):
            canon = pattern_map.get(r["pattern"])
            if canon is None:            # "Discounting the positive" etc.
                skipped += 1
                continue
            ml = {c: 0 for c in ML_COLS}
            # Multi-label from marked_patterns, not just the primary label: 93% of
            # thoughts carry 2+ patterns with a graded intensity, which matches the
            # multilabel task far better than a single label does.
            for k, v in (r.get("marked_patterns") or {}).items():
                m = pattern_map.get(k)
                if m and str(v).isdigit() and int(v) >= args.min_intensity:
                    ml[f"ml_{m}"] = 1
            ml[f"ml_{canon}"] = 1        # the primary is always present
            rows.append({
                TEXT_COL: r["thought"],
                "y_bin": 1,               # every PatternReframe row is distorted
                "y_mc": MC_CLASSES.index(canon),
                **ml,
                # Provenance, needed for a persona-disjoint holdout. The official
                # splits share ZERO personas (231/115/812, no pairwise overlap), and
                # thoughts are written FROM the persona, so a holdout that ignores
                # this boundary lets the model match persona-specific phrasing
                # instead of distortion structure. Dropped before writing.
                "_src_split": name,
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
    out.mkdir(parents=True, exist_ok=True)

    holdout_info = None
    if args.holdout != "none":
        if args.holdout == "official-valid":
            # The authors' own valid split: 961 rows over 115 personas, disjoint from
            # every other split. Costs 961 training rows. Their TEST split would be
            # the more conventional choice but costs 6,807 — that would cut training
            # data to ~1,900 and leave no more than the Annotated set already has,
            # destroying the premise of the experiment.
            #
            # Training on rows the authors called "test" is deliberate and safe here
            # because no PatternReframe benchmark number is reported: this is
            # borrowed training data plus an in-domain sanity check, not a
            # leaderboard entry.
            mask = df["_src_split"] == "valid"
            hold = df[mask].reset_index(drop=True)
            df = df[~mask].reset_index(drop=True)
        else:
            # Random stratified slice. NOT persona-disjoint — the same persona can
            # land on both sides, so the diagnostic reads optimistically. Kept only
            # as a comparison point for the official-valid holdout.
            keep = []
            for _, g in df.groupby("y_mc"):
                n = max(1, round(len(g) * args.holdout_frac))
                keep.extend(g.sample(n, random_state=args.holdout_seed).index)
            hold = df.loc[sorted(keep)].reset_index(drop=True)
            df = df.drop(index=keep).reset_index(drop=True)

        # Halve it into val/test so the dir is a normal splits dir that evaluate.py
        # can read without special-casing.
        h_out = Path(args.holdout_out or f"{args.out}_holdout")
        h_out.mkdir(parents=True, exist_ok=True)
        mid = len(hold) // 2
        df[KEEP_COLS].to_csv(h_out / "train.csv", index=False)
        hold[KEEP_COLS].iloc[:mid].to_csv(h_out / "val.csv", index=False)
        hold[KEEP_COLS].iloc[mid:].to_csv(h_out / "test.csv", index=False)
        holdout_info = {
            "dir": str(h_out),
            "mode": args.holdout,
            "persona_disjoint": args.holdout == "official-valid",
            "frac": args.holdout_frac if args.holdout == "random" else None,
            "seed": args.holdout_seed,
            "n_holdout": int(len(hold)),
            "n_val": int(mid), "n_test": int(len(hold) - mid),
            "purpose": "IN-DOMAIN diagnostic for stage A. A high score here with a low "
                       "score on the Annotated test set means the domain gap is the "
                       "problem; low on both means stage A itself failed.",
        }
        (h_out / "split_manifest.json").write_text(json.dumps(holdout_info, indent=2),
                                                   encoding="utf-8")
        print(f"[holdout] {len(hold)} rows -> {h_out} "
              f"(val={mid}, test={len(hold) - mid}); train reduced to {len(df)}")

    df = df[KEEP_COLS]          # drop _src_split; provenance never reaches a CSV

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(src),
        "purpose": "PatternReframe as Stage-2 TRAINING augmentation. No No-Distortion "
                   "rows exist, so this cannot be used for the binary task. Official "
                   "train/valid/test split is ignored; everything becomes train.",
        "n_rows": int(len(df)),
        "n_skipped_unmappable_pattern": skipped,
        "merge_discounting": bool(args.merge_discounting),
        "holdout": holdout_info,
        "per_class": {c.replace("ml_", ""): int(df[c].sum()) for c in ML_COLS},
        "median_words": int(df[TEXT_COL].str.split().str.len().median()),
        "caveats": [
            "emotional_reasoning has ZERO coverage (intensity 0 in every source row)",
            "'Discounting the positive' dropped — no counterpart, merging would corrupt mental_filter",
            "median 17 words vs ~129 in Annotated_data.csv — a real distribution shift",
        ],
    }

    if args.merge_into and args.eval_from:
        print("--merge-into and --eval-from are mutually exclusive", file=sys.stderr)
        return 1

    if args.eval_from:
        base = Path(args.eval_from)
        df.to_csv(out / "train.csv", index=False)
        for n in ("val", "test"):
            pd.read_csv(base / f"{n}.csv", encoding="utf-8-sig")[KEEP_COLS].to_csv(
                out / f"{n}.csv", index=False)
        n_val = len(pd.read_csv(out / "val.csv"))
        manifest.update({
            "eval_from": str(base),
            "n_train_patternreframe": int(len(df)),
            "note": "SEQUENTIAL stage 1: train is PatternReframe ONLY; val/test copied "
                    "from --eval-from so epoch selection targets the real distribution. "
                    "Stage 2 continues from this checkpoint on the target train set.",
        })
        print(f"train {len(df)} (PatternReframe only) — val/test from {base} "
              f"(val={n_val})")
    elif args.merge_into:
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
