"""
Build leakage-free copies of the derived split dirs.

THE PROBLEM THIS EXISTS TO FIX
------------------------------
Measured on the shipped ``data/splits_combined`` (2026-08-22):

* **195 of the 253 Annotated test texts — 77% of the test set — appear verbatim
  in ``train.csv``.** 201 val texts leak the same way.
* Only **36%** of those leaked rows carry the same label they carry in test. So
  the model is not merely shown the answer; it is shown the same text with two
  different answers, from two annotation schemes that disagree.
* ``train.csv`` holds 4,645 rows but only 3,019 unique texts — **1,626 exact
  duplicate rows**, which silently up-weights whatever is duplicated.

``data/splits_codipas_cls/train.csv`` carries the same 195 Annotated-test texts.

Any score produced by training on those dirs and testing on the Annotated test
set is inflated by an unknown amount and is not comparable to anything. Every
Experiment 3-8 run in ``experiments/kaggle_runner_flat_experiments.ipynb``
pointed at ``--splits data/splits_combined``, so all of them are affected.

WHAT THIS SCRIPT DOES
---------------------
Writes NEW directories rather than editing the existing ones — the project rule
is that splits are immutable, and the broken dirs must stay readable so old
results remain traceable to the data that produced them.

For each source dir:

1. ``val.csv`` and ``test.csv`` are copied **byte-identical**. The exam does not
   change; only what the model is allowed to study changes.
2. From ``train.csv`` it drops every row whose text appears in the **reference**
   dir's val or test (default ``data/splits`` — the Annotated yardstick), and
   also in the source dir's own val/test.
3. Exact duplicate texts within train are collapsed to one row.
4. A ``clean_manifest.json`` records exactly what was removed and why, so the
   deletion is auditable rather than a silent shrink.

Matching is on the **normalized text** (stripped, whitespace collapsed,
casefolded), not on ``Id_Number``: the same passage appears under different ids
across corpora, which is precisely how the leak got in.

Usage
-----
    python -m src.make_splits_clean                      # both derived dirs
    python -m src.make_splits_clean --check              # report only, write nothing
    python -m src.make_splits_clean --source data/splits_combined \\
        --dest data/splits_combined_clean
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TEXT_COL = "Patient Question"
ENC = "utf-8-sig"

# (source, destination) pairs rebuilt by default. data/splits itself is NOT in
# this list: it is the reference, it has no leak, and it must never be rewritten.
DEFAULT_PAIRS = [
    ("data/splits_combined", "data/splits_combined_clean"),
    ("data/splits_codipas_cls", "data/splits_codipas_clean"),
]


def normalize(s: pd.Series) -> pd.Series:
    """Text key for matching across corpora.

    Whitespace and case are normalized because the same passage is re-typed with
    different spacing between corpora; matching on Id_Number would miss all of
    it. This is deliberately conservative — it catches exact and
    whitespace/case variants, not paraphrases.
    """
    return (s.astype(str)
             .str.strip()
             .str.replace(r"\s+", " ", regex=True)
             .str.casefold())


def read(d: str | Path, name: str) -> pd.DataFrame:
    return pd.read_csv(Path(d) / f"{name}.csv", encoding=ENC)


def audit(source: str, reference: str = "data/splits") -> dict:
    """Measure the leak in one split dir without changing anything."""
    tr = read(source, "train")
    ref_val, ref_test = read(reference, "val"), read(reference, "test")
    own_val, own_test = read(source, "val"), read(source, "test")

    k = normalize(tr[TEXT_COL])
    ref_block = set(normalize(ref_val[TEXT_COL])) | set(normalize(ref_test[TEXT_COL]))
    own_block = set(normalize(own_val[TEXT_COL])) | set(normalize(own_test[TEXT_COL]))
    ref_hit, own_hit = k.isin(ref_block), k.isin(own_block)

    ref_test_keys = set(normalize(ref_test[TEXT_COL]))
    leaked_test = k[k.isin(ref_test_keys)].nunique()

    return {
        "source": source, "reference": reference,
        "train_rows": int(len(tr)),
        "train_unique_texts": int(k.nunique()),
        "duplicate_rows_within_train": int(len(tr) - k.nunique()),
        "train_rows_in_reference_val_or_test": int(ref_hit.sum()),
        "train_rows_in_own_val_or_test": int(own_hit.sum()),
        "reference_test_rows_leaked": int(leaked_test),
        "reference_test_size": int(len(ref_test)),
        "reference_test_pct_leaked": round(100 * leaked_test / max(len(ref_test), 1), 1),
    }


def clean_one(source: str, dest: str, reference: str = "data/splits",
              force: bool = False) -> dict:
    dest_p = Path(dest)
    if dest_p.exists() and not force:
        raise SystemExit(
            f"{dest} already exists. Splits are immutable — delete it "
            f"deliberately or pass --force if you really mean to regenerate.")

    before = audit(source, reference)
    tr = read(source, "train")

    ref_block = (set(normalize(read(reference, "val")[TEXT_COL]))
                 | set(normalize(read(reference, "test")[TEXT_COL])))
    own_block = (set(normalize(read(source, "val")[TEXT_COL]))
                 | set(normalize(read(source, "test")[TEXT_COL])))
    block = ref_block | own_block

    k = normalize(tr[TEXT_COL])
    kept = tr[~k.isin(block)].copy()

    # Collapse exact duplicates AFTER the block filter, keeping the first
    # occurrence so row order stays deterministic.
    kept_keys = normalize(kept[TEXT_COL])
    kept = kept[~kept_keys.duplicated(keep="first")].reset_index(drop=True)

    dest_p.mkdir(parents=True, exist_ok=True)
    kept.to_csv(dest_p / "train.csv", index=False, encoding="utf-8")
    # val/test copied byte-identical: the exam must not change.
    for name in ("val", "test"):
        shutil.copyfile(Path(source) / f"{name}.csv", dest_p / f"{name}.csv")
    src_manifest = Path(source) / "split_manifest.json"
    if src_manifest.exists():
        shutil.copyfile(src_manifest, dest_p / "split_manifest.json")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "src/make_splits_clean.py",
        "source": source, "reference": reference, "dest": dest,
        "rule": ("train rows whose normalized text appears in the reference "
                 "val/test or the source's own val/test are dropped; exact "
                 "duplicate texts within train are collapsed; val/test copied "
                 "unchanged"),
        "before": before,
        "after": {
            "train_rows": int(len(kept)),
            "rows_removed": int(before["train_rows"] - len(kept)),
            "removed_pct": round(100 * (before["train_rows"] - len(kept))
                                 / max(before["train_rows"], 1), 1),
        },
        "verified_after": audit(dest, reference),
    }
    (dest_p / "clean_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    v = manifest["verified_after"]
    if v["train_rows_in_reference_val_or_test"] or v["duplicate_rows_within_train"]:
        raise SystemExit(f"CLEAN FAILED for {dest}: {v}")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--source", default=None)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--reference", default="data/splits",
                    help="the yardstick whose val/test must never be trained on")
    ap.add_argument("--check", action="store_true",
                    help="report the leak and exit without writing anything")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing destination dir")
    args = ap.parse_args(argv)

    pairs = ([(args.source, args.dest)] if args.source
             else [p for p in DEFAULT_PAIRS if Path(p[0]).exists()])

    if args.check:
        rows = [audit(s, args.reference) for s, _ in pairs]
        # Audit the cleaned copies too, so "did the fix work" is answerable
        # from the same command rather than taken on trust.
        rows += [audit(d, args.reference) for _, d in pairs
                 if d and Path(d).exists()]
        rows.append(audit(args.reference, args.reference))   # the reference itself
        df = pd.DataFrame(rows)[
            ["source", "train_rows", "duplicate_rows_within_train",
             "train_rows_in_reference_val_or_test", "reference_test_rows_leaked",
             "reference_test_pct_leaked"]]
        print(df.to_string(index=False))
        # A LEAK is fatal: the model studied the exam. Duplicate rows are only
        # advisory — they skew the training distribution but invalidate nothing.
        leaked = df[df["train_rows_in_reference_val_or_test"] > 0]
        dupes = df[(df["train_rows_in_reference_val_or_test"] == 0)
                   & (df["duplicate_rows_within_train"] > 0)]
        for _, r in dupes.iterrows():
            print(f"\nnote: {r['source']} has {r['duplicate_rows_within_train']} "
                  f"duplicate text(s) in train but NO leak — training "
                  f"distribution is mildly skewed, results stay valid.")
        if len(leaked):
            print(f"\nLEAK: {len(leaked)} dir(s) train on the yardstick's val/test. "
                  f"Any score from them is inflated by an unknown amount.\n"
                  f"Run without --check to write clean copies.")
            return 1
        print("\nNo leaks.")
        return 0

    for src, dst in pairs:
        if not dst:
            raise SystemExit("--source needs a matching --dest")
        m = clean_one(src, dst, args.reference, args.force)
        b, a = m["before"], m["after"]
        print(f"{src} -> {dst}")
        print(f"  train {b['train_rows']} -> {a['train_rows']} "
              f"(-{a['rows_removed']}, {a['removed_pct']}%)")
        print(f"  removed {b['train_rows_in_reference_val_or_test']} leaked rows "
              f"({b['reference_test_pct_leaked']}% of the yardstick test set was "
              f"visible) and {b['duplicate_rows_within_train']} duplicates")
        print(f"  val/test copied unchanged; manifest at {dst}/clean_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
