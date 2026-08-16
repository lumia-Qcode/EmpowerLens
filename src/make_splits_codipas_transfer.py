"""
Build a LEAKAGE-FREE CODIPAS -> Annotated transfer split.

The question this answers
------------------------
"Does training on CODIPAS help on our actual target task?" — which requires
training on CODIPAS and evaluating on the *frozen Annotated test set*, not on
CODIPAS's own test set. Every existing CODIPAS result is evaluated in-domain and
therefore says nothing about transfer.

Why this needs its own script
-----------------------------
CODIPAS is NOT an independent corpus. Measured on the shipped data:

    CODIPAS unique texts                     3,276
      also present in Annotated train        2,016
      also present in Annotated val            252  (of 253)
      also present in Annotated test          252  (of 253)
      genuinely new                            756

Merging naively puts 77% of the Annotated test set into training. That is exactly
the bug that invalidated data/splits_combined (194/253 val, 189/253 test leaked),
and it inflated macro_f1_10 by ~50%.

This script removes every training row whose text appears in the Annotated val or
test set, then attaches the FROZEN Annotated val/test unchanged.

What the result can and cannot show
-----------------------------------
Read the manifest's `label_agreement` before interpreting anything. On the 2,017
shared texts, CODIPAS's DERIVED y_mc matches the human annotation only ~36.9% of
the time, and the disagreement is directional — the most common patterns are all
`human distortion -> codipas no_distortion`.

So a low score here is the expected outcome and does NOT mean "more data doesn't
help". It means CODIPAS's aggregation rule does not reproduce human annotation.
Report it as a label-transfer / annotation-agreement result, not as an
augmentation result.

Usage
-----
    python -m src.make_splits_codipas_transfer --out data/splits_codipas_transfer
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.data import DISTORTIONS, MC_CLASSES, TEXT_COL

ML_COLS = [f"ml_{d}" for d in DISTORTIONS]
KEEP_COLS = [TEXT_COL, "y_bin", "y_mc"] + ML_COLS


def _key(s):
    """Match texts up to whitespace and case.

    Exact string equality is too strict: the same reflection reaches the two
    corpora through different exports and differs in whitespace and casing, so an
    exact-match dedup would silently leave leaked rows behind.
    """
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _read(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, encoding="utf-8-sig")
    d["_k"] = d[TEXT_COL].map(_key)
    return d


def main(argv=None):
    ap = argparse.ArgumentParser(description="Leakage-free CODIPAS -> Annotated transfer split.")
    ap.add_argument("--codipas", default="data/splits_codipas_cls",
                    help="CODIPAS splits dir; ALL of its splits become training data")
    ap.add_argument("--annotated", default="data/splits",
                    help="frozen Annotated splits; its val/test are used verbatim")
    ap.add_argument("--out", default="data/splits_codipas_transfer")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"REFUSING to overwrite {out} (pass --force)", file=sys.stderr)
        return 1

    ann, cod_dir = Path(args.annotated), Path(args.codipas)
    ann_train = _read(ann / "train.csv")
    ann_val = _read(ann / "val.csv")
    ann_test = _read(ann / "test.csv")

    # CODIPAS's own train/val/test boundary is irrelevant here — none of it is the
    # evaluation set any more, so all of it is training data.
    cod = pd.concat([_read(cod_dir / f"{s}.csv") for s in ("train", "val", "test")],
                    ignore_index=True).drop_duplicates("_k")
    n_before = len(cod)

    blocked = set(ann_val._k) | set(ann_test._k)
    train = cod[~cod._k.isin(blocked)].reset_index(drop=True)
    n_removed = n_before - len(train)

    # How much of the surviving pool is genuinely new, vs Annotated train relabelled.
    in_ann_train = train._k.isin(set(ann_train._k))

    # Label agreement on the shared texts — the number that decides how this result
    # should be read.
    merged = ann_train[["_k", "y_mc"]].merge(cod[["_k", "y_mc"]], on="_k",
                                             suffixes=("_ann", "_cod"))
    agree = float((merged.y_mc_ann == merged.y_mc_cod).mean()) if len(merged) else None
    disagree = merged[merged.y_mc_ann != merged.y_mc_cod]
    top_conf = (disagree.groupby([disagree.y_mc_ann.map(lambda i: MC_CLASSES[i]),
                                 disagree.y_mc_cod.map(lambda i: MC_CLASSES[i])])
                .size().sort_values(ascending=False).head(10))

    out.mkdir(parents=True, exist_ok=True)
    train[KEEP_COLS].to_csv(out / "train.csv", index=False)
    # val/test are the FROZEN Annotated ones, copied verbatim. This is the whole
    # point of the script — the evaluation set must not move.
    ann_val[KEEP_COLS].to_csv(out / "val.csv", index=False)
    ann_test[KEEP_COLS].to_csv(out / "test.csv", index=False)

    # Prove it rather than assert it.
    tr_k = set(train._k)
    leak_val = len(tr_k & set(ann_val._k))
    leak_test = len(tr_k & set(ann_test._k))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Train on CODIPAS, evaluate on the FROZEN Annotated test set. "
                   "A label-transfer study, NOT a data-augmentation study — see "
                   "label_agreement below.",
        "sources": {"codipas": str(cod_dir), "annotated": str(ann)},
        "n_codipas_unique": int(n_before),
        "n_removed_because_in_annotated_val_or_test": int(n_removed),
        "n_train": int(len(train)),
        "n_train_overlapping_annotated_train": int(in_ann_train.sum()),
        "n_train_new_texts": int((~in_ann_train).sum()),
        "n_val": int(len(ann_val)), "n_test": int(len(ann_test)),
        "leak_check": {"train_in_val": leak_val, "train_in_test": leak_test},
        "label_agreement": {
            "n_shared_with_annotated_train": int(len(merged)),
            "y_mc_agreement": round(agree, 4) if agree is not None else None,
            "top_disagreements_annotated_to_codipas": {f"{a} -> {c}": int(n)
                                                       for (a, c), n in top_conf.items()},
        },
        "caveats": [
            "CODIPAS y_mc is DERIVED by an aggregation rule, not human-annotated.",
            "Only ~756 of its texts are new; the rest is Annotated train relabelled.",
            "A low score is the EXPECTED outcome and does not show that more data "
            "fails to help — it shows the two label sets disagree.",
        ],
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"train {len(train)}  (dropped {n_removed} rows that appear in Annotated val/test)")
    print(f"  of which {int(in_ann_train.sum())} overlap Annotated train, "
          f"{int((~in_ann_train).sum())} are new texts")
    print(f"val {len(ann_val)}  test {len(ann_test)}   (frozen Annotated, copied verbatim)")
    print(f"LEAK CHECK: train-in-val={leak_val}  train-in-test={leak_test}")
    if leak_val or leak_test:
        print("LEAKAGE REMAINS — do not train on this.", file=sys.stderr)
        return 1
    if agree is not None:
        print(f"label agreement with human annotation on {len(merged)} shared texts: {agree:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
