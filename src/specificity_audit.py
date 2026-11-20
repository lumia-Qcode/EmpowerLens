"""
D27 row D-04: does every evaluation run report specificity and false-positive rate?

Walks every ``eval_*.json`` under the given folders and, for each run and split, reads:

  * binary      specificity = recall of ``non_distorted`` (the D13 / FR-10 number)
  * multiclass  specificity = recall of ``no_distortion`` (11-class runs)
  * multilabel  ``macro_specificity_10`` if evaluate.py recorded it (per-label specificity, averaged
                over the 10 types); older multilabel runs have no specificity -> "not recorded"

FPR = 1 - specificity. Everything comes from the eval JSONs: no model is run and test.csv is never
opened. Files without a ``splits`` block (e.g. span-selector evals) are not classifier evals and are
skipped.

    python -m src.specificity_audit --out docs/test_evidence/D-04_specificity_audit.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_ROOTS = ["results", "results_RUN2"]
NEG_CLASS = {"binary": "non_distorted", "multiclass": "no_distortion"}


def _recall(per_class, name):
    for r in per_class or []:
        if r.get("class") == name:
            return float(r["recall"])
    return None


def audit(roots) -> pd.DataFrame:
    rows = []
    for root in map(Path, roots):
        for p in sorted(root.rglob("eval_*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict) or "splits" not in data:
                continue
            meta = data.get("meta", {})
            for split, block in data["splits"].items():
                m = block.get("metrics", {})
                task = meta.get("task") or m.get("task")
                spec, basis = None, "not recorded"
                if task in NEG_CLASS:
                    spec = _recall(block.get("per_class"), NEG_CLASS[task])
                    basis = f"recall of {NEG_CLASS[task]}" if spec is not None else "not recorded"
                elif task == "multilabel" and block.get("macro_specificity_10") is not None:
                    spec, basis = float(block["macro_specificity_10"]), "macro_specificity_10 (per label)"
                rows.append({
                    "folder": p.parent.as_posix(), "file": p.name, "task": task,
                    "model": meta.get("model") or m.get("model"), "seed": meta.get("seed", m.get("seed")),
                    "split": split, "specificity": spec, "fpr": None if spec is None else 1 - spec,
                    "basis": basis,
                })
    df = pd.DataFrame(rows, columns=["folder", "file", "task", "model", "seed", "split",
                                     "specificity", "fpr", "basis"])
    return df.astype({"specificity": float, "fpr": float})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="D-04: specificity / FPR present in every eval run?")
    ap.add_argument("roots", nargs="*", default=DEFAULT_ROOTS)
    ap.add_argument("--out", default=None, help="CSV path for the per-run table")
    args = ap.parse_args(argv)
    df = audit(args.roots)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.round(4).to_csv(args.out, index=False, encoding="utf-8")
    runs = df.drop_duplicates(["folder", "file"])
    print(f"{len(runs)} eval files, {len(df)} (file, split) rows")
    has = df.assign(has=df.specificity.notna())
    print("\nfiles with specificity on every split, by task:")
    per_file = has.groupby(["task", "folder", "file"]).has.all().reset_index()
    print(per_file.groupby("task").has.agg(["sum", "count"]).rename(columns={"sum": "with", "count": "files"}).to_string())
    missing = per_file[~per_file.has]
    print(f"\nfolders with runs missing specificity ({missing.folder.nunique()}):")
    for (folder, task), g in missing.groupby(["folder", "task"]):
        print(f"  {folder}  [{task}]  {len(g)} file(s)")
    if args.out:
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
