"""
D14 - relabelled copy of the frozen benchmark splits.

RUN (from repo root):
    venv\\Scripts\\python.exe -m src.make_splits_relabelled            # refuses to overwrite
    venv\\Scripts\\python.exe -m src.make_splits_relabelled --force    # rebuild

Input : data/splits/{train,val,test}.csv                         (frozen split, never re-split)
        reannotation_2026-09-27/outputs/relabel/merged_labels.csv (annotator votes)
Output: data/splits_relabelled/{train,val,test}.csv + manifest.json

Same Id_Numbers in the same files and order as data/splits. A row is relabelled when it has a
majority label under the D08 rule (reannotation_2026-09-27/scripts/label_changes.py):
  - a vote counts only if Primary is filled, it was not filled in from the original label
    (CS-majors blanks) and Unsure is not ticked (Unsure rows were never adjudicated);
  - majority = one 11-way label (calibrated pessimism -> No Distortion) with > half the valid votes.
New Dominant = that label. New Secondary = empty for No Distortion; otherwise the Secondary given by
more than half of the voters who chose the majority Dominant (a lone annotator's own Secondary),
dropped if it repeats the Dominant. The annotator's pair replaces the original pair, so an original
secondary is not carried over. y_bin / y_mc / ml_* are rebuilt with src.data.make_targets.
`label_provenance` = reannotated (has a majority) or original (labels untouched).
"""
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import (CSV_ENCODING, DISTORTIONS, DOMINANT_COL, ID_COL, LABEL_CANON, MC_CLASSES,
                      SECONDARY_COL, TEXT_COL, make_targets)

SRC_DIR = Path("data/splits")
OUT_DIR = Path("data/splits_relabelled")
MERGED = Path("reannotation_2026-09-27/outputs/relabel/merged_labels.csv")
LABEL_CHANGES = Path("reannotation_2026-09-27/scripts/label_changes.py")
SPLITS = ("train", "val", "test")
NO_DISTORTION = "No Distortion"
CP_LABEL = "No Distortion (calibrated pessimism)"
TARGET_COLS = ["y_bin", "y_mc"] + [f"ml_{d}" for d in DISTORTIONS]


def _load_classify():
    spec = importlib.util.spec_from_file_location("label_changes", LABEL_CHANGES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify


_classify = _load_classify()


def _clean(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) or not str(x).strip() else str(x).strip()


def relabel_row(orig_dom, orig_sec, votes):
    """votes: [(Primary, Secondary), ...] valid votes only. Returns (dominant, secondary) or None."""
    primaries = [p for p, _ in votes]
    new, _ = _classify(orig_dom, _clean(orig_sec), primaries)
    if new is None:
        return None
    if new == NO_DISTORTION:
        return new, None
    agree = [_clean(s) for p, s in votes if p == new]
    secs = pd.Series([s for s in agree if s is not None], dtype=object).value_counts()
    sec = secs.index[0] if len(secs) and secs.iloc[0] * 2 > len(agree) else None
    return new, (None if sec == new else sec)


def majority_table():
    m = pd.read_csv(MERGED, encoding="utf-8")
    filled = m["filled_from_original"].astype(str).str.lower() == "true"
    valid = m[m["Primary"].notna() & ~filled & m["Unsure"].isna()]
    info = m.drop_duplicates("Row ID").set_index("Row ID")
    table, reasons = {}, {}
    for rid in info.index:
        v = valid[valid["Row ID"] == rid]
        if v.empty:
            r = m[m["Row ID"] == rid]
            reasons[int(rid)] = ("filled from original" if filled[r.index].any() else
                                 "Unsure only" if (r["Primary"].notna() & r["Unsure"].notna()).any()
                                 else "blank Primary")
            continue
        res = relabel_row(str(info.at[rid, "original_dominant"]).strip(), info.at[rid, "original_secondary"],
                          list(zip(v["Primary"], v["Secondary"])))
        if res is None:
            reasons[int(rid)] = "no majority (split vote)"
            continue
        table[int(rid)] = (*res, len(v))
    return table, reasons


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _per_class(y_mc):
    return {c: int((y_mc == i).sum()) for i, c in enumerate(MC_CLASSES)}


def build(force=False):
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()) and not force:
        raise SystemExit(f"{OUT_DIR} already exists; pass --force to rebuild it.")
    table, reasons = majority_table()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "D14 relabelled splits (same rows, same files as data/splits)",
        "sources": {str(p): _sha(p) for p in [MERGED, LABEL_CHANGES, *(SRC_DIR / f"{s}.csv" for s in SPLITS)]},
        "rules": {
            "valid_vote": "Primary filled, not filled_from_original, Unsure not ticked",
            "majority": "one 11-way label (calibrated pessimism = No Distortion) with > half the valid votes",
            "dominant": "the majority label",
            "secondary": "empty for No Distortion; else the Secondary given by > half of the voters who chose "
                         "the majority Dominant, dropped if equal to the Dominant; original secondary not kept",
            "label_provenance": "reannotated = row has a majority label; original = labels untouched",
        },
        "rows_with_votes_but_not_relabelled": {
            k: int(v) for k, v in pd.Series(reasons, dtype=object).value_counts().items()},
        "splits": {},
    }
    for split in SPLITS:
        src = pd.read_csv(SRC_DIR / f"{split}.csv", encoding=CSV_ENCODING)
        out = src.copy()
        hit = out[ID_COL].map(lambda i: int(i) in table)
        for i in out.index[hit]:
            dom, sec, _ = table[int(out.at[i, ID_COL])]
            out.at[i, DOMINANT_COL] = dom
            out.at[i, SECONDARY_COL] = sec if sec is not None else np.nan
        for col in (DOMINANT_COL, SECONDARY_COL):
            bad = set(out[col].dropna().map(str).str.strip()) - set(LABEL_CANON)
            assert not bad, f"{split}: unmapped labels {bad}"
        y_bin, y_mc, y_ml = make_targets(out)
        out["y_bin"], out["y_mc"] = y_bin, y_mc
        for j, d in enumerate(DISTORTIONS):
            out[f"ml_{d}"] = y_ml[:, j].astype(int)
        out["label_provenance"] = np.where(hit, "reannotated", "original")

        # Integrity: same rows, same order, same text; untouched rows identical to the frozen file.
        assert len(out) == len(src) and out[ID_COL].tolist() == src[ID_COL].tolist(), split
        assert out[TEXT_COL].equals(src[TEXT_COL]), split
        keep = ~hit.to_numpy()
        for col in [DOMINANT_COL, SECONDARY_COL, *TARGET_COLS]:
            assert out.loc[keep, col].fillna("").astype(str).equals(src.loc[keep, col].fillna("").astype(str)), (split, col)
        assert (out["y_bin"] == out[[f"ml_{d}" for d in DISTORTIONS]].max(axis=1)).all(), split

        out[[*src.columns, "label_provenance"]].to_csv(OUT_DIR / f"{split}.csv", index=False, encoding="utf-8")
        dom_changed = out[DOMINANT_COL].ne(src[DOMINANT_COL])
        sec_changed = out[SECONDARY_COL].fillna("").ne(src[SECONDARY_COL].fillna(""))
        manifest["splits"][split] = {
            "n": int(len(out)),
            "n_reannotated": int(hit.sum()),
            "n_original": int((~hit).sum()),
            "n_dominant_changed": int(dom_changed.sum()),
            "n_secondary_changed": int(sec_changed.sum()),
            "n_any_label_changed": int((dom_changed | sec_changed).sum()),
            "n_binary_changed": int(out["y_bin"].ne(src["y_bin"]).sum()),
            "per_class_before": _per_class(src["y_mc"].to_numpy()),
            "per_class_after": _per_class(out["y_mc"].to_numpy()),
        }
        print(split, {k: v for k, v in manifest["splits"][split].items() if not k.startswith("per_class")})
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--force", action="store_true", help="overwrite data/splits_relabelled/")
    build(ap.parse_args().force)


if __name__ == "__main__":
    main()
