"""S10: score the S09 checks against the review-sheet labels, and build the final synthetic training file.

Input: the labels for data/synthetic/review_sheet.xlsx, either
  * the filled workbook itself (the "Label" tab; headers mapped back by name), or
  * a csv with review_id, dominant (+ optional secondary, distorted_part, unsure).

A row is judged WRONGLY LABELLED when the reviewer's main label differs from the panel's label
(the two No Distortion options count as the same). For each check (neighbours / model / close votes)
and for the random rows: rows flagged, rows judged wrong, precision = wrong / flagged. The random
rows' error rate is the baseline; a check is only useful well above it.

Rows judged wrong are thrown away, never relabelled: the final training file is the S07 clean file
minus those rows.

    # provisional (disclosed model-assisted labels):
    python -m src.synth_label_check --labels data/synthetic/review_labels_model.csv --labeller "model-assisted (provisional)"
    # the human S10 check, once Izza has filled the sheet:
    python -m src.synth_label_check --labels data/synthetic/review_sheet.xlsx --labeller "Izza"
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data import LABEL_CANON

KEY = Path("data/synthetic/review_key.csv")
CLEAN = Path("data/synthetic/synthetic_train_en_clean.jsonl")
FINAL = Path("data/synthetic/synthetic_train_en_final.jsonl")
OUT = Path("results/results_synth_label_check")
CHECKS = {"neighbours": "flag_neighbours", "model": "flag_model", "close votes": "flag_close_votes"}
CP_LABEL = "No Distortion (calibrated pessimism)"
HEADERS = {"Review ID": "review_id", "Main distortion\n(required)": "dominant",
           "Second distortion\n(optional)": "secondary", "Distorted words\n(copy from the text)": "distorted_part",
           "Unsure?\n(Yes / blank)": "unsure", "Notes\n(optional)": "notes"}


def canon(label: str) -> str:
    label = str(label or "").strip()
    if label == CP_LABEL:
        label = "No Distortion"
    if label not in LABEL_CANON:
        raise ValueError(f"unknown label {label!r}")
    return LABEL_CANON[label]


def read_labels(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path, sheet_name="Label", dtype=str).rename(columns=HEADERS)
    else:
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.fillna("")
    missing = df[df.dominant.str.strip() == ""].review_id.tolist()
    if missing:
        raise SystemExit(f"{len(missing)} rows have no main label yet, e.g. {missing[:5]}: finish the sheet first")
    return df


def read_key(path: Path = KEY) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", dtype=str)


def score(labels: pd.DataFrame, key: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = key.merge(labels[["review_id", "dominant"]], on="review_id", how="left", validate="one_to_one")
    if m.dominant.isna().any():
        raise SystemExit(f"labels missing for {m[m.dominant.isna()].review_id.tolist()[:5]} ...")
    m["wrong"] = [canon(a) != canon(b) for a, b in zip(m.dominant, m.panel_label)]
    rows = []
    for name, col in CHECKS.items():
        sel = m[m[col] == "yes"]
        rows.append({"check": name, "rows_flagged": len(sel), "rows_judged_wrong": int(sel.wrong.sum())})
    any_flag = m[(m[list(CHECKS.values())] == "yes").any(axis=1)]
    rows.append({"check": "any check", "rows_flagged": len(any_flag), "rows_judged_wrong": int(any_flag.wrong.sum())})
    rnd = m[m.random == "yes"]
    rows.append({"check": "random (baseline)", "rows_flagged": len(rnd), "rows_judged_wrong": int(rnd.wrong.sum())})
    t = pd.DataFrame(rows)
    t["precision"] = (t.rows_judged_wrong / t.rows_flagged).round(3)
    return t, m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S10: score the S09 checks and build the final synthetic file")
    ap.add_argument("--labels", required=True, help="filled review_sheet.xlsx, or a csv of labels")
    ap.add_argument("--labeller", required=True, help="who produced the labels (recorded in the summary)")
    args = ap.parse_args(argv)

    labels, key = read_labels(Path(args.labels)), read_key()
    table, merged = score(labels, key)
    wrong_ids = set(merged[merged.wrong].id)

    clean = [json.loads(line) for line in CLEAN.read_text(encoding="utf-8").splitlines() if line.strip()]
    final = [r for r in clean if r["id"] not in wrong_ids]
    with FINAL.open("w", encoding="utf-8") as fh:
        for r in final:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "precision_table.csv", index=False)
    removed = merged[merged.wrong][["review_id", "id", "panel_label", "dominant"]].rename(
        columns={"dominant": "reviewer_label"}).sort_values("id")
    removed.to_csv(OUT / "removed_rows.csv", index=False)
    confusion = Counter(zip(removed.panel_label, removed.reviewer_label))
    summary = {
        "labeller": args.labeller, "labels_file": str(args.labels),
        "n_reviewed": len(merged), "n_removed": len(wrong_ids),
        "n_clean": len(clean), "n_final": len(final),
        "final_per_type": dict(Counter(r["dominant_distortion"] for r in final).most_common()),
        "removed_per_panel_label": dict(Counter(removed.panel_label).most_common()),
        "panel_vs_reviewer": [{"panel": a, "reviewer": b, "n": n} for (a, b), n in confusion.most_common()],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"labeller: {args.labeller}\n")
    print(table.to_string(index=False))
    print(f"\nremoved {len(wrong_ids)} rows -> {FINAL}: {len(final)} rows")
    print(json.dumps(summary["final_per_type"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
