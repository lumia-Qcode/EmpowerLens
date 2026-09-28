"""
src/reannotate_merge_audit.py — Stage 3 of model-assisted re-annotation.

Merges a COMPLETED audit_queue.csv (i.e. corrected_primary / corrected_secondary
filled in by a human reviewer) back into the corpus, producing
Annotated_data_v2.csv with an explicit provenance column so every label's
origin is traceable — this is the file you'd cite in the methodology chapter
as "how relabeling was done," not a silent overwrite of Annotated_data.csv
(which stays frozen per CLAUDE.md).

Rows NOT in the audit queue (bucket C — model agreed confidently) keep their
original label with label_source="original_unreviewed". Rows IN the queue but
left blank by the reviewer (still pending) also keep the original label, with
label_source="pending_review", so partially-completed audit passes are safe
to merge without losing track of what's actually been checked.

Usage
-----
    python -m src.reannotate_merge_audit \\
        --audit results/reannotation/audit_queue.csv \\
        --out data/Annotated_data_v2.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data import ID_COL, load_raw


def main(argv=None):
    ap = argparse.ArgumentParser(description="Merge a completed audit queue into a new labeled dataset.")
    ap.add_argument("--path", default="Annotated_data.csv", help="original frozen source (read-only)")
    ap.add_argument("--audit", default="results/reannotation/audit_queue.csv")
    ap.add_argument("--out", default="data/Annotated_data_v2.csv")
    args = ap.parse_args(argv)

    df = load_raw(args.path)
    audit = pd.read_csv(args.audit, encoding="utf-8-sig")

    audit_by_id = audit.set_index(ID_COL)
    out = df.copy()
    out["label_source"] = "original_unreviewed"
    out["reviewer"] = ""
    out["review_notes"] = ""

    n_relabeled, n_confirmed, n_pending = 0, 0, 0

    for i, row_id in enumerate(out[ID_COL]):
        if row_id not in audit_by_id.index:
            continue  # bucket C, never queued
        a = audit_by_id.loc[row_id]
        corrected_primary = str(a.get("corrected_primary", "")).strip()
        corrected_secondary = str(a.get("corrected_secondary", "")).strip()

        if not corrected_primary or corrected_primary.lower() == "nan":
            out.loc[i, "label_source"] = "pending_review"
            n_pending += 1
            continue

        # human confirmed the model's suggestion verbatim vs. actively relabeled
        model_suggestion = str(a.get("model_predicted_primary", "")).strip()
        if corrected_primary == model_suggestion:
            out.loc[i, "label_source"] = "human_confirmed_model"
            n_confirmed += 1
        else:
            out.loc[i, "label_source"] = "human_relabeled"
            n_relabeled += 1

        out.loc[i, "Dominant Distortion"] = corrected_primary
        out.loc[i, "Secondary Distortion (Optional)"] = (
            corrected_secondary if corrected_secondary and corrected_secondary.lower() != "nan" else pd.NA
        )
        out.loc[i, "reviewer"] = a.get("reviewer", "")
        out.loc[i, "review_notes"] = a.get("review_notes", "")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"label_source breakdown:\n{out['label_source'].value_counts()}\n")
    print(f"  human_relabeled     : {n_relabeled}  (reviewer overrode the model)")
    print(f"  human_confirmed_model: {n_confirmed}  (reviewer agreed with model's suggestion)")
    print(f"  pending_review      : {n_pending}  (still blank in audit_queue.csv)")
    print(f"\nWrote {out_path} — Annotated_data.csv itself is untouched (still frozen).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
