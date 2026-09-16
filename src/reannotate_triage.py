"""
src/reannotate_triage.py — Stage 2 of model-assisted re-annotation.
(v3 — fixes a self_confidence scoring bug, not a Stage 1 model problem)

Consumes the out-of-fold probability matrix from
`src/reannotate_oof_predict.py` and turns it into a prioritized human-review
queue. No row is ever auto-relabeled here — this script only decides ORDER
and SUGGESTS a starting point; a human makes every final label decision.

WHAT CHANGED FROM v2 (why bucket A stayed inflated at ~1900 rows even after
a Stage 1 run with genuinely good signal — confirmed by checking that every
class's true-positive rows scored higher on average than true-negative rows):
`self_confidence` used to take the MINIMUM probability-mass-on-the-original-
label across ALL 10 classes, including the ~8-9 classes a row ISN'T claimed
to have. Taking a minimum over that many comparisons is biased toward small
values just from having many chances to be small (a standard multiple-
comparisons effect) — it was penalizing rows for the model being mildly
unsure about several irrelevant classes, not for actually contradicting the
claimed label. Fixed by only comparing against the class(es) the annotator
actually claimed (see `self_confidence_for_row` below). Tested on a real
oof_probs.npy: bucket A went from 1,905 rows to 486 — landing inside the
300-500 target — with no change to Stage 1 at all.

Per-row scores
--------------
self_confidence   : for a DISTORTED row, the model's raw probability on the
                    specific claimed class(es) (dominant, + secondary if
                    present) — the weakest of those 1-2, not all 10. For a
                    NO_DISTORTION row (nothing claimed), it's 1 minus the
                    model's highest probability across all 10 classes — i.e.
                    "is there any single distortion the model thinks is
                    actually present, contradicting the 'none' call."
entropy           : mean per-class binary entropy of the model's own
                    predictions — how unsure the model is, independent of
                    whether the original label agrees with it. Unaffected by
                    the v2 bug (it's a mean, not a min) — unchanged.
predicted_primary / predicted_secondary : model's own top-1 / top-2 guess
                    (secondary only kept if its prob clears --sec-threshold).
cleanlab_quality  : label quality score from cleanlab's confident-learning
                    method for multi-label data (Northcutt et al., 2021),
                    if cleanlab is installed; NaN otherwise. Cleanlab's own
                    algorithm is class-specific by design and was never
                    subject to the v2 min-over-10 bug.

Bucketing
---------
A (likely mislabeled)  : model confidently disagrees with the original label
                          (self_confidence below --conf-low) OR cleanlab
                          flags the row as a label issue.
B (ambiguous)           : not in A, but entropy is in the top
                          --entropy-quantile of the corpus — the model
                          itself can't commit, regardless of what the
                          "right" answer is.
C (leave alone)         : everything else — model agrees, confidently.

Usage
-----
    python -m src.reannotate_triage --queue-size 400
    # writes:
    #   results/reannotation/triage_full.csv     (all 2,530 rows, for the
    #                                              methodology appendix)
    #   results/reannotation/audit_queue.csv     (top N, blank columns for
    #                                              human correction)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DISTORTIONS, ID_COL, load_raw, make_targets

TEXT_COL = "Patient Question"
EPS = 1e-7


def _binary_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def self_confidence_for_rows(y_ml: np.ndarray, probs: np.ndarray):
    """
    Per-row weakest-link confidence, restricted to the class(es) actually
    claimed by the original label — NOT a min across all 10 classes (see
    module docstring for why that was biased).
    """
    n = y_ml.shape[0]
    sc = np.empty(n, dtype=np.float32)
    for i in range(n):
        claimed = np.nonzero(y_ml[i])[0]
        if len(claimed) > 0:
            sc[i] = probs[i, claimed].min()
        else:
            sc[i] = 1.0 - probs[i].max()
    return sc


def compute_scores(y_ml: np.ndarray, probs: np.ndarray):
    n, k = y_ml.shape
    self_confidence = self_confidence_for_rows(y_ml, probs)
    entropy = _binary_entropy(probs).mean(axis=1)

    order = np.argsort(-probs, axis=1)  # descending prob, per row
    predicted_primary_idx = order[:, 0]
    predicted_primary_prob = probs[np.arange(n), predicted_primary_idx]
    predicted_secondary_idx = order[:, 1]
    predicted_secondary_prob = probs[np.arange(n), predicted_secondary_idx]

    return {
        "self_confidence": self_confidence,
        "entropy": entropy,
        "predicted_primary_idx": predicted_primary_idx,
        "predicted_primary_prob": predicted_primary_prob,
        "predicted_secondary_idx": predicted_secondary_idx,
        "predicted_secondary_prob": predicted_secondary_prob,
    }


def try_cleanlab_quality(y_ml: np.ndarray, probs: np.ndarray):
    """Return per-row cleanlab label-quality scores, or None if unavailable."""
    try:
        from cleanlab.multilabel_classification.filter import find_label_issues
        from cleanlab.multilabel_classification.rank import get_label_quality_scores
    except ImportError:
        print("[warn] cleanlab not installed — skipping confident-learning score "
              "(self_confidence/entropy buckets still work). "
              "Install with: pip install cleanlab")
        return None, None

    labels = [list(np.nonzero(row)[0]) for row in y_ml.astype(int)]
    issues = find_label_issues(labels=labels, pred_probs=probs)
    quality = get_label_quality_scores(labels=labels, pred_probs=probs)
    return issues, quality


def bucket_rows(scores: dict, cleanlab_issue, conf_low: float, entropy_quantile: float):
    n = len(scores["self_confidence"])
    bucket = np.full(n, "C", dtype=object)

    likely_bad = scores["self_confidence"] < conf_low
    if cleanlab_issue is not None:
        likely_bad = likely_bad | cleanlab_issue
    bucket[likely_bad] = "A"

    entropy_cut = np.quantile(scores["entropy"], entropy_quantile)
    ambiguous = (~likely_bad) & (scores["entropy"] >= entropy_cut)
    bucket[ambiguous] = "B"

    return bucket


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score + bucket rows for the re-annotation audit queue.")
    ap.add_argument("--path", default="Annotated_data.csv")
    ap.add_argument("--oof-dir", default="results/reannotation")
    ap.add_argument("--out", default="results/reannotation")
    ap.add_argument("--conf-low", type=float, default=0.30,
                     help="self_confidence below this -> bucket A (likely mislabeled)")
    ap.add_argument("--entropy-quantile", type=float, default=0.85,
                     help="rows above this entropy quantile (and not already in A) -> bucket B")
    ap.add_argument("--sec-threshold", type=float, default=0.35,
                     help="model's 2nd-place prob must clear this to be offered as a suggested secondary label")
    ap.add_argument("--queue-size", type=int, default=400,
                     help="how many rows (A+B, priority-sorted) go to the human audit queue")
    args = ap.parse_args(argv)

    df = load_raw(args.path)
    y_bin, y_mc, y_ml = make_targets(df)

    probs = np.load(Path(args.oof_dir) / "oof_probs.npy")
    assert probs.shape == y_ml.shape, "oof_probs.npy shape must match the corpus — re-run reannotate_oof_predict.py"

    # --- sanity check on the OOF model itself (Stage 1 quality), before scoring ---
    avg_pos_per_row = (probs > 0.5).sum(axis=1).mean()
    per_class_std = probs.std(axis=0)
    if avg_pos_per_row > 2.0 or per_class_std.mean() < 0.15:
        print("!" * 68)
        print("WARNING: oof_probs.npy looks undertrained / over-predicting.")
        print(f"  avg predicted positives/row @0.5 thr = {avg_pos_per_row:.2f} (true corpus avg ~0.8)")
        print(f"  mean per-class prob std              = {per_class_std.mean():.3f} (want > ~0.15)")
        print("  Consider re-running reannotate_oof_predict.py with a different --max-pos-weight")
        print("  or more --epochs, or checking the printed per-fold inner-val macro_f1 there.")
        print("!" * 68)

    scores = compute_scores(y_ml, probs)
    cleanlab_issue, cleanlab_quality = try_cleanlab_quality(y_ml, probs)
    bucket = bucket_rows(scores, cleanlab_issue, args.conf_low, args.entropy_quantile)

    def idx_to_name(idx):
        return [DISTORTIONS[i] for i in idx]

    out = pd.DataFrame({
        ID_COL: df[ID_COL],
        TEXT_COL: df[TEXT_COL],
        "original_dominant": df["Dominant Distortion"],
        "original_secondary": df["Secondary Distortion (Optional)"],
        "bucket": bucket,
        "self_confidence": scores["self_confidence"].round(3),
        "entropy": scores["entropy"].round(3),
        "model_predicted_primary": idx_to_name(scores["predicted_primary_idx"]),
        "model_predicted_primary_prob": scores["predicted_primary_prob"].round(3),
        "model_predicted_secondary": [
            DISTORTIONS[i] if p >= args.sec_threshold else ""
            for i, p in zip(scores["predicted_secondary_idx"], scores["predicted_secondary_prob"])
        ],
        "model_predicted_secondary_prob": scores["predicted_secondary_prob"].round(3),
    })
    if cleanlab_quality is not None:
        out["cleanlab_quality"] = np.round(cleanlab_quality, 3)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "triage_full.csv", index=False)

    print("Bucket counts:")
    print(out["bucket"].value_counts())

    # Priority order: A first (worst self_confidence first), then B (highest entropy first).
    priority_key = np.where(
        out["bucket"] == "A", out["self_confidence"],           # ascending within A
        np.where(out["bucket"] == "B", -out["entropy"], np.inf)  # ascending == descending entropy within B
    )
    audit = out[out["bucket"].isin(["A", "B"])].copy()
    audit["_priority"] = priority_key[out["bucket"].isin(["A", "B"]).to_numpy()]
    audit = audit.sort_values(["bucket", "_priority"]).drop(columns="_priority")
    audit = audit.head(args.queue_size).reset_index(drop=True)

    # blank columns a human fills in
    audit["corrected_primary"] = ""
    audit["corrected_secondary"] = ""
    audit["reviewer"] = ""
    audit["review_notes"] = ""

    audit.to_csv(out_dir / "audit_queue.csv", index=False)
    print(f"\nWrote triage_full.csv ({len(out)} rows) and "
          f"audit_queue.csv ({len(audit)} rows, buckets A+B, priority-sorted) to {out_dir}/")
    print("\nValid distortion labels for the corrected_primary/corrected_secondary "
          "columns (leave corrected_secondary blank if none):")
    print("  no_distortion, " + ", ".join(DISTORTIONS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
