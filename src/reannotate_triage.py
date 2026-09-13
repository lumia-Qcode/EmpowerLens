"""
src/reannotate_triage.py — Stage 2 of model-assisted re-annotation.

Consumes the out-of-fold probability matrix from
`src/reannotate_oof_predict.py` and turns it into a prioritized human-review
queue. No row is ever auto-relabeled here — this script only decides ORDER
and SUGGESTS a starting point; a human makes every final label decision.

Per-row scores
--------------
self_confidence   : how much probability mass the model puts on the
                    ORIGINAL label set (min across originally-positive
                    classes' probs, and (1-p) across originally-negative
                    classes' probs — the weakest link in the original
                    annotation, not an average that can hide one bad flag).
entropy           : mean per-class binary entropy of the model's own
                    predictions — how unsure the model is, independent of
                    whether the original label agrees with it.
predicted_primary / predicted_secondary : model's own top-1 / top-2 guess
                    (secondary only kept if its prob clears --sec-threshold).
cleanlab_quality  : label quality score from cleanlab's confident-learning
                    method for multi-label data (Northcutt et al., 2021),
                    if cleanlab is installed; NaN otherwise (script still
                    runs — cleanlab is a refinement, not a hard dependency).

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


def compute_scores(y_ml: np.ndarray, probs: np.ndarray):
    n, k = y_ml.shape
    # self_confidence: weakest-link probability mass on the ORIGINAL label
    per_class_conf = np.where(y_ml == 1, probs, 1 - probs)  # (n, k)
    self_confidence = per_class_conf.min(axis=1)

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
