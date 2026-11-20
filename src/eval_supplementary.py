"""S08: score one seed's binary + multilabel checkpoints on the S03 supplementary-evaluation rows.

The 376 `role == evaluation` rows of data/supplementary/supplementary_labelled.csv (scraped founder
writing, labelled by Lumia and Izza, never used as seeds). Gold labels: "No Distortion (calibrated
pessimism)" counts as No Distortion; the multilabel target is dominant + secondary.

Prediction is the evaluation path (src/app_parity.run -> src/evaluate.predict_logits ->
predictions_from_logits) with the checkpoint's own D13 binary threshold, its per-type multilabel
thresholds and the 2-label cap. Metrics: specificity, FPR and sensitivity (binary); multilabel
macro-F1 via src/metrics.metric_bundle (the same function src/evaluate.py uses, over all rows); and
per-class F1 with support via src/metrics.per_class_table.

    python -m src.eval_supplementary --binary CK/binary_mental-roberta-base_42 \
        --multilabel CK/multilabel_mental-roberta-base_42 --condition train_only --out OUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DISTORTIONS, LABEL_CANON

SUPP = Path("data/supplementary/supplementary_labelled.csv")
CP_LABEL = "No Distortion (calibrated pessimism)"


def _canon(label) -> str | None:
    label = "" if label is None or (isinstance(label, float) and np.isnan(label)) else str(label).strip()
    if not label:
        return None
    if label == CP_LABEL:
        label = "No Distortion"
    return LABEL_CANON[label]


def gold(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(y_bin, y_ml[n, 10]) from the dominant / secondary columns."""
    dom = [_canon(x) for x in df["dominant"]]
    sec = [_canon(x) for x in df["secondary"]]
    if any(d is None for d in dom):
        raise SystemExit("supplementary rows without a dominant label")
    y_bin = np.array([int(d != "no_distortion") for d in dom])
    y_ml = np.array([[int(t in (d, s)) for t in DISTORTIONS] for d, s in zip(dom, sec)])
    return y_bin, y_ml


def binary_rates(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum()); fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum()); fn = int(((y_true == 1) & (y_pred == 0)).sum())
    spec = tn / (tn + fp) if tn + fp else float("nan")
    sens = tp / (tp + fn) if tp + fn else float("nan")
    return {"specificity": spec, "fpr": 1 - spec, "sensitivity": sens, "tn": tn, "fp": fp, "tp": tp, "fn": fn}


def load_eval_rows(path: Path = SUPP) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    return df[df.role == "evaluation"].reset_index(drop=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S08: binary + multilabel checkpoints on the supplementary-evaluation rows")
    ap.add_argument("--binary", required=True)
    ap.add_argument("--multilabel", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-labels", type=int, default=2)
    args = ap.parse_args(argv)

    from src.app_parity import run
    from src.evaluate import predictions_from_logits
    from src.metrics import metric_bundle, per_class_table
    from src.train_transformer import resolve_device

    df = load_eval_rows()
    y_bin, y_ml = gold(df)
    texts = df["text"].astype(str)
    dev = resolve_device(args.device)

    bmeta, blog = run(Path(args.binary), texts, dev)
    b_pred = predictions_from_logits(blog, "binary", binary_threshold=bmeta["binary_threshold"])
    mmeta, mlog = run(Path(args.multilabel), texts, dev)
    m_pred = predictions_from_logits(mlog, "multilabel", mmeta["thresholds"], max_labels=args.max_labels)

    seed = int(bmeta["seed"])
    ml = metric_bundle("multilabel", y_ml, m_pred, bmeta["model"], seed, "supplementary_evaluation")
    pc = per_class_table("multilabel", y_ml, m_pred)
    out = {
        "condition": args.condition, "seed": seed, "n": len(df),
        "n_distorted": int(y_bin.sum()), "n_non_distorted": int((y_bin == 0).sum()),
        "binary_threshold": bmeta["binary_threshold"], "binary_splits": bmeta.get("splits"),
        "multilabel_splits": mmeta.get("splits"),
        "binary": binary_rates(y_bin, b_pred),
        "multilabel": {k: ml[k] for k in ("macro_f1", "micro_f1", "weighted_f1")},
        "per_class": pc[["class", "precision", "recall", "f1", "support"]].to_dict(orient="records"),
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / f"supp_eval_{args.condition}_{seed}.json"
    p.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("condition", "seed", "n", "binary", "multilabel")}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
