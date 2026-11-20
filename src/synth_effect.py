"""S08: aggregate the Kaggle run (train only vs train + synthetic) into tables.

Expects the notebook's output folder (default results/results_synth_effect/raw), laid out as
    <raw>/<condition>/eval/eval_mental-roberta-base_{binary,multilabel}_<seed>.json   (src/evaluate.py)
    <raw>/<condition>/supp/supp_eval_<condition>_<seed>.json                        (src/eval_supplementary.py)

Per condition x split (val, test, supplementary_evaluation) x seed: specificity, FPR, sensitivity
(binary, at each checkpoint's D13 threshold) and multilabel macro-F1. Writes per_seed.csv, summary.csv
(mean +- std over seeds), paired.csv (train+synthetic minus train-only, per seed) and
supp_per_class.csv (per-type F1 with support on the supplementary-evaluation rows).

    venv\\Scripts\\python.exe -m src.synth_effect
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("results/results_synth_effect/raw")
OUT = Path("results/results_synth_effect")
CONDITIONS = ("train_only", "train_synth")
METRICS = ("specificity", "fpr", "sensitivity", "multilabel_macro_f1")
SPLITS = ("val", "test", "supplementary_evaluation")


def _binary_rates(per_class: list[dict]) -> dict:
    pc = {c["class"]: c for c in per_class}
    spec = float(pc["non_distorted"]["recall"])
    return {"specificity": spec, "fpr": 1 - spec, "sensitivity": float(pc["distorted"]["recall"])}


def collect(raw: Path = RAW) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pcs = [], []
    for cond in CONDITIONS:
        for bpath in sorted((raw / cond / "eval").glob("eval_*_binary_*.json")):
            seed = int(bpath.stem.rsplit("_", 1)[1])
            mpath = bpath.with_name(bpath.name.replace("_binary_", "_multilabel_"))
            b = json.loads(bpath.read_text(encoding="utf-8"))["splits"]
            m = json.loads(mpath.read_text(encoding="utf-8"))["splits"]
            for split in ("val", "test"):
                rows.append({"condition": cond, "seed": seed, "split": split, **_binary_rates(b[split]["per_class"]),
                             "multilabel_macro_f1": float(m[split]["metrics"]["macro_f1"])})
            s = json.loads((raw / cond / "supp" / f"supp_eval_{cond}_{seed}.json").read_text(encoding="utf-8"))
            rows.append({"condition": cond, "seed": seed, "split": "supplementary_evaluation",
                         **{k: s["binary"][k] for k in ("specificity", "fpr", "sensitivity")},
                         "multilabel_macro_f1": float(s["multilabel"]["macro_f1"])})
            for c in s["per_class"]:
                pcs.append({"condition": cond, "seed": seed, "class": c["class"], "f1": c["f1"],
                            "support": int(c["support"])})
    return pd.DataFrame(rows), pd.DataFrame(pcs)


def summarise(per_seed: pd.DataFrame) -> pd.DataFrame:
    g = per_seed.groupby(["condition", "split"])[list(METRICS)]
    mean, std, n = g.mean(), g.std(ddof=1), g.size()
    out = []
    for (cond, split), r in mean.iterrows():
        row = {"condition": cond, "split": split, "n_seeds": int(n[(cond, split)])}
        for k in METRICS:
            row[f"{k}_mean"] = round(float(r[k]), 4)
            row[f"{k}_std"] = round(float(std.loc[(cond, split), k]), 4)
        out.append(row)
    return pd.DataFrame(out)


def paired(per_seed: pd.DataFrame) -> pd.DataFrame:
    a = per_seed[per_seed.condition == "train_only"].set_index(["split", "seed"])[list(METRICS)]
    b = per_seed[per_seed.condition == "train_synth"].set_index(["split", "seed"])[list(METRICS)]
    d = (b - a).dropna().reset_index()
    return d.sort_values(["split", "seed"])


def per_class(pcs: pd.DataFrame) -> pd.DataFrame:
    t = pcs.groupby(["class", "condition"]).agg(f1_mean=("f1", "mean"), f1_std=("f1", lambda x: x.std(ddof=1)),
                                                support=("support", "first")).reset_index()
    w = t.pivot(index=["class", "support"], columns="condition", values=["f1_mean", "f1_std"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    return w.reset_index().round(4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S08: aggregate train-only vs train+synthetic results")
    ap.add_argument("--raw", default=str(RAW))
    args = ap.parse_args(argv)
    per_seed, pcs = collect(Path(args.raw))
    if per_seed.empty:
        raise SystemExit(f"no results under {args.raw}")
    OUT.mkdir(parents=True, exist_ok=True)
    per_seed.round(4).to_csv(OUT / "per_seed.csv", index=False)
    s = summarise(per_seed)
    s.to_csv(OUT / "summary.csv", index=False)
    p = paired(per_seed)
    p.round(4).to_csv(OUT / "paired.csv", index=False)
    pc = per_class(pcs)
    pc.to_csv(OUT / "supp_per_class.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.to_string(index=False)); print("\npaired (synth - only):\n", p.round(3).to_string(index=False))
    print("\nsupplementary per class:\n", pc.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
