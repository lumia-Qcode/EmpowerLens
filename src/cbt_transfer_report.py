"""D21 — CBT-Bench (CBT-CD) transfer evaluation of the D12 winner (MentalRoBERTa multilabel).

Reads the per-seed CSVs written by ``notebooks/kaggle_runner_d21_cbt.ipynb`` and writes
``results_cbt_transfer/transfer_evaluation_final.md``: per-class precision / recall / F1 /
support and macro- / micro-F1 as mean +/- std over seeds 42, 1337, 2024, with Izza's
4 Oct numbers beside them. With no seed CSVs present it writes a scaffold (4 Oct column
filled, everything else "pending Kaggle run").

Expected layout (one pair per seed):

    results_cbt_transfer/d21/cbt_per_class_<seed>.csv
        input, decision, subset, label, precision, recall, f1, support
        (label is one of the 10 distortions, or "macro" / "micro")
    results_cbt_transfer/d21/cbt_predictions_<seed>.csv
        item-level predictions (not read here; kept for inspection)

Run:
    venv\\Scripts\\python.exe -m src.cbt_transfer_report
    venv\\Scripts\\python.exe -m src.cbt_transfer_report --dir results_cbt_transfer/d21 --out results_cbt_transfer/transfer_evaluation_final.md

The helpers ``strict_canon_map`` / ``per_class_rows`` are also imported by the notebook,
so the label mapping and the metric code are the same ones the tests cover.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src.data import DISTORTIONS

SEEDS = (42, 1337, 2024)
DEFAULT_DIR = Path("results_cbt_transfer/d21")
DEFAULT_OUT = Path("results_cbt_transfer/transfer_evaluation_final.md")

# Copied verbatim from results_cbt_transfer/kaggle-runner-transfer-eval.ipynb (cell 4,
# Izza's 4 Oct run). Exact-match only; an unmapped label raises (no fuzzy matching).
CBT_LABEL_CANON = {
    'all-or-nothing thinking': 'all_or_nothing',
    'magnification': 'magnification',
    'mind reading': 'mind_reading',
    'overgeneralization': 'overgeneralization',
    'personalization': 'personalization',
    'emotional reasoning': 'emotional_reasoning',
    'mental filter': 'mental_filter',
    'should statements': 'should_statements',
    'labeling': 'labeling',
    'fortune-telling': 'fortune_telling'
}

# Izza's 4 Oct multilabel numbers ("PHASE 2: MULTI-LABEL ARCHITECTURE RESULTS"),
# hard-coded from results_cbt_transfer/transfer_evaluation.md. That run: MentalRoBERTa
# multilabel, seed 42 only, trainer defaults (not the D12 recipe), fixed 0.5 threshold,
# input = ori_text, all 146 items. Values are rounded to 2 dp in the source.
OCT4_SOURCE = "results_cbt_transfer/transfer_evaluation.md (Phase 2, multi-label; 4 Oct, seed 42)"
OCT4 = {  # label: (precision, recall, f1, support)
    "emotional_reasoning": (0.27, 0.42, 0.33, 36),
    "overgeneralization": (0.33, 0.41, 0.36, 32),
    "mental_filter": (0.09, 0.19, 0.12, 21),
    "should_statements": (0.23, 0.39, 0.29, 28),
    "all_or_nothing": (0.73, 0.42, 0.53, 65),
    "mind_reading": (0.47, 0.49, 0.48, 47),
    "fortune_telling": (0.34, 0.41, 0.37, 44),
    "magnification": (0.19, 0.56, 0.29, 25),
    "personalization": (0.30, 0.43, 0.35, 42),
    "labeling": (0.17, 0.28, 0.21, 29),
    "micro": (0.30, 0.41, 0.34, 369),
    "macro": (0.31, 0.40, 0.33, 369),
}

# The headline configuration: same input and item set as 4 Oct, decision rule as in
# src/evaluate.py (per-type val thresholds, at most 2 labels per text).
PRIMARY = ("ori_text", "cap2", "all")
INPUT_DESC = {
    "ori_text": "the original free-written post (same input as 4 Oct)",
    "situation_thoughts": "CBT-Bench's `situation` and `thoughts` fields joined with a space",
}
DECISION_DESC = {
    "cap2": "per-type val thresholds, at most 2 labels (src/evaluate.py default `--max-labels 2`)",
    "nocap": "per-type val thresholds, no cap",
}
SUBSET_DESC = {
    "all": "all 146 items",
    "not_in_train": "items whose Id_Number is not in data/splits/train.csv",
}
PER_CLASS_COLS = ["input", "decision", "subset", "label", "precision", "recall", "f1", "support"]
PENDING = "pending Kaggle run"


# --------------------------------------------------------------------------- helpers

def strict_canon_map(label_array):
    """Map CBT-Bench label strings to our DISTORTIONS names (4 Oct logic, verbatim)."""
    mapped = []
    for label in label_array:
        key = str(label).strip()
        if key not in CBT_LABEL_CANON:
            raise ValueError(f"Unmapped external label {label!r}. No fuzzy matching allowed.")
        mapped.append(CBT_LABEL_CANON[key])
    return mapped


def to_matrix(label_lists, labels=DISTORTIONS):
    """List of label-name lists -> (n, 10) 0/1 matrix in DISTORTIONS order."""
    idx = {lab: j for j, lab in enumerate(labels)}
    m = np.zeros((len(label_lists), len(labels)), dtype=int)
    for i, labs in enumerate(label_lists):
        for lab in labs:
            m[i, idx[lab]] = 1
    return m


def per_class_rows(y_true, y_pred, labels=DISTORTIONS):
    """Per-class P/R/F1/support plus 'macro' and 'micro' rows (support = positive labels)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(range(len(labels))),
                                                 average=None, zero_division=0)
    rows = [{"label": lab, "precision": float(p[j]), "recall": float(r[j]), "f1": float(f[j]),
             "support": int(s[j])} for j, lab in enumerate(labels)]
    total = int(y_true.sum())
    for avg in ("macro", "micro"):
        pa, ra, fa, _ = precision_recall_fscore_support(y_true, y_pred, average=avg, zero_division=0)
        rows.append({"label": avg, "precision": float(pa), "recall": float(ra), "f1": float(fa),
                     "support": total})
    return rows


# --------------------------------------------------------------------------- report

def load_seed_tables(results_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(results_dir).glob("cbt_per_class_*.csv")):
        m = re.fullmatch(r"cbt_per_class_(\d+)\.csv", path.name)
        if not m:
            continue
        df = pd.read_csv(path, encoding="utf-8")
        missing = set(PER_CLASS_COLS) - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        df["seed"] = int(m.group(1))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=PER_CLASS_COLS + ["seed"])
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample std (ddof=1) over seeds per (input, decision, subset, label)."""
    g = df.groupby(["input", "decision", "subset", "label"], sort=False)
    out = g.agg(n_seeds=("seed", "nunique"),
                precision=("precision", "mean"), precision_std=("precision", "std"),
                recall=("recall", "mean"), recall_std=("recall", "std"),
                f1=("f1", "mean"), f1_std=("f1", "std"),
                support=("support", "mean"), support_min=("support", "min"),
                support_max=("support", "max")).reset_index()
    return out.fillna({"precision_std": 0.0, "recall_std": 0.0, "f1_std": 0.0})


def _ms(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def _support(row):
    lo, hi = int(row["support_min"]), int(row["support_max"])
    return str(lo) if lo == hi else f"{lo}–{hi}"


def _oct4(label):
    p, r, f, s = OCT4[label]
    return f"{p:.2f} / {r:.2f} / {f:.2f}"


def render(summary: pd.DataFrame | None, seeds_found: list[int]) -> str:
    have = summary is not None and not summary.empty
    L = []
    L.append("# D21 — CBT-Bench (CBT-CD) transfer: final model\n")
    L.append("Model: **MentalRoBERTa** (`mental/mental-roberta-base`, D12 winner, `docs/model_selection.md`), "
             "multilabel, D12 recipe (lr 3e-5, 5 epochs, weight decay 0.1, max length 512, batch 16, "
             "`data/splits`), per-type thresholds fitted on val during training (`docs/thresholds.md`). "
             "No fine-tuning on CBT-Bench.\n")
    L.append("Data: `Psychotherapy-LLM/CBT-Bench`, config `distortions_test` (CBT-CD), 146 items, 0 dropped; "
             "labels mapped with the 4 Oct notebook's `CBT_LABEL_CANON`, copied verbatim "
             "(`results_cbt_transfer/kaggle-runner-transfer-eval.ipynb`, cell 4). 369 gold labels "
             "(1–3 per item).\n")
    L.append("Generated by `venv\\Scripts\\python.exe -m src.cbt_transfer_report` from "
             "`results_cbt_transfer/d21/cbt_per_class_<seed>.csv` (written by "
             "`notebooks/kaggle_runner_d21_cbt.ipynb`).\n")
    seeds_txt = ", ".join(map(str, seeds_found)) if seeds_found else "none yet"
    L.append(f"Seeds found: **{seeds_txt}** (expected {', '.join(map(str, SEEDS))}). "
             "Std is the sample std (ddof=1) over seeds.\n")
    if not have:
        L.append(f"**Status: {PENDING}.** Only the 4 Oct column is filled.\n")

    L.append("## Read this first\n")
    L.append("- **Format difference:** CBT-CD gives each case as a *situation* plus an automatic *thought* "
             "(plus the original post, `ori_text`), not one free-written reflection like our training data. "
             "The headline table scores `ori_text`, as the 4 Oct run did; `situation + thoughts` is reported "
             "below as a second input.")
    L.append("- **Overlap with our data:** CBT-CD's 146 `ori_text` posts come from the same source as "
             "`Annotated_data.csv` (same Id_Numbers and text). Checked 20 Nov against the frozen splits: "
             "110 items are in our **train** split, 18 in val, 18 in test. Only the *labels* are independent "
             "(CBT-Bench's own annotators and rules). So the 146-item score measures agreement with a "
             "different labelling of mostly-seen texts, not transfer to unseen text; the `not_in_train` "
             "subset (36 items) is the cleaner, much noisier check. The same overlap applied to the 4 Oct run.")
    L.append("- **Ties:** with about 15–65 items per class (21–65 here), a difference smaller than the seed "
             "std counts as a tie, not a win or loss. The 4 Oct column is a single seed (42) with a fixed 0.5 "
             "threshold and default training settings, so it has no std and is not the same recipe.")
    L.append("- Expect a big drop from our own test scores; that is normal across datasets (Chapter 3 cites "
             "74.5% → 23.7% F1).\n")

    L.append("## Per-class results (headline: `ori_text`, all 146 items, thresholds + max 2 labels)\n")
    L.append("| Distortion | Support | Precision | Recall | F1 | 4 Oct P / R / F1 |")
    L.append("|---|---|---|---|---|---|")
    prim = None
    if have:
        prim = summary[(summary["input"] == PRIMARY[0]) & (summary["decision"] == PRIMARY[1])
                       & (summary["subset"] == PRIMARY[2])].set_index("label")
    for lab in list(DISTORTIONS) + ["macro", "micro"]:
        name = {"macro": "**Macro avg**", "micro": "**Micro avg**"}.get(lab, lab)
        if prim is not None and lab in prim.index:
            r = prim.loc[lab]
            L.append(f"| {name} | {_support(r)} | {_ms(r.precision, r.precision_std)} | "
                     f"{_ms(r.recall, r.recall_std)} | {_ms(r.f1, r.f1_std)} | {_oct4(lab)} |")
        else:
            L.append(f"| {name} | {OCT4[lab][3]} | {PENDING} | {PENDING} | {PENDING} | {_oct4(lab)} |")
    L.append(f"\n4 Oct source: {OCT4_SOURCE}. Supports for macro/micro rows are total gold labels.\n")

    L.append("## Macro-F1 and micro-F1, all variants (mean ± std over seeds)\n")
    L.append("| Input | Decision rule | Items | Macro-F1 | Micro-F1 | Seeds |")
    L.append("|---|---|---|---|---|---|")
    if have:
        keys = summary[["input", "decision", "subset"]].drop_duplicates()
        order = {k: i for i, k in enumerate(["ori_text", "situation_thoughts", "cap2", "nocap",
                                             "all", "not_in_train"])}
        keys = sorted(map(tuple, keys.to_numpy()), key=lambda k: tuple(order.get(x, 99) for x in k))
        for inp, dec, sub in keys:
            s = summary[(summary["input"] == inp) & (summary["decision"] == dec)
                        & (summary["subset"] == sub)].set_index("label")
            if "macro" not in s.index or "micro" not in s.index:
                continue
            ma, mi = s.loc["macro"], s.loc["micro"]
            L.append(f"| `{inp}` | {dec} | {sub} | {_ms(ma.f1, ma.f1_std)} | {_ms(mi.f1, mi.f1_std)} | "
                     f"{int(ma.n_seeds)} |")
    else:
        L.append(f"| `ori_text` | cap2 | all | {PENDING} | {PENDING} | 0 |")
    L.append(f"| `ori_text` (4 Oct) | fixed 0.5, no cap | all | {OCT4['macro'][2]:.2f} (1 seed) | "
             f"{OCT4['micro'][2]:.2f} (1 seed) | 1 |\n")
    L.append("Inputs: " + "; ".join(f"`{k}` = {v}" for k, v in INPUT_DESC.items()) + ".  ")
    L.append("Decision rules: " + "; ".join(f"{k} = {v}" for k, v in DECISION_DESC.items()) + ".  ")
    L.append("Items: " + "; ".join(f"{k} = {v}" for k, v in SUBSET_DESC.items()) + ".\n")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default=str(DEFAULT_DIR), help="folder with cbt_per_class_<seed>.csv")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    df = load_seed_tables(Path(args.dir))
    seeds = sorted(df["seed"].unique().tolist()) if not df.empty else []
    summary = summarize(df) if not df.empty else None
    if seeds and set(seeds) != set(SEEDS):
        print(f"[warn] seeds found {seeds}, expected {list(SEEDS)}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(summary, [int(s) for s in seeds]), encoding="utf-8")
    print(f"Wrote {out} ({'scaffold, ' + PENDING if summary is None else f'{len(seeds)} seed(s)'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
