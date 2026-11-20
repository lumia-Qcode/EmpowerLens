"""
D20 - build results/results_multitask/summary.md: multi-task vs flat vs cascade.

RUN (after copying the Kaggle eval files into results/results_multitask/):
    python -m src.multitask_report

Reads only committed result files (no model, no test.csv):
  multi-task  results/results_multitask/eval_multitask_<seed>.json         (src.train_multitask)
  flat        results/results_thresholds/specificity.csv                    (D13, binary at val FPR <= 0.15)
              results/results_mentalroberta_frozensplits/eval_mental-roberta-base_multilabel_<seed>.json
              results/results_RUN2/experiments.md  Exp 6 (flat multilabel, transcribed below)
  cascade     results/results_RUN2/results_cascade/eval_cascade_*.json      (Exp 7, end-to-end)
              results/results_RUN2/results_stage1/eval_mental-roberta-base_binary_<seed>.json
              results/results_cascade_inputs/summary_per_seed.csv           (D17, selected sentences only)
              results/results_span_selector/eval_test_<seed>.json           (D03 selector)

If no multi-task eval file exists the multi-task row reads "pending Kaggle run"; the
fallback paragraph for "not reached by the freeze" is always included at the end.
Std is the population std (ddof=0), as in src.aggregate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [42, 1337, 2024]
R = Path("results")
MT_DIR = R / "results_multitask"
PENDING = "pending Kaggle run"

# results/results_RUN2/experiments.md, "Stage 2: End-to-End Multilabel Classification",
# Flat Baseline (Exp 6) test macro-F1 per seed. Markdown only, so transcribed here.
RUN2_FLAT_EXP6 = {42: 0.2568, 1337: 0.2656, 2024: 0.2886}


def fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "n/a"
    s = f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"
    return s if len(vals) == len(SEEDS) else f"{s} (n={len(vals)})"


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def multitask_rows(mt_dir=MT_DIR):
    rows = []
    for s in SEEDS:
        p = Path(mt_dir) / f"eval_multitask_{s}.json"
        if not p.exists():
            continue
        d = _json(p)
        st = d["settings"]
        for part in ("val", "test"):
            r = d["splits"][part]
            rows.append({
                "seed": s, "split": part, "best_epoch": st.get("best_epoch"),
                "specificity": r["binary"]["specificity"], "fpr": r["binary"]["fpr"],
                "sensitivity": r["binary"]["sensitivity"], "binary_macro_f1": r["binary"]["macro_f1"],
                "ml_macro_f1": r["multilabel"]["macro_f1"], "ml_micro_f1": r["multilabel"]["micro_f1"],
                "ml_macro_f1_distorted": r["multilabel"]["macro_f1_distorted_rows"],
                "sel_precision": r["selector"]["precision"], "sel_recall": r["selector"]["recall"],
                "sel_f1": r["selector"]["f1"], "sel_row_hit": r["selector"]["row_hit_rate"],
                "n_rows": r["n_rows"], "n_sel_rows": r["selector"]["n_rows"],
                "n_sentences": r["selector"]["n_sentences"],
                "n_truncated_sentences": r["selector"].get("n_sentences_truncated"),
                "settings": st,
            })
    return pd.DataFrame(rows)


def flat_binary_spec():
    df = pd.read_csv(R / "results_thresholds/specificity.csv", encoding="utf-8")
    df = df[(df["model"] == "mental/mental-roberta-base") & (df["split"] == "test")]
    return [float(df.loc[df["seed"] == s, "specificity"].iloc[0]) for s in SEEDS if (df["seed"] == s).any()]


def flat_multilabel():
    out = []
    for s in SEEDS:
        p = R / f"results_mentalroberta_frozensplits/eval_mental-roberta-base_multilabel_{s}.json"
        if p.exists():
            out.append(float(_json(p)["splits"]["test"]["metrics"]["macro_f1"]))
    return out


def cascade_run2():
    ml, spec = [], []
    for s in SEEDS:
        for p in sorted((R / "results_RUN2/results_cascade").glob(f"eval_cascade_binary_*_{s}_multilabel_*_{s}_*.json")):
            ml.append(float(_json(p)["splits"]["test"]["multilabel_metrics"]["macro_f1"]))
        p = R / f"results_RUN2/results_stage1/eval_mental-roberta-base_binary_{s}.json"
        if p.exists():
            pc = _json(p)["splits"]["test"]["per_class"]
            spec.append(float(next(c["recall"] for c in pc if c["class"] == "non_distorted")))
    return spec, ml


def cascade_d17():
    df = pd.read_csv(R / "results_cascade_inputs/summary_per_seed.csv", encoding="utf-8")
    df = df[df["row"] == "crop_sel"]
    return [float(df.loc[df["seed"] == s, "test_macro_f1"].iloc[0]) for s in SEEDS if (df["seed"] == s).any()]


def selector_d03():
    out = []
    for s in SEEDS:
        p = R / f"results_span_selector/eval_test_{s}.json"
        if p.exists():
            out.append(float(_json(p)["f1"]))
    return out


FALLBACK = (
    "**If the run is not done by the freeze.** The multi-task model was not trained by the "
    "D3 freeze. What was built: `src/train_multitask.py` (one MentalRoBERTa encoder with a "
    "binary head, a 10-way multilabel head and a sentence-selection head that scores each "
    "sentence from the mean of its token states; the selection loss is masked to zero for "
    "No-Distortion rows and the 13 rows whose quote could not be located), its unit tests "
    "(`tests/test_multitask.py`, including B-06), the Kaggle runner "
    "(`notebooks/kaggle_runner_d20_multitask.ipynb`) and this report script. Chapter 9 "
    "therefore reports flat vs cascade from `results/results_RUN2/experiments.md`, and "
    "Chapter 10 lists the multi-task model as the first FYP-2 experiment."
)


def build(mt_dir=MT_DIR) -> str:
    mt = multitask_rows(mt_dir)
    mt_test = mt[mt["split"] == "test"] if len(mt) else mt
    have_mt = len(mt_test) > 0

    def mtcol(c):
        return fmt(mt_test[c].tolist()) if have_mt else PENDING

    spec_r2, ml_r2 = cascade_run2()
    L = ["# D20 - Multi-task model vs flat vs cascade", ""]
    status = (f"Multi-task results present for seeds {sorted(mt_test['seed'].tolist())}."
              if have_mt else "**Status: pending Kaggle run.** The multi-task row fills in when "
              "`eval_multitask_<seed>.json` files are copied into `results/results_multitask/` and "
              "`python -m src.multitask_report` is re-run.")
    L += [status, ""]

    if have_mt:
        st = mt_test.iloc[0]["settings"]
        L += ["**Multi-task setup.** " + (
            f"`{st['model']}`, lr {st['lr']}, {st['epochs']} epochs, weight decay {st['weight_decay']}, "
            f"max length {st['max_length']} (head truncation), batch {st['batch_size']}, loss weights "
            f"{st['loss_weights']}. One shared encoder, three heads; selection loss masked for "
            f"No-Distortion and the 13 unlocated rows. Epoch chosen on val ({st['epoch_criterion']}). "
            "Binary threshold: lowest with val FPR <= 0.15; multilabel thresholds per type on val; "
            "selector threshold max val sentence F1. Test scored once."), ""]
    else:
        L += ["**Multi-task setup (planned).** `mental/mental-roberta-base`, D12 recipe (lr 3e-5, 5 epochs, "
              "weight decay 0.1, max length 512, batch 16), seeds 42 / 1337 / 2024, loss weights 1/1/1. "
              "One shared encoder, three heads; selection loss masked for No-Distortion and the 13 unlocated "
              "rows. Epoch chosen on val by the mean of val binary macro-F1, multilabel macro-F1 and selector "
              "F1. Binary threshold: lowest with val FPR <= 0.15; multilabel thresholds per type on val; "
              "selector threshold max val sentence F1. Test scored once.", ""]

    L += ["## Test results (mean ± std over 3 seeds, ddof=0)", "",
          "| Model | Binary specificity | Multilabel macro-F1 (all rows) | Multilabel macro-F1 (distorted rows) | Selector sentence F1 | Source |",
          "| --- | --- | --- | --- | --- | --- |",
          f"| **Multi-task (D20)** | {mtcol('specificity')} | {mtcol('ml_macro_f1')} | {mtcol('ml_macro_f1_distorted')} | {mtcol('sel_f1')} | `results/results_multitask/eval_multitask_<seed>.json` |",
          f"| Flat MentalRoBERTa, separate binary + multilabel models (D12 recipe) | {fmt(flat_binary_spec())} | {fmt(flat_multilabel())} | n/a | n/a (no selector) | `results/results_thresholds/specificity.csv`; `results/results_mentalroberta_frozensplits/eval_mental-roberta-base_multilabel_<seed>.json` |",
          f"| Flat multilabel, RUN 2 Exp 6 | n/a | {fmt(list(RUN2_FLAT_EXP6.values()))} | n/a | n/a | `results/results_RUN2/experiments.md` (Flat Baseline) |",
          f"| Cascade end-to-end, RUN 2 Exp 7 (binary gate -> multilabel) | {fmt(spec_r2)} (a) | {fmt(ml_r2)} | n/a | n/a (no selector) | `results/results_RUN2/results_cascade/eval_cascade_*.json`; stage 1 `results/results_RUN2/results_stage1/` |",
          f"| Cascade D17: trained selector -> selected sentences -> multilabel | see (b) | n/a | {fmt(cascade_d17())} | {fmt(selector_d03())} | `results/results_cascade_inputs/summary_per_seed.csv` (crop_sel); `results/results_span_selector/eval_test_<seed>.json` |",
          "",
          "(a) RUN 2 stage-1 specificity is at the argmax decision (P >= 0.5), **not** a val FPR <= 0.15 "
          "threshold, so it is not comparable with the other specificity numbers.",
          "(b) The D17 cascade runs on distorted rows only and has no binary stage; deployed, its gate "
          "would be the flat binary model, i.e. the flat row's specificity.", ""]

    L += ["## Where the comparison is not like-for-like", "",
          "- **Binary specificity.** Multi-task and flat (D13) both use the lowest threshold with val FPR <= 0.15, "
          "same splits, same encoder and D12 recipe: comparable. RUN 2 stage 1 uses argmax, 12 epochs, lr 2e-5.",
          "- **Multilabel, all rows vs distorted rows.** Multi-task and both flat rows score all test rows "
          "(253, No-Distortion rows have no labels). The D17 cascade scores distorted rows with a located span only "
          "(159 test rows), so compare it with the multi-task *distorted rows* column, which is over all distorted "
          "test rows (161; the 2 unlocated rows are included there).",
          "- **Training budget.** Multi-task and the D12 flat rows: 5 epochs, lr 3e-5. RUN 2 Exp 6/7: 12 epochs, lr 2e-5. "
          "D17: up to 12 epochs with early stopping, deterministic.",
          "- **RUN 2 flat discrepancy.** `experiments.md` Exp 6 (0.2568 / 0.2656 / 0.2886) does not match the "
          "`results_RUN2/results_multilabel_flat` eval JSONs (0.2730 / 0.2651 / 0.2494); the table uses "
          "`experiments.md` as the brief asks.",
          "- **Selector.** The D03 selector sees one sentence with no context (max length 128); the multi-task "
          "selector scores sentences in context from the shared 512-token pass. Both are scored on the same T12 "
          "sentence labels with a val-chosen threshold; sentences truncated away in the multi-task pass get "
          "score 0 (counted in the eval files).",
          ""]

    if have_mt:
        L += ["## Multi-task per seed", "",
              "| Seed | Split | Best epoch | Spec | FPR | Sens | Bin macro-F1 | ML macro-F1 | ML micro-F1 | Sel P | Sel R | Sel F1 | Row hit | Rows | Sel rows / sentences | Truncated sents |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for r in mt.sort_values(["split", "seed"], ascending=[False, True]).itertuples():
            L.append(f"| {r.seed} | {r.split} | {r.best_epoch} | {r.specificity:.3f} | {r.fpr:.3f} | {r.sensitivity:.3f} | "
                     f"{r.binary_macro_f1:.3f} | {r.ml_macro_f1:.3f} | {r.ml_micro_f1:.3f} | {r.sel_precision:.3f} | "
                     f"{r.sel_recall:.3f} | {r.sel_f1:.3f} | {r.sel_row_hit:.3f} | {r.n_rows} | "
                     f"{r.n_sel_rows} / {r.n_sentences} | {r.n_truncated_sentences} |")
        L.append("")

    L += ["## Fallback", "", FALLBACK, ""]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--mt-dir", default=str(MT_DIR))
    ap.add_argument("--out", default=str(MT_DIR / "summary.md"))
    a = ap.parse_args(argv)
    text = build(Path(a.mt_dir))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
