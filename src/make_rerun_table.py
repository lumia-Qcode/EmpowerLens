"""
Generate docs/RERUN_EXPERIMENTS.md — the DistilBERT re-run suite in one document,
across all three tasks, every loss, and whichever data splits were run.

Why generated rather than hand-written: same reason as src/make_results_table.py.
The numbers already exist in results_tutorial_distilbert/*.json; a hand-maintained
table drifts the moment a run is repeated, and a drifted table is worse than none
because nobody knows which half is stale.

    python -m src.tutorial_distilbert --tasks binary,multiclass,multilabel \\
        --ablation --seeds 42,1337,2024
    python -m src.make_rerun_table

WHAT IT READS
-------------
``val_metrics_<task>_<model>_<seed><suffix>.json``  one per (task, loss, seed) —
    the richest source, carrying the config, both threshold scorings where they
    exist, and the per-seed detail the "best per seed" section needs.
``per_class_val_mean_<task><suffix>.csv``           per-class, mean over seeds.
``eval_*.json``                                     test-set numbers, once
    src.evaluate has been run. Absent until then, and the doc says so.
``results/all_experiments.csv``                     OPTIONAL. Prior runs already
    in this repo, so DistilBERT's numbers can be read next to roberta-base and
    mental-bert on the same (dataset, task) exam.

FOUR THINGS THIS DOC IS CAREFUL ABOUT
-------------------------------------
1. **Tasks are not comparable to each other.** binary is 2 classes, multiclass 11,
   multilabel 10 — three different exams. macro-F1 0.41 on binary is not "better"
   than 0.27 on multilabel. Every table is therefore grouped BY task, never sorted
   across them.
2. **Nor are datasets.** `data/splits` (2,024 train) and `data/splits_combined`
   (4,652) have different test sets. The dataset is part of every row's identity.
3. **val and test are never averaged.** Every row carries its split.
4. **Findings are not invented.** The generator emits only arithmetic it can do
   from the data (which arm won, by how much) and leaves interpretation as an
   explicit TODO. A generated document asserting a conclusion nobody checked is
   how a wrong claim ends up in a thesis.

Usage
-----
    python -m src.make_rerun_table
    python -m src.make_rerun_table --results results_tutorial_distilbert
    python -m src.make_rerun_table --out docs/RERUN_EXPERIMENTS.md --no-prior
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

TASK_ORDER = ["binary", "multiclass", "multilabel"]

TASK_BLURB = {
    "binary": ("2 classes — is any distortion present?",
               "`positive_class_f1` is the headline: catching the distorted rows "
               "is the job, and 36.9% of rows are undistorted so accuracy flatters."),
    "multiclass": ("11 classes — `no_distortion` plus the 10 distortions, one "
                   "label per row",
                   "`macro_f1_10` is the headline: plain macro-F1 includes the "
                   "easy `no_distortion` majority class and overstates the model."),
    "multilabel": ("10 independent labels — a row may carry two distortions, or "
                   "none",
                   "`macro_f1` is the headline; an all-zero row means No "
                   "Distortion, so there is no 11th column."),
}

# Human-readable name + one-line mechanism for each --loss value.
LOSS_LABEL = {
    "bce": ("BCE (unweighted)", "the tutorial's loss: every example counts the same"),
    "pos_bce": ("BCE + pos_weight", "each positive counts negatives/positives times more"),
    "focal": ("Focal + pos_weight", "pos_weight, plus down-weighting of easy examples"),
    "asl": ("Asymmetric (ASL)", "separate gammas for positives/negatives, replaces pos_weight"),
    "ce": ("CrossEntropy (unweighted)", "softmax baseline: every row counts the same"),
    "weighted_ce": ("CrossEntropy + class weights", "inverse-frequency weight per class"),
}
LOSS_ORDER = ["bce", "pos_bce", "focal", "asl", "ce", "weighted_ce"]

# Metric columns rendered per task: the shared set, then the task's own headline.
# Precision and recall sit next to macro-F1 deliberately. F1 is their harmonic
# mean, so it collapses two opposite failures into one number: a model that
# fires almost nothing (high P, ~0 R) and one that fires everything (~0 P, high
# R) can post the same F1 and need opposite fixes. The pair disambiguates.
SHARED = [("macro_precision", "macro-P"), ("macro_recall", "macro-R"),
          ("macro_f1", "macro-F1"), ("micro_f1", "micro-F1"),
          ("weighted_f1", "weighted-F1"), ("accuracy", "accuracy"),
          ("roc_auc", "ROC-AUC")]
TASK_EXTRA = {
    "binary": [("positive_class_f1", "**positive-F1**")],
    "multiclass": [("macro_f1_10", "**macro-F1(10)**"), ("no_distortion_f1", "no-dist F1")],
    "multilabel": [("mean_labels_predicted", "labels/row")],
}

# Named experiments per task. Each is a QUESTION mapped onto (loss, mode) arms,
# so one training run can serve several experiments without being repeated.
EXPERIMENTS = {
    "multilabel": [
        ("ML1", "Baseline: the tutorial as published",
         "What does the wellally.tech recipe score on real data, unmodified?",
         [("bce", "fixed")]),
        ("ML2", "Threshold alone",
         "Same weights, same probabilities — only the decision line moves. If "
         "this recovers the gap, the model learned fine and only reported badly.",
         [("bce", "fixed"), ("bce", "tuned")]),
        ("ML3", "Loss alone",
         "Same flat 0.5 threshold, different training. If only this moves "
         "macro-F1, the unweighted loss really did stop it learning rare classes.",
         [("bce", "fixed"), ("pos_bce", "fixed"), ("focal", "fixed"), ("asl", "fixed")]),
        ("ML4", "Both fixes together",
         "The best configuration available, for the headline comparison.",
         [("pos_bce", "tuned"), ("focal", "tuned"), ("asl", "tuned")]),
    ],
    "binary": [
        ("B1", "Baseline: unweighted cross-entropy",
         "Can the model tell distorted from undistorted at all?",
         [("ce", "argmax")]),
        ("B2", "Class weighting",
         "1,278 of 2,024 training rows are distorted — mild imbalance. Does "
         "inverse-frequency weighting help, or just trade precision for recall?",
         [("ce", "argmax"), ("weighted_ce", "argmax")]),
    ],
    "multiclass": [
        ("MC1", "Baseline: unweighted cross-entropy",
         "Can one softmax over 11 mutually exclusive classes pick the dominant "
         "distortion?",
         [("ce", "argmax")]),
        ("MC2", "Class weighting",
         "`no_distortion` is 36.9% of rows and `all_or_nothing` 5.0%. Does "
         "inverse-frequency weighting lift `macro_f1_10`?",
         [("ce", "argmax"), ("weighted_ce", "argmax")]),
    ],
}


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    return f"{v:.{nd}f}"


def _pm(mean, std, nd=3):
    if mean is None or pd.isna(mean):
        return "—"
    if std is None or pd.isna(std):
        return f"{mean:.{nd}f}"
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def _mode_label(mode):
    return {"fixed": "0.5 flat", "tuned": "swept on val",
            "argmax": "argmax"}.get(mode, mode)


def _cols(task):
    return SHARED + TASK_EXTRA.get(task, [])


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_runs(results_dir: Path) -> pd.DataFrame:
    """One row per (task, loss, seed, threshold_mode). Empty if nothing ran."""
    rows = []
    for f in sorted(results_dir.glob("val_metrics_*.json")):
        if f.name.startswith("val_metrics_per_seed"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        task = d.get("task", "multilabel")
        modes = ([("fixed", "metrics"), ("tuned", "metrics_tuned")]
                 if task == "multilabel" else [("argmax", "metrics")])
        for mode, key in modes:
            if not d.get(key):
                continue
            rows.append({
                "task": task, "loss": d.get("loss", "bce"), "seed": d["seed"],
                "threshold_mode": mode, "split": "val",
                "model": d["model"], "dataset": d.get("splits", "data/splits"),
                "epochs": d.get("epochs"), "lr": d.get("lr"),
                "batch_size": d.get("batch_size"), "max_length": d.get("max_length"),
                "smoke": d.get("smoke", False),
                "n_train": d.get("n_train"), "n_val": d.get("n_val"),
                "selection_metric": d.get("selection_metric", "micro_f1"),
                "tuned_thresholds": d.get("tuned_thresholds"),
                **d[key],
            })
    return pd.DataFrame(rows)


def load_test(results_dir: Path) -> pd.DataFrame:
    """Test-set rows written by src.evaluate, if it has been run."""
    rows = []
    for f in sorted(results_dir.glob("eval_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        meta = d.get("meta", {})
        for split_name, block in d.get("splits", {}).items():
            m = block.get("metrics", {})
            rows.append({
                "task": meta.get("task", "?"), "loss": meta.get("loss", "?"),
                "seed": meta.get("seed"), "threshold_mode": "as-saved",
                "split": split_name, "model": meta.get("model", "?"),
                "dataset": meta.get("splits", "data/splits"),
                **{k: (float(v) if isinstance(v, (int, float)) and v != "" else None)
                   for k, v in m.items()
                   if k in ("macro_f1", "micro_f1", "weighted_f1", "macro_f1_10",
                            "positive_class_f1", "no_distortion_f1")},
            })
    return pd.DataFrame(rows)


def load_prior(task_set, dataset_set) -> pd.DataFrame:
    """Prior runs already in this repo, for context. Never mixed into our rows."""
    p = Path("results/all_experiments.csv")
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p)
    d = d[d["task"].isin(task_set) & d["splits"].isin(dataset_set)]
    if "valid" in d:
        d = d[d["valid"].astype(str).str.lower().isin(["true", "1"])]
    if d.empty:
        return d
    g = d.groupby(["splits", "model", "task", "split"], dropna=False)
    out = g[["macro_f1", "micro_f1", "weighted_f1", "macro_f1_10",
             "positive_class_f1"]].mean()
    out["seeds"] = g["seed"].nunique()
    return out.reset_index()


def agg(df: pd.DataFrame, metrics) -> pd.DataFrame:
    """Mean ± std over seeds per (task, dataset, model, loss, mode, split)."""
    if df.empty:
        return df
    have = [m for m in metrics if m in df.columns]
    keys = ["task", "dataset", "model", "split", "loss", "threshold_mode"]
    g = df.groupby(keys, dropna=False)
    out = g[have].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["seeds"] = g["seed"].nunique()
    return out.reset_index()


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _sort(a: pd.DataFrame) -> pd.DataFrame:
    return a.assign(
        _t=[TASK_ORDER.index(t) if t in TASK_ORDER else 99 for t in a["task"]],
        _l=[LOSS_ORDER.index(l) if l in LOSS_ORDER else 99 for l in a["loss"]],
        _m=[{"fixed": 0, "argmax": 0, "tuned": 1}.get(m, 2) for m in a["threshold_mode"]],
    ).sort_values(["_t", "dataset", "model", "split", "_l", "_m"])


def task_table(a: pd.DataFrame, task: str, exp_of: dict) -> str:
    sub = _sort(a[a["task"] == task])
    if sub.empty:
        return "_Not run yet._\n"
    cols = _cols(task)
    lines = ["| exp | dataset | model | loss | threshold | split | seeds | "
             + " | ".join(h for _, h in cols) + " |",
             "|" + "---|" * (7 + len(cols))]
    for _, r in sub.iterrows():
        codes = ", ".join(exp_of.get((r["loss"], r["threshold_mode"]), [])) or "—"
        cells = [_pm(r.get(f"{m}_mean"), r.get(f"{m}_std"),
                     2 if m == "mean_labels_predicted" else 3) for m, _ in cols]
        lines.append(
            f"| {codes} | `{r['dataset']}` | `{r['model']}` | "
            f"{LOSS_LABEL.get(r['loss'], (r['loss'], ''))[0]} | "
            f"{_mode_label(r['threshold_mode'])} | **{r['split']}** | "
            f"{r['seeds']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def best_per_seed(runs: pd.DataFrame, task: str, headline: str) -> str:
    sub = runs[(runs["task"] == task) & (runs["split"] == "val")]
    if sub.empty or headline not in sub.columns:
        return "_Not run yet._"
    lines = [f"| seed | best config | {headline} | macro-F1 | accuracy |",
             "|---|---|---|---|---|"]
    for seed, grp in sub.groupby("seed"):
        b = grp.loc[grp[headline].idxmax()]
        lines.append(
            f"| {seed} | {LOSS_LABEL.get(b['loss'], (b['loss'], ''))[0]} @ "
            f"{_mode_label(b['threshold_mode'])} | **{_fmt(b[headline])}** | "
            f"{_fmt(b.get('macro_f1'))} | {_fmt(b.get('accuracy'))} |")
    # A result that only one seed produces is not a result.
    spread = sub.groupby(["loss", "threshold_mode"])[headline].std(ddof=0).max()
    if pd.notna(spread):
        lines += ["", f"Largest across-seed spread for any single config: "
                      f"**± {spread:.3f}** on `{headline}`. Treat any gap smaller "
                      f"than this as noise."]
    return "\n".join(lines)


def experiment_section(code, title, question, arms, a, task, headline) -> str:
    out = [f"#### {code} — {title}", "", f"**Question.** {question}", ""]
    # pandas 3.x refuses `Series & list`, so the arm mask is built as a Series
    # sharing a's index rather than a bare list of bools.
    in_arms = pd.Series([(l, m) in arms for l, m
                         in zip(a["loss"], a["threshold_mode"])], index=a.index)
    sub = a[(a["task"] == task) & (a["split"] == "val") & in_arms]
    if sub.empty:
        return "\n".join(out + ["_Not run yet._", ""])

    cols = _cols(task)
    out += ["| loss | mechanism | threshold | " + " | ".join(h for _, h in cols) + " |",
            "|" + "---|" * (3 + len(cols))]
    sub = _sort(sub)
    for _, r in sub.iterrows():
        name, mech = LOSS_LABEL.get(r["loss"], (r["loss"], ""))
        cells = [_pm(r.get(f"{m}_mean"), r.get(f"{m}_std"),
                     2 if m == "mean_labels_predicted" else 3) for m, _ in cols]
        out.append(f"| {name} | {mech} | {_mode_label(r['threshold_mode'])} | "
                   + " | ".join(cells) + " |")
    out.append("")

    key = f"{headline}_mean"
    if key in sub.columns and sub[key].notna().any():
        best = sub.loc[sub[key].idxmax()]
        base = sub[sub["loss"].isin(["bce", "ce"])
                   & sub["threshold_mode"].isin(["fixed", "argmax"])]
        delta = ""
        if len(base) and best.name != base.index[0]:
            d = best[key] - base.iloc[0][key]
            delta = f", **{d:+.3f}** vs the baseline arm"
        out += [f"**Measured.** Best arm: `{best['loss']}` @ "
                f"{_mode_label(best['threshold_mode'])} — {headline} "
                f"{_fmt(best[key])}{delta}.", ""]
    out += ["> **Finding:** _(write after reading the run — what does this mean, "
            "and does it change the recommendation?)_", ""]
    if any(m == "tuned" for _, m in arms):
        out += ["> ⚠️ Tuned thresholds are selected on val, so tuned **val** "
                "numbers are optimistic. Confirm on test before quoting.", ""]
    return "\n".join(out)


def per_class_section(results_dir: Path, task: str) -> str:
    files = list(results_dir.glob(f"per_class_val_mean_{task}*.csv"))
    if not files:
        return "_Not run yet._\n"

    def loss_of(path):
        stem = path.stem.replace(f"per_class_val_mean_{task}", "").lstrip("_")
        stem = stem.replace("smoke", "").strip("_")
        return stem or ("bce" if task == "multilabel" else "ce")

    files.sort(key=lambda p: (LOSS_ORDER.index(loss_of(p))
                              if loss_of(p) in LOSS_ORDER else 99))
    out = []
    for f in files:
        pc = pd.read_csv(f, index_col=0).round(3)
        out += [f"**loss = `{loss_of(f)}`** (mean over seeds)", "",
                "| class | precision | recall | F1 | val support |",
                "|---|---|---|---|---|"]
        for cls, r in pc.iterrows():
            flag = " ⚠️" if r["f1"] == 0 else ""
            out.append(f"| `{cls}` | {r['precision']:.3f} | {r['recall']:.3f} | "
                       f"**{r['f1']:.3f}**{flag} | {int(r['support'])} |")
        dead = list(pc.index[pc["f1"] == 0])
        if dead:
            out += ["", f"⚠️ {len(dead)}/{len(pc)} classes scored F1 = 0.000: "
                        f"{', '.join('`' + str(d) + '`' for d in dead)}."]
        out.append("")
    return "\n".join(out)


def prior_section(prior: pd.DataFrame, task: str) -> str:
    sub = prior[prior["task"] == task] if not prior.empty else prior
    if sub.empty:
        return ("_No prior runs for this task in `results/all_experiments.csv`._\n")
    lines = ["| dataset | model | split | seeds | macro-F1 | micro-F1 | "
             "weighted-F1 | macro-F1(10) | positive-F1 |", "|" + "---|" * 9]
    for _, r in sub.sort_values(["splits", "model", "split"]).iterrows():
        lines.append(f"| `{r['splits']}` | `{r['model']}` | **{r['split']}** | "
                     f"{r['seeds']} | {_fmt(r['macro_f1'])} | {_fmt(r['micro_f1'])} "
                     f"| {_fmt(r['weighted_f1'])} | {_fmt(r.get('macro_f1_10'))} | "
                     f"{_fmt(r.get('positive_class_f1'))} |")
    lines += ["", "> These are **prior** runs from other scripts, shown for "
                  "context only. They differ in backbone and in recipe, so a gap "
                  "here mixes several causes — the controlled comparison is the "
                  "loss ablation above, which holds everything else fixed."]
    return "\n".join(lines)


# --------------------------------------------------------------------------

def build(results_dir: Path, out_path: Path, include_prior: bool = True) -> str:
    runs = load_runs(results_dir)
    test = load_test(results_dir)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    md = ["# Re-run experiments — DistilBERT on EmpowerLens\n",
          f"**Generated** {stamp} by `python -m src.make_rerun_table` from "
          f"`{results_dir}/`. Do not edit by hand — re-run the generator.\n"]

    if runs.empty:
        md += ["_No runs found. Run the suite first:_\n", "```",
               "python -m src.tutorial_distilbert --tasks binary,multiclass,"
               "multilabel --ablation --seeds 42,1337,2024", "```\n"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(md), encoding="utf-8")
        return "\n".join(md)

    all_metrics = [m for m, _ in SHARED] + [m for v in TASK_EXTRA.values() for m, _ in v]
    a = agg(runs, all_metrics)
    if not test.empty:
        a = pd.concat([a, agg(test, all_metrics)], ignore_index=True)

    tasks = [t for t in TASK_ORDER if t in set(runs["task"])]
    headline = {"binary": "positive_class_f1", "multiclass": "macro_f1_10",
                "multilabel": "macro_f1"}
    cfg = runs.iloc[0]
    prior = (load_prior(set(tasks), set(runs["dataset"]))
             if include_prior else pd.DataFrame())

    if runs["smoke"].any():
        md += ["> 🚧 **THIS IS SMOKE DATA — every number below is meaningless.**\n"
               "> Produced by `--smoke` (64 rows, 1 epoch) purely to exercise the\n"
               "> template. Regenerate after the real run before showing anyone.\n"]

    md += ["## Setup\n", "| | |", "|---|---|",
           f"| **Model** | `{cfg['model']}` |",
           f"| **Datasets** | " + ", ".join(f"`{d}`" for d in sorted(set(runs["dataset"]))) + " |",
           f"| **Tasks** | " + ", ".join(f"`{t}`" for t in tasks) + " |",
           f"| **Recipe** | lr {cfg['lr']}, batch {cfg['batch_size']}, "
           f"{cfg['epochs']} epoch{'' if cfg['epochs'] == 1 else 's'}, "
           f"max_length {cfg['max_length']} |",
           f"| **Seeds** | {', '.join(str(s) for s in sorted(runs['seed'].unique()))} |",
           f"| **Selection** | best epoch on val, by each task's headline metric; "
           f"test read only by `src.evaluate` |",
           f"| **Source recipe** | [wellally.tech DistilBERT tutorial]"
           f"(https://www.wellally.tech/blog/python-cognitive-distortion-transformer-tutorial) |",
           "",
           "> **Read this before comparing any two numbers.** Rows are grouped by "
           "task on purpose. `binary` is 2 classes, `multiclass` 11, `multilabel` "
           "10 — three different exams, so a higher macro-F1 on one is not a "
           "better model. The same applies across datasets: `data/splits` and "
           "`data/splits_combined` have different test sets.\n"]

    for task in tasks:
        what, note = TASK_BLURB[task]
        exp_of = {}
        for code, _, _, arms in EXPERIMENTS.get(task, []):
            for arm in arms:
                exp_of.setdefault(arm, []).append(code)

        stop = "" if what.endswith(("?", ".", "!")) else "."
        md += [f"---\n", f"## Task: `{task}`\n", f"**{what}{stop}** {note}\n",
               "### All runs\n", task_table(a, task, exp_of), ""]
        if task == "multilabel":
            md += ["`labels/row` is how many of the 10 labels fire on average; the "
                   "true rate is **0.79**. Far below is under-firing, far above is "
                   "spraying labels.\n"]
        md += [f"### Best per seed — by `{headline[task]}`\n",
               best_per_seed(runs, task, headline[task]), "",
               "### Named experiments\n"]
        for code, title, question, arms in EXPERIMENTS.get(task, []):
            md.append(experiment_section(code, title, question, arms, a, task,
                                         headline[task]))
        md += ["#### Test-set confirmation\n",
               "**Question.** Does the val result hold on data never used for any "
               "decision?\n"]
        t_sub = a[(a["task"] == task) & (a["split"] == "test")]
        if t_sub.empty:
            cap = " --max-labels 0" if task == "multilabel" else ""
            md += ["_Not run yet._ After choosing a configuration on val:\n", "```",
                   f"python -m src.evaluate --checkpoint checkpoints/tutorial_"
                   f"{task}_{cfg['model'].split('/')[-1]}_42{cap} --out {results_dir}",
                   "```\n"]
        else:
            md += [task_table(a[(a["task"] == task) & (a["split"] != "val")],
                              task, exp_of), ""]
        md += ["> **Finding:** _(write after the test pass — how big is the "
               "val→test drop, and is the ranking of configurations the same?)_\n",
               "### Per-class breakdown\n", per_class_section(results_dir, task)]
        if include_prior:
            md += ["### Prior runs in this repo, same task\n",
                   prior_section(prior, task), ""]

    md += ["---\n", "## Reproduce\n", "```",
           "python -m src.tutorial_distilbert --tasks binary,multiclass,multilabel "
           "\\", "    --ablation --seeds 42,1337,2024",
           "python -m src.make_rerun_table", "```\n",
           "## Protocol notes\n",
           "- **Never compare across tasks or across datasets.** Different class "
           "counts and different test sets are different exams.",
           "- **val and test are different exams too.** Never average them, and "
           "never quote a test number that was used to pick a configuration.",
           "- **Tuned thresholds are swept on val** (multilabel only), so tuned "
           "val numbers are optimistic by construction. The test row is honest.",
           "- **Accuracy is never a headline.** On multilabel it is exact subset "
           "match (a model predicting nothing scores 0.375 on val); on binary and "
           "multiclass it is dominated by `no_distortion` at 36.9% of rows.",
           "- **Three seeds, mean ± std** (project convention). A gap smaller than "
           "the across-seed spread is not a result.",
           "- **The loss is the only thing varying** within a task's ablation — "
           "architecture, data and every hyperparameter are held fixed, so a "
           "difference is attributable to the loss.",
           ""]

    text = "\n".join(md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")   # explicit: Windows cp1252 breaks ±/⚠️
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate docs/RERUN_EXPERIMENTS.md from a results dir.")
    ap.add_argument("--results", default="results_tutorial_distilbert")
    ap.add_argument("--out", default="docs/RERUN_EXPERIMENTS.md")
    ap.add_argument("--no-prior", action="store_true",
                    help="omit the context tables of prior runs from "
                         "results/all_experiments.csv")
    args = ap.parse_args(argv)

    out = Path(args.out)
    build(Path(args.results), out, include_prior=not args.no_prior)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
