"""
Generate docs/RESULTS.md — EVERY experiment in one table: dataset, architecture,
task, score.

Why generated rather than hand-written: the numbers already exist in
results/all_experiments.csv, and a hand-maintained table drifts. Re-run after any
new result lands and the doc cannot disagree with the data.

    python -m src.compile_results && python -m src.make_results_table

AUTO-DISCOVERY
--------------
Every (results_dir, task) pair in the CSV becomes a row. Nothing is hardcoded, so a
new run cannot silently go missing — an unrecognised dir shows up with "?" metadata
rather than being dropped. (An earlier version keyed a dict by results_dir alone,
which meant results_cascade's binary and multilabel rows collided and one vanished.)

The frozen classical results (binary_f1_results.csv, multiclass_f1_results.csv from
cd_pipeline.py) are folded in too, since "every experiment" includes Month 1.

THREE THINGS A NAIVE GROUPBY GETS WRONG
---------------------------------------
1. **Filter on `task`.** A cascade eval JSON holds a binary block AND a multilabel
   block, both carrying a column called `macro_f1` meaning different things
   (2 classes vs 10). Averaging across them once produced a spurious 0.479 +/- 0.261.
   n != 3 is the tell.
2. **The TEST SET is part of a run's identity.** CODIPAS runs are scored on CODIPAS's
   test set, the stage-A holdout on PatternReframe's. Different exams, not better
   models.
3. **Distorted-only is its own league.** Stage 2 and the sequential stages never see
   No-Distortion rows, so their macro_f1 is not comparable to a flat or cascade
   number over the same ten classes.

Usage
-----
    python -m src.make_results_table [--out docs/RESULTS.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# (results_dir, task) -> (trained on, TESTED ON, comparison group)
# Group letters define what may be ranked together. "X" = scored on a different test
# set; "L" = leaked, excluded from ranking entirely.
META = {
    ("results_stage1", "binary"):            ("Annotated", "Annotated", "A"),
    ("results_cascade", "binary"):           ("Annotated", "Annotated", "A"),
    ("results", "binary"):                   ("Annotated", "Annotated", "A"),
    ("results_multiclass_v2", "multiclass"): ("Annotated", "Annotated", "B"),
    ("results", "multiclass"):               ("Annotated", "Annotated", "B"),
    ("results_cascade", "multilabel"):       ("Annotated", "Annotated", "C"),
    ("results_multilabel_flat", "multilabel"): ("Annotated", "Annotated", "C"),
    ("results", "multilabel"):               ("Annotated", "Annotated", "C"),
    ("results_stage2", "multilabel"):        ("Annotated distorted-only", "Annotated distorted-only", "D"),
    ("results_seq_stageA", "multilabel"):    ("PatternReframe only", "Annotated distorted-only", "D"),
    ("results_seq_stageB", "multilabel"):    ("PatternReframe -> Annotated", "Annotated distorted-only", "D"),
    ("results_seq_stageA_v2", "multilabel"): ("PatternReframe only (V2: mask+merge)", "Annotated distorted-only", "D"),
    ("results_seq_stageB_v2", "multilabel"): ("PatternReframe -> Annotated (V2)", "Annotated distorted-only", "D"),
    # CODIPAS transfer: trained on CODIPAS, scored on the FROZEN Annotated test set.
    ("results_stage1_codipas_transfer", "binary"):            ("CODIPAS (deduped)", "Annotated", "A"),
    ("results_cascade_codipas_transfer", "binary"):           ("CODIPAS (deduped)", "Annotated", "A"),
    ("results_cascade_codipas_transfer", "multilabel"):       ("CODIPAS (deduped)", "Annotated", "C"),
    ("results_multilabel_flat_codipas_transfer", "multilabel"): ("CODIPAS (deduped)", "Annotated", "C"),
    ("results_multiclass_v2_codipas_transfer", "multiclass"):  ("CODIPAS (deduped)", "Annotated", "B"),
    ("results_stage1_codipas_transfer_matched", "binary"):     ("CODIPAS (size-matched)", "Annotated", "A"),
    ("results_cascade_codipas_transfer_matched", "multilabel"): ("CODIPAS (size-matched)", "Annotated", "C"),
    ("results_multilabel_flat_codipas_transfer_matched", "multilabel"): ("CODIPAS (size-matched)", "Annotated", "C"),
    ("results_multiclass_v2_codipas_transfer_matched", "multiclass"): ("CODIPAS (size-matched)", "Annotated", "B"),
    # Scored on someone else's test set.
    ("results_codipas", "binary"):                    ("CODIPAS", "CODIPAS", "X"),
    ("results_codipas", "multiclass"):                ("CODIPAS", "CODIPAS", "X"),
    ("results_codipas", "multilabel"):                ("CODIPAS", "CODIPAS", "X"),
    ("results_stage1_codipas_cls", "binary"):         ("CODIPAS", "CODIPAS", "X"),
    ("results_cascade_codipas_cls", "binary"):        ("CODIPAS", "CODIPAS", "X"),
    ("results_cascade_codipas_cls", "multilabel"):    ("CODIPAS", "CODIPAS", "X"),
    ("results_stage2_codipas_cls", "multilabel"):     ("CODIPAS distorted-only", "CODIPAS distorted-only", "X"),
    ("results_multilabel_flat_codipas_cls", "multilabel"): ("CODIPAS", "CODIPAS", "X"),
    ("results_multiclass_v2_codipas_cls", "multiclass"):   ("CODIPAS", "CODIPAS", "X"),
    ("results_stage2_codipas_transfer", "multilabel"):     ("CODIPAS (deduped)", "CODIPAS distorted-only", "X"),
    ("results_stage2_codipas_transfer_matched", "multilabel"): ("CODIPAS (size-matched)", "CODIPAS distorted-only", "X"),
    ("results_seq_stageA_holdout", "multilabel"):     ("PatternReframe only", "PatternReframe held-out", "X"),
    ("results_seq_stageA_v2_holdout", "multilabel"):  ("PatternReframe only (V2)", "PatternReframe held-out", "X"),
    # Leaked.
    ("results_combined", "binary"):     ("Annotated + CODIPAS (LEAKED)", "leaked", "L"),
    ("results_combined", "multiclass"): ("Annotated + CODIPAS (LEAKED)", "leaked", "L"),
    ("results_combined", "multilabel"): ("Annotated + CODIPAS (LEAKED)", "leaked", "L"),
}

GROUPS = {
    "A": ("Binary — is a distortion present?", "positive_class_f1",
          "Scored on the Annotated test set. Directly comparable to Shreevastava & Foltz's reported **0.79**."),
    "B": ("Multiclass — 11-way (10 distortions + no_distortion)", "macro_f1_10",
          "Scored on the Annotated test set. `macro_f1_10` drops `no_distortion`, so the easy class cannot flatter the score."),
    "C": ("Multilabel — full task, negatives included", "macro_f1",
          "Scored on the Annotated test set. This is where cascade-vs-flat is settled."),
    "D": ("Multilabel — distorted-only (the Stage 2 league)", "macro_f1",
          "Scored on the Annotated **distorted-only** test set. No No-Distortion rows on either side, so these are NOT comparable to group C."),
}
HEADLINE = {"binary": "positive_class_f1", "multiclass": "macro_f1_10", "multilabel": "macro_f1"}


def _classical(path, task):
    """The frozen Month-1 classical results (cd_pipeline.py). Best cell per file."""
    p = Path(path)
    if not p.exists():
        return None
    d = pd.read_csv(p).rename(columns={"Unnamed: 0": "model"}).set_index("model")
    best = d.stack().idxmax()
    return d.stack().max(), f"{best[0]} + {best[1]}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/all_experiments.csv")
    ap.add_argument("--out", default="docs/RESULTS.md")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    df = df[df["split"] == "test"]

    rows = []
    for (dirn, task), g in df.groupby(["results_dir", "task"]):
        metric = HEADLINE[task]
        v = g[metric].dropna()
        if not len(v):
            continue
        tr, te, grp = META.get((dirn, task), ("?", "?", "X"))
        models = sorted({str(m).replace("[stage2-isolated]", "").split("[")[0]
                         for m in g["model"]})
        arch = ", ".join(m.split("/")[-1] for m in models)[:46] or "?"
        rows.append({"group": grp, "task": task, "metric": metric,
                     "mean": v.mean(), "std": v.std(), "n": len(v),
                     "dir": dirn, "trained": tr, "tested": te, "arch": arch})

    L = ["# Results — every experiment in one table\n",
         "Generated by `src/make_results_table.py` from `results/all_experiments.csv`. "
         "**Do not edit by hand** — re-run it instead:\n",
         "```bash\npython -m src.compile_results && python -m src.make_results_table\n```\n",
         "**Rows may only be compared within a section.** Each section fixes the task "
         "*and* the test set, which is what makes its rows rankable. A model scored on "
         "CODIPAS's or PatternReframe's own test set is sitting a different exam; those "
         "are quarantined near the bottom.\n",
         "`n` is how many runs were averaged — normally the 3 seeds (42, 1337, 2024). "
         "**If `n` is not 3 the grouping is wrong**, unless several backbones were "
         "pooled deliberately (the `results/` directory does that).\n"]

    for gid in ("A", "B", "C", "D"):
        title, metric, note = GROUPS[gid]
        sub = sorted([r for r in rows if r["group"] == gid], key=lambda r: -r["mean"])
        if not sub:
            continue
        L += [f"\n## {title}\n", f"{note}\n",
              f"| {metric} | trained on | architecture | results dir | n |",
              "|---|---|---|---|---|"]
        top = sub[0]["mean"]
        for r in sub:
            sd = f" ± {r['std']:.3f}" if r["std"] == r["std"] else ""
            star = " **←best**" if r["mean"] == top else ""
            L.append(f"| **{r['mean']:.3f}**{sd}{star} | {r['trained']} | {r['arch']} | "
                     f"`{r['dir']}` | {r['n']} |")

    # Month 1 classical, from the frozen CSVs.
    cb, cm = _classical("binary_f1_results.csv", "binary"), _classical("multiclass_f1_results.csv", "multiclass")
    if cb or cm:
        L += ["\n## Month 1 — classical baselines (frozen `cd_pipeline.py`)\n",
              "Best cell from each frozen results file. **These are not leakage-free** — "
              "`cd_pipeline.py` fits Word2Vec/Doc2Vec on the full corpus before splitting, "
              "which is the specific bug `src/` exists to avoid. Kept for the Month-1 "
              "record, not for comparison.\n",
              "| F1 | task | best feature set | source |", "|---|---|---|---|"]
        if cb:
            L.append(f"| {cb[0]:.2f} | binary | {cb[1]} | `binary_f1_results.csv` |")
        if cm:
            L.append(f"| {cm[0]:.2f} | multiclass | {cm[1]} | `multiclass_f1_results.csv` |")

    sub = sorted([r for r in rows if r["group"] == "X"], key=lambda r: (r["tested"], -r["mean"]))
    if sub:
        L += ["\n## Scored on a DIFFERENT test set — not comparable to anything above\n",
              "Real numbers, different exam. They answer *\"how well does this model do on "
              "the dataset it was trained on?\"*, never *\"how well does it do on our "
              "task?\"* The two highest numbers in this whole document live here, which is "
              "exactly why the section exists.\n",
              "| score | metric | trained on | **tested on** | results dir | n |",
              "|---|---|---|---|---|---|"]
        for r in sub:
            sd = f" ± {r['std']:.3f}" if r["std"] == r["std"] else ""
            L.append(f"| {r['mean']:.3f}{sd} | {r['metric']} | {r['trained']} | "
                     f"**{r['tested']}** | `{r['dir']}` | {r['n']} |")

    sub = [r for r in rows if r["group"] == "L"]
    if sub:
        L += ["\n## Excluded — leaked, do not cite\n",
              "`data/splits_combined` put 194/253 val and 189/253 test rows into train, "
              "because CODIPAS overlaps `Annotated_data.csv` on 1,937 of its 2,621 rows. "
              "Inflated `macro_f1_10` by ~50%. Flagged `valid=False` in the CSV.\n",
              "| score | metric | results dir | n |", "|---|---|---|---|"]
        for r in sorted(sub, key=lambda r: -r["mean"]):
            L.append(f"| ~~{r['mean']:.3f}~~ | {r['metric']} | `{r['dir']}` | {r['n']} |")

    L += ["\n---\n",
          "Experiment design and narrative: `docs/EXPERIMENTS.md`. "
          "Label-agreement analysis: `docs/codipas_agreement.md`. "
          "Open work: `docs/TODO.md`.\n"]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {args.out} — {len(rows)} (dir, task) runs")
    unknown = sorted({r["dir"] for r in rows if (r["dir"], r["task"]) not in META})
    if unknown:
        print(f"[note] no metadata for: {unknown} — add them to META")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
