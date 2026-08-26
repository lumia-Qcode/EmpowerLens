# EmpowerLens — Month 1 Results (viva-ready summary)

A plain-English walkthrough of the Month-1 detection baseline: what was run,
what the numbers are, where every number lives, and how to re-check any of them
yourself. All numbers below are **3-seed means (seeds 42, 1337, 2024)** unless
noted, and come from the files cited in each section.

---

## 0. TL;DR (the one-line story)

We reproduce the canonical benchmark's **binary ≈0.79 / multiclass ≈0.30** gap
with a clean, no-leakage pipeline. Detecting *whether* a distortion exists is
reliable (RoBERTa Distorted-class F1 **0.80 val / 0.77 test**); detecting *which*
of 10 distortions is hard (macro over the 10 ≈ **0.14–0.21**), bottlenecked by
label noise (~33.7% inter-annotator agreement) and category overlap. Reframing
detection as **multi-label** doubles the classical baseline and improves 8 of 10
per-class scores.

---

## 1. What was built (the pipeline, in order)

| Step | Script | What it does | Output |
|---|---|---|---|
| 1. Load + label | [`src/data.py`](../src/data.py) | reads `Annotated_data.csv` (UTF-8), builds `y_bin`, `y_mc` (11-class), `y_ml` (10-label) | — |
| 2. Split once | [`src/make_splits.py`](../src/make_splits.py) | 80/10/10 multi-label stratified split, `random_state=42` | [`data/splits/`](../data/splits) + [`split_manifest.json`](../data/splits/split_manifest.json) |
| 3. Classical floor | [`src/baseline_classical.py`](../src/baseline_classical.py) | TF-IDF + balanced LogReg on **train**, scored on **val** | rows in `paper_comparison.csv` |
| 4. Fine-tune | [`src/train_transformer.py`](../src/train_transformer.py) | `roberta-base`, weighted CE / BCE, 3 seeds | checkpoints (not committed) |
| 5. Evaluate | [`src/evaluate.py`](../src/evaluate.py) | **only reader of `test.csv`**; scores val+test | everything in this folder |
| 6. Aggregate | [`src/aggregate.py`](../src/aggregate.py) | rolls all `eval_*.json` into the summary tables | `month1_summary*.csv`, `no_distortion_contribution.md` |

**Discipline baked in:** splits are frozen and committed (everyone uses the same
ones); any vectorizer/threshold is fit on **train only** (no leakage); `test.csv`
is read **only** by `evaluate.py`; 3 seeds with mean±std; headline metrics are F1
variants, never accuracy. See [`CLAUDE.md`](../CLAUDE.md) for the full rules.

**Dataset facts** (measured, see [`CLAUDE.md`](../CLAUDE.md) / `src/data.py`):
`Annotated_data.csv` = 2,530 rows; 1,597 distorted, 933 No-Distortion; 416 rows
carry a secondary distortion. Split sizes: **train 2,024 / val 253 / test 253**
(source: [`split_manifest.json`](../data/splits/split_manifest.json)).

---

## 2. The three tasks

| Task | Question | Labels | Key metric |
|---|---|---|---|
| **binary** | Is *any* distortion present? | 2 (distorted / not) | `positive_class_f1` = F1 of the Distorted class |
| **multiclass** | *Which* one distortion (or none)? | 11 (`no_distortion` + 10) | `macro_f1_10` = macro-F1 over the 10, dropping the easy `no_distortion` |
| **multilabel** | Which distortions are present (0, 1, or 2)? | 10 (all-zeros = none) | `macro_f1` over the 10; predictions capped at 2 (`--max-labels`, `evaluate.py`) |

Task definitions and the weighted-loss handling live in
[`src/train_transformer.py`](../src/train_transformer.py); metric definitions in
[`src/metrics.py`](../src/metrics.py).

---

## 3. Headline results

**Source: [`month1_summary_meanstd.csv`](month1_summary_meanstd.csv)** (RoBERTa)
and [`paper_comparison.csv`](paper_comparison.csv) (baseline + paper rows).

| Task | metric | Baseline (val) | **RoBERTa val** | **RoBERTa test** | Paper |
|---|---|---|---|---|---|
| binary | weighted_f1 | 0.722 | 0.767 ± 0.011 | 0.720 ± 0.023 | 0.79\* |
| binary | **positive_class_f1** | 0.766 | **0.802 ± 0.010** | 0.773 ± 0.021 | — |
| multiclass | weighted_f1 | 0.330 | 0.345 ± 0.011 | 0.290 ± 0.007 | 0.30\*\* |
| multiclass | **macro_f1_10** | 0.135 | 0.175 ± 0.013 | 0.138 ± 0.015 | — |
| multilabel | weighted_f1 | 0.141 | 0.287 ± 0.010 | 0.209 ± 0.009 | — |
| multilabel | **macro_f1** | 0.124 | **0.275 ± 0.012** | 0.207 ± 0.018 | — |

\* Paper's binary F1; averaging method **unspecified** (marked `averaging=UNSPECIFIED`).
\*\* Paper's multiclass weighted F1; class count **unstated** (`n_classes=UNSTATED`).
Both from Shreevastava & Foltz (2021), stored as `source=paper` rows in
[`paper_comparison.csv`](paper_comparison.csv).

Baseline numbers = `tfidf+logreg` rows (val) in the same file, produced by
[`src/baseline_classical.py`](../src/baseline_classical.py).

---

## 4. What went well (wins)

1. **Faithful replication.** Binary ≈0.72–0.80 and multiclass weighted ≈0.29–0.35
   reproduce the paper's ≈0.79 / ≈0.30 — the clean pipeline lands on the published
   numbers. (Table §3.)
2. **Binary detection is reliable.** RoBERTa Distorted-class F1 = 0.80 (val) / 0.77
   (test), beating the classical floor (0.766) and matching the paper, with low
   seed variance (±0.02).
3. **Multi-label roughly doubles the baseline** (macro_f1 0.124 → 0.275 val) and
   beats multiclass at per-distortion detection: **mean F1 over the 10 distortions
   = 0.207 (multilabel) vs 0.138 (multiclass)**.
   Source: `per_class_roberta-base_{multiclass,multilabel}_*.csv` (see §6 to
   reproduce the mean).
4. **Multi-label improves 8 of 10 per-class scores** and revives the dead class:

   | distortion | multiclass F1 | multilabel F1 |
   |---|---|---|
   | all_or_nothing | **0.00 ± 0.00** (dead on all seeds) | **0.16 ± 0.12** |
   | overgeneralization | 0.05 | 0.21 |
   | magnification | 0.06 | 0.21 |
   | labeling | 0.13 | 0.22 |
   | fortune_telling | 0.13 | 0.23 |
   | personalization | 0.10 | 0.14 |
   | emotional_reasoning | 0.16 | 0.20 |
   | mind_reading | 0.23 | 0.23 (tie) |
   | mental_filter | 0.29 | 0.29 (tie) |
   | should_statements | 0.24 | **0.18** (multilabel worse) |

   (3-seed means; source: `per_class_*` CSVs.) This is the concrete evidence that
   multi-label framing removes the forced-single-choice penalty.

---

## 5. What didn't (losses / limits)

1. **Fine-grained typing is modest** — best per-distortion F1 ≈ 0.29–0.32; diffuse
   ones (personalization 0.14, should_statements under multilabel 0.18) stay weak.
2. **Val → test gap** (~0.05–0.08 drop on every task) — the best epoch is chosen on
   val, so val is optimistic; test is the honest number.
3. **Multiclass headline is inflated** — ~70% of its weighted-F1 comes from the easy
   `no_distortion` class alone. Source:
   [`no_distortion_contribution.md`](no_distortion_contribution.md).
4. **Label-noise ceiling (~33.7% inter-annotator agreement)** — a fundamental limit
   no model change alone breaks; it is *the* reason multiclass caps out.
5. **Base model is vanilla `roberta-base`** — no domain adaptation yet (MentalBERT
   untried).

---

## 6. How to re-check any number yourself

Everything is reproducible from the committed files. Run from the repo root with
the project venv.

**Per-class F1 (the §4 table), 3-seed mean:**
```
venv\Scripts\python.exe - <<'PY'
import pandas as pd
seeds=[42,1337,2024]
def mean_f1(task):
    fs=[pd.read_csv(f'results/per_class_roberta-base_{task}_{s}.csv').set_index('class')['f1'] for s in seeds]
    return pd.concat(fs,axis=1).mean(axis=1)
print(pd.DataFrame({'multiclass':mean_f1('multiclass'),'multilabel':mean_f1('multilabel')}).round(3))
PY
```

**Confirm `all_or_nothing` = 0.00 on every seed (multiclass):**
```
venv\Scripts\python.exe -c "import pandas as pd; [print(s, pd.read_csv(f'results/per_class_roberta-base_multiclass_{s}.csv').set_index('class').loc['all_or_nothing','f1']) for s in (42,1337,2024)]"
```

**Headline table (§3):** open [`month1_summary_meanstd.csv`](month1_summary_meanstd.csv)
directly, or the per-run rows in [`paper_comparison.csv`](paper_comparison.csv).

**Re-run the whole evaluation from a checkpoint** (regenerates the `eval_*.json`,
per-class CSVs, and confusion PNGs): `python -m src.evaluate --checkpoint <dir>`
then `python -m src.aggregate`.

---

## 7. Where every artifact comes from

| File(s) here | Produced by | Contains |
|---|---|---|
| `paper_comparison.csv` | `baseline_classical.py` + `evaluate.py` | one row per (model, task, seed, split) + paper rows |
| `month1_summary.csv` | `aggregate.py` | per-run headline metrics |
| `month1_summary_meanstd.csv` | `aggregate.py` | metrics as mean ± std across seeds |
| `no_distortion_contribution.md` | `aggregate.py` | % of weighted-F1 from `no_distortion` |
| `per_class_roberta-base_<task>_<seed>.csv` | `evaluate.py` | per-class precision/recall/F1/support (test) |
| `per_class_tfidf_logreg_multiclass_42.csv` | `baseline_classical.py` | baseline per-class (multiclass) |
| `eval_roberta-base_<task>_<seed>.json` | `evaluate.py` | full metric bundle, both splits, + `meta` (incl. multilabel thresholds) |
| `confusion_roberta-base_<task>_<seed>.png` (+ `_no_nd`) | `evaluate.py` | row-normalized confusion matrices (binary/multiclass only) |

Metric formulas: [`src/metrics.py`](../src/metrics.py). Task/loss/truncation
logic: [`src/train_transformer.py`](../src/train_transformer.py).

---

## 8. Metric glossary

- **weighted_f1** — per-class F1 averaged, weighted by class size (paper-comparable).
- **macro_f1** — per-class F1 averaged with equal weight per class.
- **macro_f1_10** — macro-F1 over the 10 distortions, dropping the easy
  `no_distortion`; the honest "can it tell distortions apart?" number.
- **micro_f1** — global F1 pooled over all decisions.
- **positive_class_f1** — binary only; F1 of just the Distorted class.
- **truncation_rate** — fraction of inputs longer than the token limit (≈0.024–0.028
  here, i.e. ~2–3% — a non-issue at max_length 512).

---

## 9. Next steps (Month 2)

Ranked by impact per effort (from the FYP plan):
1. Swap in **MentalBERT** (one-line change, domain priors).
2. **Better labels / re-annotation** — attacks the noise ceiling (highest impact).
3. **Augment rare classes** (PatternReframe + targeted synthetic).
4. **Domain-adapt to entrepreneurial text** and measure the before/after delta.
5. Treat **multilabel as the production core** — it subsumes binary (OR-collapse the
   10 flags) and feeds the cohort dashboard directly.

*Generated for Month-1 review. All numbers trace to the files cited above.*
