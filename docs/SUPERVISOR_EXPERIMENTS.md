# Experiments 1–8 — status and results

Organised against the eight experiments as specified. Every number is traceable to a
file in this repo; nothing here is estimated.

| | experiment | status |
|---|---|---|
| **E1** | Data/label audit | ✅ **complete** |
| **E2** | Dataset ablation | ⚠️ **partial — protocol violation** |
| **E3** | CE vs class-weighted CE | ❌ not run (runnable) |
| **E4** | Focal / class-balanced loss | ❌ not run (runnable) |
| **E5** | Weighted sampling | ❌ not run (runnable) |
| **E6** | Multilabel per-label + thresholds | ✅ **complete** |
| **E7** | Flat vs cascade | ✅ **complete** |
| **E8** | Long-context model | ✅ **complete — gate says do not proceed** |

Four of eight answered. E3/E4/E5 are implemented in
`experiments/experiments_flat_mentalroberta.py` and need GPU time, not code.

> **Read this before comparing any two numbers below.** Results come from two
> different corpora with different test sets:
>
> * **Annotated** (`data/splits`) — 2,024 / 253 / 253, human-annotated, leakage-free.
> * **Combined** (`data/splits_combined`, rebuilt) — 2,503 / 313 / 313, Annotated +
>   CODIPAS deduplicated, leakage-free.
>
> They are different exams. A higher number on Combined is not a better model.

---

## Experiment 1 — Data/label audit ✅

Full output: **`docs/audit_data.md`** (`python -m src.audit_data`). No GPU required.

### Split sizes and class imbalance

`data/splits` — 2,024 / 253 / 253 (80/10/10), multi-label stratified, `random_state=42`.

**Imbalance ratio 9.4 : 1** — `no_distortion` 746 rows vs `all_or_nothing` 79 in train.
This is substantial, which is what makes **E3 worth running** (the audit was specified
as its gate).

### Multilabel prevalence — train / val / test

| label | train | prevalence | val | test |
|---|---|---|---|---|
| `mind_reading` | 232 | 11.5% | 32 | 31 |
| `overgeneralization` | 222 | 11.0% | 27 | 28 |
| `magnification` | 196 | 9.7% | 24 | 25 |
| `fortune_telling` | 168 | 8.3% | 21 | 21 |
| `personalization` | 162 | 8.0% | 20 | 20 |
| `labeling` | 162 | 8.0% | 20 | 21 |
| `emotional_reasoning` | 135 | 6.7% | 17 | 17 |
| `mental_filter` | 121 | 6.0% | 15 | 15 |
| `should_statements` | 108 | 5.3% | 14 | 13 |
| `all_or_nothing` | 101 | 5.0% | 12 | 13 |

**No split issue.** Every label's val and test counts sit within ±1 of its expected
share, which is what the stratified split was for.

### Label co-occurrence

Mean **0.79 labels per row**, max 2, and **16.3%** of rows carry 2+ labels. Top pairs:

| pair | count |
|---|---|
| `personalization` + `labeling` | 17 |
| `mental_filter` + `magnification` | 16 |
| `emotional_reasoning` + `fortune_telling` | 15 |
| `overgeneralization` + `fortune_telling` | 13 |
| `mind_reading` + `magnification` | 12 |

**Worth stating plainly:** only one row in six is genuinely multi-label, and no row
exceeds two labels — an artefact of the source annotation, which records a *dominant*
distortion plus one optional *secondary*. So the multilabel head is modelling a task
that is single-label 84% of the time. That caps how much a multilabel formulation can
add over multiclass, and it is a fair criticism of the framing rather than of the models.

### Per-class precision / recall / F1 and confusion matrices

Produced per run: `results_*/per_class_<model>_<task>_<seed>.csv` and
`results_*/confusion_<model>_<task>_<seed>.png`, with a `_no_nd` variant that drops
`no_distortion` so it cannot dominate the picture.

---

## Experiment 2 — Dataset ablation ⚠️

The three arms exist, but **not under a common evaluation protocol**, which is what
the experiment asked for.

| arm | trained on | **tested on** | macro_f1 | seeds |
|---|---|---|---|---|
| Annotated only | Annotated | Annotated (253) | 0.237 ± 0.030 | 42/1337/2024 |
| CODIPAS only | CODIPAS | **CODIPAS (328)** | 0.270 ± 0.027 | 42/1337/2024 |
| Annotated + CODIPAS | Combined | **Combined (313)** | 0.308 ± 0.007 | 42/1337/2024 |

Individual seed values are in `results/all_experiments.csv` (one row per seed) and
`results_experiments/exp6/exp6_all_seed_results.csv`.

### Why this cannot yet answer the question

**Each arm is scored on its own test set.** The ranking above may be measuring which
test set is easiest, not which training set is best. Two specific confounds:

1. **Label convention.** CODIPAS's labels are produced by an aggregation rule, not
   human annotators. On the 2,520 texts the two corpora share, they agree **36.8%**
   of the time (Cohen's κ = 0.199, "slight"), and disagreement is directional — 559
   cases of human-says-distorted / rule-says-clean against 286 the other way. Full
   analysis in `docs/codipas_agreement.md`. A rule-generated label set is more
   self-consistent and therefore easier to predict, independent of data volume.
2. **Configuration.** The Combined arm ran at `max_length 512`; the Annotated arm at
   `256`. Per E8 below, that changes the truncated fraction from 24.9% to 2.4% — a
   large difference that has nothing to do with the dataset.

### What would fix it

`data/splits_codipas_transfer_matched` is already built and verified: CODIPAS-trained,
scored on the **frozen Annotated test set**, downsampled to 2,024 rows so volume is not
a second confound, `train-in-test = 0`. Three runs make arm 2 comparable. The Combined
arm needs re-running at matched `max_length`.

---

## Experiment 3 — Multiclass imbalance ❌

**Not run.** The audit gate is satisfied — 9.4:1 imbalance is substantial, so the
comparison is warranted.

Implemented in `experiments/experiments_flat_mentalroberta.py` as `--loss ce` vs
`--loss weighted_ce` (`--experiment 3`). Every existing multiclass result uses
**weighted CE**, so the unweighted baseline has never been measured.

*Expectation to state up front:* weighting trades precision for rare-class recall, and
macro-F1 penalises that trade. Plain CE may well win. The value of the run is
justifying the default, not confirming it.

---

## Experiment 4 — Alternative loss ❌

**Not run.** `--loss focal` and `--loss class_balanced` (Cui et al. effective-number
weighting, `--cb-beta 0.999`) are implemented for binary and multiclass.

Focal loss *has* been used on multilabel (`--loss focal --focal-gamma 2.0`), but that
is E6's territory, not the multiclass comparison specified here.

---

## Experiment 5 — Sampling ❌

**Not run.** `--sampler weighted` is implemented (`build_sampler`, wired through a
custom `_get_train_sampler`). Every result to date used `sampler: none`, as recorded
in each run's `meta.json`.

Blocked on E3/E4 by design — it compares against *the best loss-based approach*, which
those two identify.

---

## Experiment 6 — Multilabel ✅

`results_experiments/exp6/`. Backbone `mental/mental-roberta-base`, 10 labels,
weighted BCE (`pos_weight` from class frequency = the label-wise loss weighting asked
for), 3 seeds, **Combined corpus**, `max_length 512`, head truncation.

### Test results

| metric | mean ± SD |
|---|---|
| macro_f1 | **0.308 ± 0.007** |
| weighted_f1 | 0.319 ± 0.007 |
| micro_f1 | 0.310 ± 0.005 |

Per seed (42 / 1337 / 2024): macro_f1 **0.316 / 0.307 / 0.302** — consistent, SD 0.007.

### Per-label F1

| label | F1 | | label | F1 |
|---|---|---|---|---|
| `fortune_telling` | 0.427 ± 0.025 | | `mind_reading` | 0.329 ± 0.041 |
| `labeling` | 0.426 ± 0.029 | | `personalization` | 0.323 ± 0.018 |
| `should_statements` | 0.414 ± 0.030 | | `mental_filter` | 0.303 ± 0.035 |
| `emotional_reasoning` | 0.329 ± 0.047 | | `overgeneralization` | 0.262 ± 0.050 |
| | | | `all_or_nothing` | 0.160 ± 0.056 |
| | | | `magnification` | 0.112 ± 0.049 |

`magnification` (0.112) and `all_or_nothing` (0.160) are the weak classes, and their
SDs are the largest — they are also two of the three rarest labels, so this is the
imbalance problem showing through at the per-label level.

### Threshold optimisation

Per-class thresholds tuned on **validation only**, then held fixed for test — exactly
as specified. Seed 42's selected thresholds, recorded in the checkpoint's `meta.json`:

```
[0.60, 0.65, 0.70, 0.75, 0.45, 0.60, 0.65, 0.65, 0.65, 0.65]
```

All above the 0.5 default except `all_or_nothing` at 0.45, i.e. the tuning is mostly
*suppressing* over-prediction driven by `pos_weight`.

---

## Experiment 7 — Flat vs cascade ✅

**Not answered by the flat suite** — `experiments/HOW_TO_RUN.txt` states the cascade
"is not run or compared anywhere in this suite, including Experiment 7", and
`results_experiments/exp7/` contains three **0-byte** files. It is answered by the
cascade track instead.

Everything below: `mental/mental-roberta-base`, `data/splits`, identical
preprocessing (`max_length 256`, `head_tail`) and training budget, 3 seeds.

| system | metric | test |
|---|---|---|
| **Flat multilabel** | macro_f1 | **0.237 ± 0.030** |
| **Cascade end-to-end** | macro_f1 | **0.240 ± 0.012** |
| Cascade — Stage 1 (binary) | positive_class_f1 | 0.794 ± 0.012 |
| Cascade — Stage 2 (multilabel, isolated) | macro_f1 | 0.277 ± 0.016 |

### Finding

**The cascade and the flat model are indistinguishable: +0.003, against seed SDs of
0.030 and 0.012.** The two-stage architecture buys nothing at this data scale.

Three supporting points:

* **Stage 1 is not the bottleneck.** At `positive_class_f1` 0.794 it matches the
  reference paper's reported 0.79. Detection works; fine-grained typing is what fails.
* **Stage 2 in isolation scores 0.277**, so the cascade retains ~86% of it end to end.
  The 0.037 lost is Stage 1's false negatives, which Stage 2 never sees.
* **Replicated on a second corpus.** On CODIPAS the same comparison gives cascade
  0.259 ± 0.007 vs flat 0.270 ± 0.027 — again overlapping, and if anything favouring
  flat.

Stage 2's isolated number is **not** a cascade result and must not be quoted as one:
it is measured only on rows already known to be distorted, so it hides every Stage 1
false negative. `src/evaluate.py` refuses distorted-only splits without an explicit
`--allow-distorted-only` flag to prevent that error.

---

## Experiment 8 — Long-context model ✅

Gate analysis first, as specified. Tokenizer `mental/mental-roberta-base`, lengths
include special tokens. Source: `docs/audit_data.md`.

### Sequence length in tokens

| split | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| train | 158 | 443 | 491 | 605 | 1396 |
| val | 144 | 377 | 457 | 545 | 585 |
| test | **163** | **422** | **476** | 528 | 610 |

### Truncated fraction

| max_length | train | val | test |
|---|---|---|---|
| 128 | 61.1% | 62.5% | 63.6% |
| 256 | 28.5% | 22.9% | **24.9%** |
| **512** | 3.3% | 2.8% | **2.4%** |

### Verdict: do not proceed to Longformer

At 512 tokens only **2.4% of test rows** are truncated, and p95 is 476 — comfortably
inside the window. The condition set for this experiment ("only if a substantial
amount of information is being truncated") is **not met**. A Longformer would add
cost and complexity to recover roughly one row in forty, most of which overrun by a
short margin.

### But this exposes a real problem in our own configuration

The cascade and flat runs (E7) used **`max_length 256`, where 24.9% of test rows are
truncated** — a quarter of the evaluation set losing content. The E6/Combined runs used
512 (2.4%).

So part of the gap between 0.237 (E7, Annotated, 256) and 0.308 (E6, Combined, 512) may
be truncation rather than data. `head_tail` truncation softens it — it keeps the first
128 and last N tokens, and a reflection's conclusion often carries the distortion — but
it does not eliminate it.

**Recommended action:** re-run E7 at `max_length 512` before drawing any conclusion
from the E6-vs-E7 difference. This is the cheapest high-value run outstanding, and it
was surfaced by the E8 analysis rather than by any model result.

---

## Provenance notes

**`results_experiments/` contains duplicates.** Its 18 top-level `eval_*.json` files,
`all_experiments.csv`, the confusion PNGs and `codipas_agreement_per_class.csv` are
**byte-identical copies** of files in `results/`. The genuinely new outputs are
`exp6/` (complete) and `exp7/` (empty). The `valid` column in that CSV comes from
`src/compile_results.py` and flags the old leaked `splits_combined`, unrelated to
E1–E8.

**exp6's splits are not in the repo.** They came from `izza-space` commit `5b976a4`,
which was excluded from `main` because the same commit also rewrites
`data/splits_stage2` — the splits every Stage-2 and cascade result depends on. exp6's
numbers are leakage-free and valid, but not currently reproducible from `main`. Landing
the `splits_combined` fix separately would resolve this.

**One result set is excluded entirely.** The old `data/splits_combined` put 201/253 val
and 195/253 test rows into train; `results_combined/` is flagged `valid=False` and struck
through in `docs/RESULTS.md`. Do not cite it.

---

## Suggested order for the remaining work

1. **Re-run E7 at `max_length 512`** — removes the truncation confound from every
   cross-corpus comparison. 6 runs.
2. **E2 properly** — CODIPAS arm on `data/splits_codipas_transfer_matched`, Combined
   arm at matched `max_length`. 6 runs.
3. **E3** (CE vs weighted CE) — gate satisfied at 9.4:1. 6 runs.
4. **E4** then **E5**, in that order — E5 is defined relative to E4's winner.

E8 needs no further work: the gate closed it.

---

*Numbers: `results/all_experiments.csv`, `docs/RESULTS.md`, `docs/audit_data.md`,
`results_experiments/exp6/`. Design notes: `docs/EXPERIMENTS.md`. Label agreement:
`docs/codipas_agreement.md`.*
