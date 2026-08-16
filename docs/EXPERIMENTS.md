# Experiment log — EmpowerLens

Chronological record of every configuration tried, what it scored, and what was
concluded. All figures are **test split, mean ± std over seeds 42 / 1337 / 2024**
unless noted. Compiled table: `results/all_experiments.csv`
(regenerate with `python -m src.compile_results`).

**Reference points**

| | binary F1 | multiclass F1 |
|---|---|---|
| Shreevastava & Foltz (2021) headline | 0.79 | all classifiers < 0.30 |
| Human inter-annotator agreement (distortion *type*) | — | **~33.7%** ⚠️ unverified |

⚠️ Two figures still need checking against the source paper before citation:
which F1 variant the 0.79 refers to (weighted vs positive-class), and the 33.7%.

---

## E1 — Classical replication (Month 1, frozen)

`cd_pipeline.py` + `binary_f1_results.csv` / `multiclass_f1_results.csv`.

| | best | config |
|---|---|---|
| binary | 0.74 | SVM + SIF |
| multiclass | 0.33 | MLP + LIWC |

**Invalid — leakage.** Word2Vec/Doc2Vec/SIF were fitted on the full corpus before
splitting, and the pipeline re-split on every call instead of using frozen splits.
Kept frozen as the submitted artifact; **not** a performance claim.

## E2 — Classical, leakage fixed

`cd_pipeline_fixed.py`. Every embedding fitted on train only, transformed via
inference, against the committed `data/splits/`.

| | best | config | vs E1 |
|---|---|---|---|
| binary | **0.69** | SVM + SIF | −0.05 |
| multiclass | **0.30** | k-NN (k=15) + SIF | −0.03 |

**Result: leakage was worth ~0.05 F1 on binary.** This is the honest classical
floor and the baseline every transformer must beat.

---

## E3 — Transformers, `data/splits` (Annotated only, 2,024 / 253 / 253)

| model | binary pos_f1 | multiclass macro_f1_10 | multilabel macro_f1 |
|---|---|---|---|
| roberta-base | **0.773 ± 0.026** | **0.138 ± 0.018** | **0.207 ± 0.021** |
| mental-bert-base-uncased | 0.768 ± 0.008 | 0.125 ± 0.012 | 0.191 ± 0.016 |

**What was better:** transformers beat the classical floor on binary
(0.773 vs 0.69, **+12%**).

**What lost:** the *domain-adapted* model. `mental-bert`, pre-trained on
mental-health text, is **behind plain `roberta-base` on all three tasks** — though
every gap is inside the seed spread, so "no benefit" is the safe claim, not
"worse". `mental-bert` is however noticeably more stable (±0.008 vs ±0.026).

**Also lost:** fine-grained typing. macro_f1_10 of 0.138 is far below the binary
number and near the annotation ceiling.

## E4 — CODIPAS (`data/splits_codipas_cls`, 2,621 / 328 / 328)

| model | binary pos_f1 | multiclass macro_f1_10 | multilabel macro_f1 |
|---|---|---|---|
| mental-bert-base-uncased | 0.746 ± 0.011 | **0.243 ± 0.025** | **0.276 ± 0.031** |

**What was better:** the hard tasks, dramatically — multiclass **+76%** relative
over Annotated (0.243 vs 0.138), multilabel **+33%**.

**What lost:** binary, slightly (0.746 vs 0.773).

**Unexplained.** Candidate causes and why each is weak:
- *More data* — only +29% rows; doesn't plausibly yield +76%.
- *Derived labels being easier* — CODIPAS annotates **spans**, and
  `make_splits_codipas_classification.py` collapses them to one message label by
  majority vote with an order-based tie-break. Measured: the rule only decides
  **8%** of messages (6% by arbitrary tie-break). Too small to explain the gap.
- *Class distribution* — comparable (support 16–35 vs 13–31; imbalance 3.5× vs 2.3×).

**Not comparable to the benchmark**: different test set, differently constructed
labels. Do not present as "better performance".

## E5 — Combined Annotated + CODIPAS ❌ INVALID

`data/splits_combined`, mental-roberta-base and deberta-v3-base.

| model | binary pos_f1 | multiclass macro_f1_10 | multilabel macro_f1 |
|---|---|---|---|
| mental-roberta-base | 0.821 | 0.188 | 0.279 |
| deberta-v3-base | 0.807 | 0.172 | 0.193 |

**These are the highest numbers in the repo and all of them are leakage.**

```
data/splits          train-in-val   0    train-in-test   0     clean
data/splits_combined train-in-val 194    train-in-test 189     76.7% / 74.7%
```

**Root cause:** `make_splits_combined.py` is correct — it keeps val/test frozen.
Its *assumption* is wrong: it treats CODIPAS as disjoint from Annotated. It isn't.
**1,937 of CODIPAS's 2,621 rows (74%) are already in `Annotated_data.csv`** — 1,554
in train, 194 in val, 189 in test. Only 684 rows are genuinely new.

### Leakage quantified — a result in its own right

Same model, same metric, leaked vs clean:

| metric | leaked | clean | inflation |
|---|---|---|---|
| multiclass `macro_f1_10` | 0.188 | 0.125 | **+50%** |
| multiclass `weighted_f1` | 0.338 | 0.280 | **+21%** |

**Macro inflates 2.4× more than weighted.** Memorising a rare test example lifts
macro-F1 far more than weighted, so leakage flatters exactly the metric that
matters most for the rare distortion classes. *Caveat: the clean run also had
fewer rows (2,024 vs 4,645), so this is an upper bound on the leakage effect.*

---

## E6 — Two-stage cascade, `data/splits` (2026-08-16)

`mental-roberta-base` for both stages. Stage 1 = binary on the full splits;
Stage 2 = multilabel on `data/splits_stage2` (1,278 / 158 / 161, derived by
*filtering* to `y_bin == 1`, not re-splitting). `--max-length 256 --truncation
head_tail --batch-size 32`; Stage 2 adds focal loss (γ=2.0) + LLRD (0.9).

| | weighted_f1 | macro_f1 | micro_f1 | pos_class_f1 |
|---|---|---|---|---|
| **Stage 1** (binary) | 0.738 ± 0.012 | 0.720 ± 0.010 | 0.735 ± 0.014 | **0.785 ± 0.018** |
| **Cascade** (end-to-end) | **0.229 ± 0.023** | **0.225 ± 0.027** | 0.235 ± 0.025 | — |
| Flat multiclass | 0.280 ± 0.022 | — | 0.275 ± 0.028 | 0.124 ± 0.008 (macro_f1_10) |
| *Stage 2 isolated* ‡ | *0.273 ± 0.017* | *0.262 ± 0.023* | *0.278 ± 0.015* | — |

‡ distorted-only inputs — removes 92 No-Distortion test rows (36% of the set) and
hides Stage 1's false negatives. Tagged `[stage2-isolated]`. Diagnostic only.

**What was better**

- **The cascade tax is small.** 0.262 → 0.225 = **86% retained**. Predicted to be
  much worse; Stage 1's recall (0.785) is high enough that few distorted rows are
  lost. *Error propagation is not the cascade's weakness.*
- **Stage 1 essentially matches the paper** — 0.785 vs 0.79, and clears the
  leakage-fixed classical floor of 0.69 by **+14%**.

**What lost / unresolved**

- **No measurable win over flat.** Cascade 0.225 ± 0.027 vs flat roberta-base
  0.207 ± 0.021 — the gap (0.018) is inside the error bars. Honest claim:
  *"indistinguishable"*, not *"better"*. And backbone + truncation config differ,
  so even that is not like-for-like. **A matched flat baseline is still missing.**
- **Stage 2 overfits badly** — val 0.337 → test 0.262, a **22% drop**.
- **Seed instability** — cascade macro_f1 by seed: 0.250 / 0.229 / 0.197, **±12%**
  of the mean.

**Conclusion: the binding constraint is data volume, not architecture.**
~128 training examples per class (rarest 101). The overfitting and seed variance
are both classic small-data signatures. Rearranging models has now been tested and
does not move the number; more data is the untested lever.

---

## Cross-cutting findings

**1. Binary is solved-ish; typing is not.** Across every model and dataset, binary
sits at **0.75–0.79** and fine-grained typing at **0.12–0.28**. That gap is the
annotation ceiling (~33.7% human agreement on type), not a modelling failure.

**2. Model choice barely matters.** roberta-base ≈ mental-bert ≈ mental-roberta,
all within seed noise. Domain-adapted pre-training bought nothing measurable.
Do not spend more time on backbone selection.

**3. A macro-F1 of 0.70 on the 10-class task is not a reachable target.** Humans
agree ~1/3 of the time on which distortion is present. Any result far above that
is more likely a bug than a breakthrough — as E5 demonstrated, where leakage alone
moved macro_f1_10 from 0.125 to 0.188.

---

## Infrastructure defects found (methodologically relevant)

Each of these silently corrupted or blocked results; all are fixed.

| defect | effect |
|---|---|
| `sh()` deadlock on a full 64KB pipe buffer | training appeared to "hang" at random; killed 3 sessions incl. one 11.4-hour loss |
| Notebook state in kernel memory | `NameError: STAGE2_SPLITS` after any restart — killed the entire Stage 2 + cascade eval |
| Cell 5 shadowing the bootstrap | Stage 2 silently rebuilt from the **leaked** splits |
| `67dccf6` merge regression | dropped 11 CLI flags from `train_transformer.py`; the cascade notebook on `main` could not run at all |
| Seed embedded in cascade `model_name` | mean ± std came out `NaN` |
| Dataset absent from cascade `model_name` | a CODIPAS run would have **silently replaced** Annotated rows in `paper_comparison.csv` |
| `meta.json` did not record `splits` | impossible to tell afterwards which dataset a checkpoint was trained on |
| No guard on distorted-only eval | isolated Stage-2 numbers could be reported as cascade results |

**Lesson for the writeup:** three of the four "surprising" results this project
produced turned out to be infrastructure defects, not findings. Every headline
number should carry its splits directory and a leakage check.

---

## Moving ahead — ranked

| # | Experiment | Cost | Question it answers |
|---|---|---|---|
| 1 | **Flat multilabel on `data/splits`**, matched backbone + config | ~5 min GPU | Is the cascade actually better than flat? (currently unanswerable) |
| 2 | **Cascade on CODIPAS** | ~15 min GPU | Is CODIPAS's advantage in the data or the architecture? |
| 3 | **PatternReframe → Stage 2** (`src/make_splits_patternreframe.py`) | ~20 min GPU | Does 7.8× more Stage-2 data lift the rare classes? |
| 4 | **Span-level training** (§4b of TODO) | ~150 lines + GPU | Does removing derived labels while keeping context beat both? |
| 5 | **Synthetic pipeline** (`src/synth/`) | ~₨1,000 | Does in-domain generated data transfer where borrowed data doesn't? |

**#3 detail:** PatternReframe adds **8,712 rows**, taking Stage 2 from 1,278 →
**9,990** (7.8×), balanced at ~1,000–2,400 per class. Two known limits: it has
**no** No-Distortion rows (so it cannot help Stage 1) and **zero** coverage of
`emotional_reasoning`. Main risk is a length mismatch — median **17 words** vs
**129** in Annotated. If it fails, that is itself a result about what shape the
synthetic data needs to be.

**Still open, non-experimental:** dedupe CODIPAS against the frozen Annotated
val/test so `splits_combined` becomes usable (and tell Lumia — her committed
results depend on it); fix the corrupted `CLAUDE.md` on main; verify the paper's
F1 variant and the 33.7% agreement figure.
