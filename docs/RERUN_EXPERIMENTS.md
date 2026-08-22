# Re-run experiments — DistilBERT on EmpowerLens

**Generated** 2026-08-22 16:40 UTC by `python -m src.make_rerun_table` from `results_tutorial_distilbert_smoke/`. Do not edit by hand — re-run the generator.

> 🚧 **THIS IS SMOKE DATA — every number below is meaningless.**
> Produced by `--smoke` (64 rows, 1 epoch) purely to exercise the
> template. Regenerate after the real run before showing anyone.

## Setup

| | |
|---|---|
| **Model** | `distilbert-base-uncased` |
| **Datasets** | `data/splits` |
| **Tasks** | `binary`, `multiclass`, `multilabel` |
| **Recipe** | lr 2e-05, batch 8, 1 epoch, max_length 128 |
| **Seeds** | 42, 1337 |
| **Selection** | best epoch on val, by each task's headline metric; test read only by `src.evaluate` |
| **Source recipe** | [wellally.tech DistilBERT tutorial](https://www.wellally.tech/blog/python-cognitive-distortion-transformer-tutorial) |

> **Read this before comparing any two numbers.** Rows are grouped by task on purpose. `binary` is 2 classes, `multiclass` 11, `multilabel` 10 — three different exams, so a higher macro-F1 on one is not a better model. The same applies across datasets: `data/splits` and `data/splits_combined` have different test sets.

---

## Task: `binary`

**2 classes — is any distortion present?** `positive_class_f1` is the headline: catching the distorted rows is the job, and 36.9% of rows are undistorted so accuracy flatters.

### All runs

| exp | dataset | model | loss | threshold | split | seeds | macro-P | macro-R | macro-F1 | micro-F1 | weighted-F1 | accuracy | ROC-AUC | **positive-F1** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B1, B2 | `data/splits` | `distilbert-base-uncased` | CrossEntropy (unweighted) | argmax | **val** | 2 | 0.352 ± 0.000 | 0.500 ± 0.000 | 0.413 ± 0.000 | 0.703 ± 0.000 | 0.581 ± 0.000 | 0.703 ± 0.000 | 0.678 ± 0.065 | 0.826 ± 0.000 |
| B2 | `data/splits` | `distilbert-base-uncased` | CrossEntropy + class weights | argmax | **val** | 2 | 0.461 ± 0.155 | 0.536 ± 0.051 | 0.449 ± 0.050 | 0.594 ± 0.155 | 0.534 ± 0.066 | 0.594 ± 0.155 | 0.673 ± 0.066 | 0.659 ± 0.236 |

### Best per seed — by `positive_class_f1`

| seed | best config | positive_class_f1 | macro-F1 | accuracy |
|---|---|---|---|---|
| 42 | CrossEntropy (unweighted) @ argmax | **0.826** | 0.413 | 0.703 |
| 1337 | CrossEntropy (unweighted) @ argmax | **0.826** | 0.413 | 0.703 |

Largest across-seed spread for any single config: **± 0.167** on `positive_class_f1`. Treat any gap smaller than this as noise.

### Named experiments

#### B1 — Baseline: unweighted cross-entropy

**Question.** Can the model tell distorted from undistorted at all?

| loss | mechanism | threshold | macro-P | macro-R | macro-F1 | micro-F1 | weighted-F1 | accuracy | ROC-AUC | **positive-F1** |
|---|---|---|---|---|---|---|---|---|---|---|
| CrossEntropy (unweighted) | softmax baseline: every row counts the same | argmax | 0.352 ± 0.000 | 0.500 ± 0.000 | 0.413 ± 0.000 | 0.703 ± 0.000 | 0.581 ± 0.000 | 0.703 ± 0.000 | 0.678 ± 0.065 | 0.826 ± 0.000 |

**Measured.** Best arm: `ce` @ argmax — positive_class_f1 0.826.

> **Finding:** _(write after reading the run — what does this mean, and does it change the recommendation?)_

#### B2 — Class weighting

**Question.** 1,278 of 2,024 training rows are distorted — mild imbalance. Does inverse-frequency weighting help, or just trade precision for recall?

| loss | mechanism | threshold | macro-P | macro-R | macro-F1 | micro-F1 | weighted-F1 | accuracy | ROC-AUC | **positive-F1** |
|---|---|---|---|---|---|---|---|---|---|---|
| CrossEntropy (unweighted) | softmax baseline: every row counts the same | argmax | 0.352 ± 0.000 | 0.500 ± 0.000 | 0.413 ± 0.000 | 0.703 ± 0.000 | 0.581 ± 0.000 | 0.703 ± 0.000 | 0.678 ± 0.065 | 0.826 ± 0.000 |
| CrossEntropy + class weights | inverse-frequency weight per class | argmax | 0.461 ± 0.155 | 0.536 ± 0.051 | 0.449 ± 0.050 | 0.594 ± 0.155 | 0.534 ± 0.066 | 0.594 ± 0.155 | 0.673 ± 0.066 | 0.659 ± 0.236 |

**Measured.** Best arm: `ce` @ argmax — positive_class_f1 0.826.

> **Finding:** _(write after reading the run — what does this mean, and does it change the recommendation?)_

#### Test-set confirmation

**Question.** Does the val result hold on data never used for any decision?

_Not run yet._ After choosing a configuration on val:

```
python -m src.evaluate --checkpoint checkpoints/tutorial_binary_distilbert-base-uncased_42 --out results_tutorial_distilbert_smoke
```

> **Finding:** _(write after the test pass — how big is the val→test drop, and is the ranking of configurations the same?)_

### Per-class breakdown

**loss = `ce`** (mean over seeds)

| class | precision | recall | F1 | val support |
|---|---|---|---|---|
| `non_distorted` | 0.000 | 0.000 | **0.000** ⚠️ | 19 |
| `distorted` | 0.703 | 1.000 | **0.826** | 45 |

⚠️ 1/2 classes scored F1 = 0.000: `non_distorted`.

**loss = `weighted_ce`** (mean over seeds)

| class | precision | recall | F1 | val support |
|---|---|---|---|---|
| `non_distorted` | 0.170 | 0.395 | **0.238** | 19 |
| `distorted` | 0.752 | 0.678 | **0.659** | 45 |

### Prior runs in this repo, same task

| dataset | model | split | seeds | macro-F1 | micro-F1 | weighted-F1 | macro-F1(10) | positive-F1 |
|---|---|---|---|---|---|---|---|---|
| `data/splits` | `cascade[binary_mental-roberta-base+multilabel_mental-roberta-base]` | **test** | 3 | 0.717 | 0.738 | 0.738 | — | 0.794 |
| `data/splits` | `cascade[binary_mental-roberta-base+multilabel_mental-roberta-base]` | **val** | 3 | 0.757 | 0.769 | 0.771 | — | 0.812 |
| `data/splits` | `mental/mental-bert-base-uncased` | **test** | 3 | 0.700 | 0.715 | 0.718 | — | 0.768 |
| `data/splits` | `mental/mental-bert-base-uncased` | **val** | 3 | 0.733 | 0.742 | 0.745 | — | 0.781 |
| `data/splits` | `mental/mental-roberta-base` | **test** | 3 | 0.717 | 0.738 | 0.738 | — | 0.794 |
| `data/splits` | `mental/mental-roberta-base` | **val** | 3 | 0.757 | 0.769 | 0.771 | — | 0.812 |
| `data/splits` | `roberta-base` | **test** | 3 | 0.700 | 0.718 | 0.720 | — | 0.773 |
| `data/splits` | `roberta-base` | **val** | 3 | 0.755 | 0.764 | 0.767 | — | 0.802 |

> These are **prior** runs from other scripts, shown for context only. They differ in backbone and in recipe, so a gap here mixes several causes — the controlled comparison is the loss ablation above, which holds everything else fixed.

---

## Task: `multiclass`

**11 classes — `no_distortion` plus the 10 distortions, one label per row.** `macro_f1_10` is the headline: plain macro-F1 includes the easy `no_distortion` majority class and overstates the model.

### All runs

| exp | dataset | model | loss | threshold | split | seeds | macro-P | macro-R | macro-F1 | micro-F1 | weighted-F1 | accuracy | ROC-AUC | **macro-F1(10)** | no-dist F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MC1, MC2 | `data/splits` | `distilbert-base-uncased` | CrossEntropy (unweighted) | argmax | **val** | 2 | 0.039 ± 0.006 | 0.085 ± 0.024 | 0.051 ± 0.010 | 0.242 ± 0.077 | 0.144 ± 0.015 | 0.242 ± 0.077 | 0.514 ± 0.006 | 0.011 ± 0.008 | 0.449 ± 0.026 |
| MC2 | `data/splits` | `distilbert-base-uncased` | CrossEntropy + class weights | argmax | **val** | 2 | 0.065 ± 0.050 | 0.105 ± 0.006 | 0.034 ± 0.001 | 0.102 ± 0.033 | 0.050 ± 0.034 | 0.102 ± 0.033 | 0.512 ± 0.004 | 0.027 ± 0.012 | 0.095 ± 0.135 |

### Best per seed — by `macro_f1_10`

| seed | best config | macro_f1_10 | macro-F1 | accuracy |
|---|---|---|---|---|
| 42 | CrossEntropy + class weights @ argmax | **0.036** | 0.033 | 0.078 |
| 1337 | CrossEntropy + class weights @ argmax | **0.019** | 0.034 | 0.125 |

Largest across-seed spread for any single config: **± 0.009** on `macro_f1_10`. Treat any gap smaller than this as noise.

### Named experiments

#### MC1 — Baseline: unweighted cross-entropy

**Question.** Can one softmax over 11 mutually exclusive classes pick the dominant distortion?

| loss | mechanism | threshold | macro-P | macro-R | macro-F1 | micro-F1 | weighted-F1 | accuracy | ROC-AUC | **macro-F1(10)** | no-dist F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CrossEntropy (unweighted) | softmax baseline: every row counts the same | argmax | 0.039 ± 0.006 | 0.085 ± 0.024 | 0.051 ± 0.010 | 0.242 ± 0.077 | 0.144 ± 0.015 | 0.242 ± 0.077 | 0.514 ± 0.006 | 0.011 ± 0.008 | 0.449 ± 0.026 |

**Measured.** Best arm: `ce` @ argmax — macro_f1_10 0.011.

> **Finding:** _(write after reading the run — what does this mean, and does it change the recommendation?)_

#### MC2 — Class weighting

**Question.** `no_distortion` is 36.9% of rows and `all_or_nothing` 5.0%. Does inverse-frequency weighting lift `macro_f1_10`?

| loss | mechanism | threshold | macro-P | macro-R | macro-F1 | micro-F1 | weighted-F1 | accuracy | ROC-AUC | **macro-F1(10)** | no-dist F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CrossEntropy (unweighted) | softmax baseline: every row counts the same | argmax | 0.039 ± 0.006 | 0.085 ± 0.024 | 0.051 ± 0.010 | 0.242 ± 0.077 | 0.144 ± 0.015 | 0.242 ± 0.077 | 0.514 ± 0.006 | 0.011 ± 0.008 | 0.449 ± 0.026 |
| CrossEntropy + class weights | inverse-frequency weight per class | argmax | 0.065 ± 0.050 | 0.105 ± 0.006 | 0.034 ± 0.001 | 0.102 ± 0.033 | 0.050 ± 0.034 | 0.102 ± 0.033 | 0.512 ± 0.004 | 0.027 ± 0.012 | 0.095 ± 0.135 |

**Measured.** Best arm: `weighted_ce` @ argmax — macro_f1_10 0.027, **+0.016** vs the baseline arm.

> **Finding:** _(write after reading the run — what does this mean, and does it change the recommendation?)_

#### Test-set confirmation

**Question.** Does the val result hold on data never used for any decision?

_Not run yet._ After choosing a configuration on val:

```
python -m src.evaluate --checkpoint checkpoints/tutorial_multiclass_distilbert-base-uncased_42 --out results_tutorial_distilbert_smoke
```

> **Finding:** _(write after the test pass — how big is the val→test drop, and is the ranking of configurations the same?)_

### Per-class breakdown

**loss = `ce`** (mean over seeds)

| class | precision | recall | F1 | val support |
|---|---|---|---|---|
| `no_distortion` | 0.327 | 0.763 | **0.449** | 19 |
| `emotional_reasoning` | 0.000 | 0.000 | **0.000** ⚠️ | 4 |
| `overgeneralization` | 0.016 | 0.083 | **0.027** | 6 |
| `mental_filter` | 0.000 | 0.000 | **0.000** ⚠️ | 4 |
| `should_statements` | 0.000 | 0.000 | **0.000** ⚠️ | 2 |
| `all_or_nothing` | 0.000 | 0.000 | **0.000** ⚠️ | 6 |
| `mind_reading` | 0.000 | 0.000 | **0.000** ⚠️ | 2 |
| `fortune_telling` | 0.083 | 0.083 | **0.083** | 6 |
| `magnification` | 0.000 | 0.000 | **0.000** ⚠️ | 4 |
| `personalization` | 0.000 | 0.000 | **0.000** ⚠️ | 5 |
| `labeling` | 0.000 | 0.000 | **0.000** ⚠️ | 6 |

⚠️ 8/11 classes scored F1 = 0.000: `emotional_reasoning`, `mental_filter`, `should_statements`, `all_or_nothing`, `mind_reading`, `magnification`, `personalization`, `labeling`.

**loss = `weighted_ce`** (mean over seeds)

| class | precision | recall | F1 | val support |
|---|---|---|---|---|
| `no_distortion` | 0.500 | 0.053 | **0.095** | 19 |
| `emotional_reasoning` | 0.000 | 0.000 | **0.000** ⚠️ | 4 |
| `overgeneralization` | 0.000 | 0.000 | **0.000** ⚠️ | 6 |
| `mental_filter` | 0.037 | 0.500 | **0.069** | 4 |
| `should_statements` | 0.000 | 0.000 | **0.000** ⚠️ | 2 |
| `all_or_nothing` | 0.000 | 0.000 | **0.000** ⚠️ | 6 |
| `mind_reading` | 0.000 | 0.000 | **0.000** ⚠️ | 2 |
| `fortune_telling` | 0.052 | 0.500 | **0.094** | 6 |
| `magnification` | 0.000 | 0.000 | **0.000** ⚠️ | 4 |
| `personalization` | 0.125 | 0.100 | **0.111** | 5 |
| `labeling` | 0.000 | 0.000 | **0.000** ⚠️ | 6 |

⚠️ 7/11 classes scored F1 = 0.000: `emotional_reasoning`, `overgeneralization`, `should_statements`, `all_or_nothing`, `mind_reading`, `magnification`, `labeling`.

### Prior runs in this repo, same task

| dataset | model | split | seeds | macro-F1 | micro-F1 | weighted-F1 | macro-F1(10) | positive-F1 |
|---|---|---|---|---|---|---|---|---|
| `data/splits` | `mental/mental-bert-base-uncased` | **test** | 3 | 0.162 | 0.275 | 0.271 | 0.125 | — |
| `data/splits` | `mental/mental-bert-base-uncased` | **val** | 3 | 0.210 | 0.333 | 0.337 | 0.170 | — |
| `data/splits` | `mental/mental-roberta-base` | **test** | 3 | 0.168 | 0.279 | 0.285 | 0.129 | — |
| `data/splits` | `mental/mental-roberta-base` | **val** | 3 | 0.229 | 0.323 | 0.333 | 0.195 | — |
| `data/splits` | `roberta-base` | **test** | 3 | 0.176 | 0.292 | 0.290 | 0.138 | — |
| `data/splits` | `roberta-base` | **val** | 3 | 0.215 | 0.354 | 0.345 | 0.175 | — |

> These are **prior** runs from other scripts, shown for context only. They differ in backbone and in recipe, so a gap here mixes several causes — the controlled comparison is the loss ablation above, which holds everything else fixed.

---

## Task: `multilabel`

**10 independent labels — a row may carry two distortions, or none.** `macro_f1` is the headline; an all-zero row means No Distortion, so there is no 11th column.

### All runs

| exp | dataset | model | loss | threshold | split | seeds | macro-P | macro-R | macro-F1 | micro-F1 | weighted-F1 | accuracy | ROC-AUC | labels/row |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ML1, ML2, ML3 | `data/splits` | `distilbert-base-uncased` | BCE (unweighted) | 0.5 flat | **val** | 2 | 0.012 ± 0.007 | 0.107 ± 0.057 | 0.022 ± 0.012 | 0.074 ± 0.037 | 0.018 ± 0.016 | 0.117 ± 0.166 | 0.484 ± 0.044 | 1.02 ± 0.85 |
| ML2 | `data/splits` | `distilbert-base-uncased` | BCE (unweighted) | swept on val | **val** | 2 | 0.091 ± 0.002 | 0.973 ± 0.009 | 0.166 ± 0.003 | 0.167 ± 0.004 | 0.173 ± 0.002 | 0.000 ± 0.000 | 0.484 ± 0.044 | 9.43 ± 0.28 |
| ML3 | `data/splits` | `distilbert-base-uncased` | BCE + pos_weight | 0.5 flat | **val** | 2 | 0.045 ± 0.015 | 0.493 ± 0.131 | 0.082 ± 0.028 | 0.145 ± 0.022 | 0.087 ± 0.030 | 0.000 ± 0.000 | 0.494 ± 0.024 | 4.97 ± 1.10 |
| ML4 | `data/splits` | `distilbert-base-uncased` | BCE + pos_weight | swept on val | **val** | 2 | 0.090 ± 0.001 | 0.960 ± 0.037 | 0.164 ± 0.002 | 0.166 ± 0.005 | 0.172 ± 0.001 | 0.000 ± 0.000 | 0.494 ± 0.024 | 9.41 ± 0.45 |
| ML3 | `data/splits` | `distilbert-base-uncased` | Focal + pos_weight | 0.5 flat | **val** | 2 | 0.060 ± 0.007 | 0.571 ± 0.040 | 0.103 ± 0.004 | 0.148 ± 0.001 | 0.110 ± 0.007 | 0.000 ± 0.000 | 0.494 ± 0.019 | 5.98 ± 0.69 |
| ML4 | `data/splits` | `distilbert-base-uncased` | Focal + pos_weight | swept on val | **val** | 2 | 0.091 ± 0.004 | 0.958 ± 0.012 | 0.165 ± 0.006 | 0.165 ± 0.006 | 0.172 ± 0.003 | 0.000 ± 0.000 | 0.494 ± 0.019 | 9.35 ± 0.36 |
| ML3 | `data/splits` | `distilbert-base-uncased` | Asymmetric (ASL) | 0.5 flat | **val** | 2 | 0.083 ± 0.023 | 0.639 ± 0.042 | 0.138 ± 0.029 | 0.168 ± 0.013 | 0.153 ± 0.038 | 0.000 ± 0.000 | 0.521 ± 0.016 | 6.00 ± 0.24 |
| ML4 | `data/splits` | `distilbert-base-uncased` | Asymmetric (ASL) | swept on val | **val** | 2 | 0.102 ± 0.011 | 0.918 ± 0.069 | 0.175 ± 0.007 | 0.165 ± 0.003 | 0.186 ± 0.011 | 0.000 ± 0.000 | 0.521 ± 0.016 | 8.71 ± 0.76 |

`labels/row` is how many of the 10 labels fire on average; the true rate is **0.79**. Far below is under-firing, far above is spraying labels.

### Best per seed — by `macro_f1`

| seed | best config | macro_f1 | macro-F1 | accuracy |
|---|---|---|---|---|
| 42 | Focal + pos_weight @ swept on val | **0.170** | 0.170 | 0.000 |
| 1337 | Asymmetric (ASL) @ swept on val | **0.179** | 0.179 | 0.000 |

Largest across-seed spread for any single config: **± 0.021** on `macro_f1`. Treat any gap smaller than this as noise.

### Named experiments

#### ML1 — Baseline: the tutorial as published

**Question.** What does the wellally.tech recipe score on real data, unmodified?

| loss | mechanism | threshold | macro-P | macro-R | macro-F1 | micro-F1 | weighted-F1 | accuracy | ROC-AUC | labels/row |
|---|---|---|---|---|---|---|---|---|---|---|
| BCE (unweighted) | the tutorial's loss: every example counts the same | 0.5 flat | 0.012 ± 0.007 | 0.107 ± 0.057 | 0.022 ± 0.012 | 0.074 ± 0.037 | 0.018 ± 0.016 | 0.117 ± 0.166 | 0.484 ± 0.044 | 1.02 ± 0.85 |

**Measured.** Best arm: `bce` @ 0.5 flat — macro_f1 0.022.

> **Finding:** _(write after reading the run — what does this mean, and does it change the recommendation?)_

#### ML2 — Threshold alone

**Question.** Same weights, same probabilities — only the decision line moves. If this recovers the gap, the model learned fine and only reported badly.

| loss | mechanism | threshold | macro-P | macro-R | macro-F1 | micro-F1 | weighted-F1 | accuracy | ROC-AUC | labels/row |
|---|---|---|---|---|---|---|---|---|---|---|
| BCE (unweighted) | the tutorial's loss: every example counts the same | 0.5 flat | 0.012 ± 0.007 | 0.107 ± 0.057 | 0.022 ± 0.012 | 0.074 ± 0.037 | 0.018 ± 0.016 | 0.117 ± 0.166 | 0.484 ± 0.044 | 1.02 ± 0.85 |
| BCE (unweighted) | the tutorial's loss: every example counts the same | swept on val | 0.091 ± 0.002 | 0.973 ± 0.009 | 0.166 ± 0.003 | 0.167 ± 0.004 | 0.173 ± 0.002 | 0.000 ± 0.000 | 0.484 ± 0.044 | 9.43 ± 0.28 |

**Measured.** Best arm: `bce` @ swept on val — macro_f1 0.166, **+0.144** vs the baseline arm.

> **Finding:** _(write after reading the run — what does this mean, and does it change the recommendation?)_

> ⚠️ Tuned thresholds are selected on val, so tuned **val** numbers are optimistic. Confirm on test before quoting.

#### ML3 — Loss alone

**Question.** Same flat 0.5 threshold, different training. If only this moves macro-F1, the unweighted loss really did stop it learning rare classes.

| loss | mechanism | threshold | macro-P | macro-R | macro-F1 | micro-F1 | weighted-F1 | accuracy | ROC-AUC | labels/row |
|---|---|---|---|---|---|---|---|---|---|---|
| BCE (unweighted) | the tutorial's loss: every example counts the same | 0.5 flat | 0.012 ± 0.007 | 0.107 ± 0.057 | 0.022 ± 0.012 | 0.074 ± 0.037 | 0.018 ± 0.016 | 0.117 ± 0.166 | 0.484 ± 0.044 | 1.02 ± 0.85 |
| BCE + pos_weight | each positive counts negatives/positives times more | 0.5 flat | 0.045 ± 0.015 | 0.493 ± 0.131 | 0.082 ± 0.028 | 0.145 ± 0.022 | 0.087 ± 0.030 | 0.000 ± 0.000 | 0.494 ± 0.024 | 4.97 ± 1.10 |
| Focal + pos_weight | pos_weight, plus down-weighting of easy examples | 0.5 flat | 0.060 ± 0.007 | 0.571 ± 0.040 | 0.103 ± 0.004 | 0.148 ± 0.001 | 0.110 ± 0.007 | 0.000 ± 0.000 | 0.494 ± 0.019 | 5.98 ± 0.69 |
| Asymmetric (ASL) | separate gammas for positives/negatives, replaces pos_weight | 0.5 flat | 0.083 ± 0.023 | 0.639 ± 0.042 | 0.138 ± 0.029 | 0.168 ± 0.013 | 0.153 ± 0.038 | 0.000 ± 0.000 | 0.521 ± 0.016 | 6.00 ± 0.24 |

**Measured.** Best arm: `asl` @ 0.5 flat — macro_f1 0.138, **+0.116** vs the baseline arm.

> **Finding:** _(write after reading the run — what does this mean, and does it change the recommendation?)_

#### ML4 — Both fixes together

**Question.** The best configuration available, for the headline comparison.

| loss | mechanism | threshold | macro-P | macro-R | macro-F1 | micro-F1 | weighted-F1 | accuracy | ROC-AUC | labels/row |
|---|---|---|---|---|---|---|---|---|---|---|
| BCE + pos_weight | each positive counts negatives/positives times more | swept on val | 0.090 ± 0.001 | 0.960 ± 0.037 | 0.164 ± 0.002 | 0.166 ± 0.005 | 0.172 ± 0.001 | 0.000 ± 0.000 | 0.494 ± 0.024 | 9.41 ± 0.45 |
| Focal + pos_weight | pos_weight, plus down-weighting of easy examples | swept on val | 0.091 ± 0.004 | 0.958 ± 0.012 | 0.165 ± 0.006 | 0.165 ± 0.006 | 0.172 ± 0.003 | 0.000 ± 0.000 | 0.494 ± 0.019 | 9.35 ± 0.36 |
| Asymmetric (ASL) | separate gammas for positives/negatives, replaces pos_weight | swept on val | 0.102 ± 0.011 | 0.918 ± 0.069 | 0.175 ± 0.007 | 0.165 ± 0.003 | 0.186 ± 0.011 | 0.000 ± 0.000 | 0.521 ± 0.016 | 8.71 ± 0.76 |

**Measured.** Best arm: `asl` @ swept on val — macro_f1 0.175.

> **Finding:** _(write after reading the run — what does this mean, and does it change the recommendation?)_

> ⚠️ Tuned thresholds are selected on val, so tuned **val** numbers are optimistic. Confirm on test before quoting.

#### Test-set confirmation

**Question.** Does the val result hold on data never used for any decision?

_Not run yet._ After choosing a configuration on val:

```
python -m src.evaluate --checkpoint checkpoints/tutorial_multilabel_distilbert-base-uncased_42 --max-labels 0 --out results_tutorial_distilbert_smoke
```

> **Finding:** _(write after the test pass — how big is the val→test drop, and is the ranking of configurations the same?)_

### Per-class breakdown

**loss = `bce`** (mean over seeds)

| class | precision | recall | F1 | val support |
|---|---|---|---|---|
| `emotional_reasoning` | 0.000 | 0.000 | **0.000** ⚠️ | 6 |
| `overgeneralization` | 0.000 | 0.000 | **0.000** ⚠️ | 8 |
| `mental_filter` | 0.000 | 0.000 | **0.000** ⚠️ | 4 |
| `should_statements` | 0.037 | 0.333 | **0.067** | 3 |
| `all_or_nothing` | 0.000 | 0.000 | **0.000** ⚠️ | 7 |
| `mind_reading` | 0.000 | 0.000 | **0.000** ⚠️ | 6 |
| `fortune_telling` | 0.034 | 0.333 | **0.062** | 6 |
| `magnification` | 0.050 | 0.400 | **0.089** | 5 |
| `personalization` | 0.000 | 0.000 | **0.000** ⚠️ | 5 |
| `labeling` | 0.000 | 0.000 | **0.000** ⚠️ | 6 |

⚠️ 7/10 classes scored F1 = 0.000: `emotional_reasoning`, `overgeneralization`, `mental_filter`, `all_or_nothing`, `mind_reading`, `personalization`, `labeling`.

**loss = `pos_bce`** (mean over seeds)

| class | precision | recall | F1 | val support |
|---|---|---|---|---|
| `emotional_reasoning` | 0.047 | 0.500 | **0.086** | 6 |
| `overgeneralization` | 0.062 | 0.500 | **0.111** | 8 |
| `mental_filter` | 0.031 | 0.500 | **0.059** | 4 |
| `should_statements` | 0.023 | 0.500 | **0.045** | 3 |
| `all_or_nothing` | 0.061 | 0.429 | **0.107** | 7 |
| `mind_reading` | 0.000 | 0.000 | **0.000** ⚠️ | 6 |
| `fortune_telling` | 0.047 | 0.500 | **0.086** | 6 |
| `magnification` | 0.040 | 0.500 | **0.075** | 5 |
| `personalization` | 0.039 | 0.500 | **0.072** | 5 |
| `labeling` | 0.101 | 1.000 | **0.184** | 6 |

⚠️ 1/10 classes scored F1 = 0.000: `mind_reading`.

**loss = `focal`** (mean over seeds)

| class | precision | recall | F1 | val support |
|---|---|---|---|---|
| `emotional_reasoning` | 0.047 | 0.500 | **0.086** | 6 |
| `overgeneralization` | 0.062 | 0.500 | **0.111** | 8 |
| `mental_filter` | 0.031 | 0.500 | **0.059** | 4 |
| `should_statements` | 0.023 | 0.500 | **0.045** | 3 |
| `all_or_nothing` | 0.094 | 0.714 | **0.165** | 7 |
| `mind_reading` | 0.049 | 0.417 | **0.088** | 6 |
| `fortune_telling` | 0.118 | 0.583 | **0.163** | 6 |
| `magnification` | 0.039 | 0.500 | **0.072** | 5 |
| `personalization` | 0.039 | 0.500 | **0.072** | 5 |
| `labeling` | 0.094 | 1.000 | **0.171** | 6 |

**loss = `asl`** (mean over seeds)

| class | precision | recall | F1 | val support |
|---|---|---|---|---|
| `emotional_reasoning` | 0.047 | 0.500 | **0.086** | 6 |
| `overgeneralization` | 0.180 | 0.750 | **0.271** | 8 |
| `mental_filter` | 0.038 | 0.500 | **0.070** | 4 |
| `should_statements` | 0.023 | 0.500 | **0.045** | 3 |
| `all_or_nothing` | 0.100 | 0.643 | **0.173** | 7 |
| `mind_reading` | 0.147 | 0.667 | **0.211** | 6 |
| `fortune_telling` | 0.121 | 0.833 | **0.207** | 6 |
| `magnification` | 0.041 | 0.500 | **0.076** | 5 |
| `personalization` | 0.039 | 0.500 | **0.072** | 5 |
| `labeling` | 0.094 | 1.000 | **0.171** | 6 |

### Prior runs in this repo, same task

| dataset | model | split | seeds | macro-F1 | micro-F1 | weighted-F1 | macro-F1(10) | positive-F1 |
|---|---|---|---|---|---|---|---|---|
| `data/splits` | `cascade[binary_mental-roberta-base+multilabel_mental-roberta-base]` | **test** | 3 | 0.240 | 0.250 | 0.247 | — | — |
| `data/splits` | `cascade[binary_mental-roberta-base+multilabel_mental-roberta-base]` | **val** | 3 | 0.250 | 0.264 | 0.261 | — | — |
| `data/splits` | `mental/mental-bert-base-uncased` | **test** | 3 | 0.191 | 0.202 | 0.195 | — | — |
| `data/splits` | `mental/mental-bert-base-uncased` | **val** | 3 | 0.262 | 0.255 | 0.263 | — | — |
| `data/splits` | `mental/mental-roberta-base` | **test** | 3 | 0.237 | 0.239 | 0.238 | — | — |
| `data/splits` | `mental/mental-roberta-base` | **val** | 3 | 0.272 | 0.277 | 0.282 | — | — |
| `data/splits` | `roberta-base` | **test** | 3 | 0.207 | 0.212 | 0.209 | — | — |
| `data/splits` | `roberta-base` | **val** | 3 | 0.275 | 0.281 | 0.287 | — | — |

> These are **prior** runs from other scripts, shown for context only. They differ in backbone and in recipe, so a gap here mixes several causes — the controlled comparison is the loss ablation above, which holds everything else fixed.

---

## Reproduce

```
python -m src.tutorial_distilbert --tasks binary,multiclass,multilabel \
    --ablation --seeds 42,1337,2024
python -m src.make_rerun_table
```

## Protocol notes

- **Never compare across tasks or across datasets.** Different class counts and different test sets are different exams.
- **val and test are different exams too.** Never average them, and never quote a test number that was used to pick a configuration.
- **Tuned thresholds are swept on val** (multilabel only), so tuned val numbers are optimistic by construction. The test row is honest.
- **Accuracy is never a headline.** On multilabel it is exact subset match (a model predicting nothing scores 0.375 on val); on binary and multiclass it is dominated by `no_distortion` at 36.9% of rows.
- **Three seeds, mean ± std** (project convention). A gap smaller than the across-seed spread is not a result.
- **The loss is the only thing varying** within a task's ablation — architecture, data and every hyperparameter are held fixed, so a difference is attributable to the loss.
