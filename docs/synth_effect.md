# S08 — Does synthetic data help?

**Setup.**
- **Model:** D12 (MentalRoBERTa, `mental/mental-roberta-base`), with the T14 settings: lr 3e-5, 5 epochs, weight
  decay 0.1, max length 512, batch 16.
- **Two training conditions, 3 seeds each** (42, 1337, 2024):

  | Condition | Training rows |
  |---|---|
  | **Train only** | `data/splits/train.csv` (2,024) |
  | **Train + synthetic** | `data/splits_synth/train.csv`: the same 2,024 + the 232 rows of `synthetic_train_en_final.jsonl` (S10), train only |

- **Same val, same thresholds.** Val is the same 253 rows in both conditions, and every threshold is fitted on it:
  - binary: the D13 rule, lowest threshold with val FPR ≤ 0.15;
  - multilabel: per-type thresholds; predictions are capped at 2 labels.
- **Evaluation:**
  - val and test through `src/evaluate.py` (test is read only there);
  - the **376 S03 supplementary-evaluation rows** (scraped founder writing, 45 distorted / 331 not) through
    `src/eval_supplementary.py`, which uses the same prediction path.
- **Run:** Kaggle T4, notebook `notebooks/kaggle_runner_s08_synth_effect.ipynb`, code commit `6469d82`.
  - No run needed the failed-run re-run.
  - Each run's meta records the splits it trained on.
- **Results:** `results/results_synth_effect/`:
  - `summary.csv`, `per_seed.csv`, `paired.csv`, `supp_per_class.csv`;
  - `raw/` holds every eval JSON and meta file.

## Results (mean ± std over 3 seeds)

| Training | Split | Specificity | FPR | Sensitivity | Multilabel macro-F1 |
|---|---|---|---|---|---|
| Train only | val | 0.853 ± 0.000 | 0.147 ± 0.000 | 0.675 ± 0.026 | 0.300 ± 0.028 |
| Train + synthetic | val | 0.853 ± 0.000 | 0.147 ± 0.000 | 0.709 ± 0.050 | 0.322 ± 0.029 |
| Train only | test | **0.794 ± 0.011** | 0.206 ± 0.011 | 0.648 ± 0.048 | 0.266 ± 0.023 |
| Train + synthetic | test | **0.721 ± 0.033** | 0.279 ± 0.033 | 0.692 ± 0.038 | 0.279 ± 0.024 |
| Train only | supplementary evaluation | **0.941 ± 0.002** | 0.059 ± 0.002 | **0.274 ± 0.034** | 0.119 ± 0.006 |
| Train + synthetic | supplementary evaluation | **0.798 ± 0.059** | 0.202 ± 0.059 | **0.526 ± 0.084** | 0.151 ± 0.003 |

- **Val specificity is 0.853 in every run by construction:** each binary threshold is set to the same val FPR.
- **The train-only rows reproduce the D12/D13 numbers** (test specificity 0.768 ± 0.020, sensitivity 0.667 ± 0.046
  in `docs/thresholds.md`) within seed noise.

### Same seed, train + synthetic minus train only

| Split | Specificity | Sensitivity | Multilabel macro-F1 |
|---|---|---|---|
| val | 0, 0, 0 | 0.000, +0.063, +0.038 | +0.068, −0.009, +0.006 |
| test | **−0.054, −0.098, −0.065** | +0.025, +0.031, +0.075 | +0.014, −0.010, +0.035 |
| supplementary evaluation | **−0.082, −0.148, −0.199** | **+0.200, +0.311, +0.244** | **+0.031, +0.028, +0.036** |

**Binary ROC-AUC** (threshold-free) is also in the evaluate.py JSONs:

| Split | Train only | Train + synthetic | Change |
|---|---|---|---|
| val | 0.829 ± 0.006 | 0.841 ± 0.007 | up |
| test | 0.803 ± 0.004 | 0.790 ± 0.007 | down, by about 2× the seed std |

**Multilabel ROC-AUC:**

| Split | Train only | Train + synthetic |
|---|---|---|
| val | 0.754 | 0.767 |
| test | 0.747 ± 0.004 | 0.754 ± 0.003 |

### Per-type F1 on the supplementary-evaluation rows (mean over 3 seeds)

| Type | Support | Train only | Train + synthetic |
|---|---|---|---|
| Overgeneralization | 15 | 0.137 | 0.190 |
| Fortune-telling | 9 | 0.432 | 0.510 |
| All-or-nothing thinking | 8 | 0.027 | 0.051 |
| Labeling | 8 | 0.020 | 0.073 |
| Mind Reading | 7 | 0.451 | 0.421 |
| Should statements | 7 | 0.011 | 0.088 |
| Magnification | 4 | 0.114 | 0.173 |
| Emotional Reasoning | 0 | — | — |
| Mental filter | 0 | — | — |
| Personalization | 0 | — | — |

Supports are tiny: 45 distorted rows in all, and three types never occur as gold labels. Per-type standard
deviations are in `supp_per_class.csv`; most are about the size of the differences.

## Did synthetic data help?

**It made things worse for the binary decision on the benchmark, and moved the binary decision point on the
scraped rows. For distortion types it made no difference beyond seed noise on the benchmark, and a small but
consistent gain on the scraped rows.**

**Binary decision, benchmark.** Adding the 232 synthetic rows lowered test specificity:
- from 0.794 to 0.721;
- in all three seeds (−0.05 to −0.10), well beyond the seed std;
- test ROC-AUC also fell (0.803 → 0.790), so this isn't just a shifted threshold.

The rise in test sensitivity (+0.04) is smaller than the seed noise.

**Binary decision, scraped rows.** Every synthetic seed flags far more of the scraped rows:
- sensitivity 0.27 → 0.53;
- FPR 0.06 → 0.20.

All three seeds move the same way. This is a different operating point, not a free gain. With 84 No Distortion
rows written by the same generator, the model learns that founder-style setbacks are often distorted, which helps
recall on real founder writing but raises false alarms.

**Multilabel, benchmark.** Macro-F1 rises by +0.02 on val and +0.013 on test, but neither exceeds the seed std, and
one seed goes down on each split. That counts as **no difference beyond seed noise**.

**Multilabel, scraped rows.** Macro-F1 is higher in all three seeds: 0.119 → 0.151, +0.03, with a std of 0.006.
Most of that comes from Fortune-telling, Overgeneralization, Labeling and Should statements. But the scores are low
and the supports tiny, so treat it as a direction, not a size.

**The S07 keyword-leak caveat.** S07 found the synthetic rows over-use give-away words for Personalization,
Overgeneralization and Mental filter.
- Personalization and Mental filter have no gold examples in the scraped evaluation rows, so the leak can't be
  tested there.
- Overgeneralization does gain on the scraped rows (+0.05, support 15), about one seed std. That's not enough to
  tell genuine transfer from "spot the word" learning.

**Practical reading for the project.**
- **Binary model:** keep the train-only model as the app's binary model. Synthetic data cost benchmark specificity,
  and that is the quantity the D13 FPR target protects.
- **Multilabel model:** synthetic data can be considered for the multilabel model only, with the gain on scraped
  text treated as provisional until a larger labelled founder set exists (FYP-2).
- **This is the first result on the scraped evaluation rows** that Chapter 5 promises. Both models find far fewer
  distortions in real founder writing than in the benchmark (sensitivity 0.27 vs 0.65 for train only), a domain gap
  the synthetic rows only partly close.
