# D13 — Decision thresholds at a stated FPR target

## Target

**Binary: false-positive rate ≤ 0.15 on validation**, i.e. at most 15% of non-distorted texts flagged.
Set 13 Nov 2026, before any threshold results were seen (an earlier draft target of 0.20 was replaced by 0.15
the same day, also before any results). The threshold is chosen on val and applied unchanged to test.

Rule (`src/fit_binary_threshold.py`): the lowest threshold on P(distorted) whose val FPR is ≤ 0.15, which keeps
as much sensitivity as the false-positive budget allows. `src/evaluate.py` applies it to val and test; test is
read only there.

Multilabel: one threshold per distortion type, each chosen on val for that type's F1
(`sweep_thresholds` in `src/train_transformer.py`), from the D09 / earlier encoder runs.

## Setup

- Splits: `data/splits` (train 2,024 / val 253 / test 253). Seeds 42, 1337, 2024.
- Binary models retrained for D13 with each encoder's original settings: RoBERTa and MentalBERT lr 2e-5,
  4 epochs; MentalRoBERTa lr 3e-5, 5 epochs, weight decay 0.1 (T14). Max length 512, batch 16, one T4 GPU.
- Code: `izza-space` commit `e130ddb`. Run on Kaggle (notebook "D13 binary thresholds FPR 0.15", Version 3).
- **MentalRoBERTa seed 1337 collapsed** in the Kaggle run (val macro-F1 0.273 vs 0.72–0.78 for every other run;
  val sensitivity 0.076). It was re-run once with identical code and settings on Google Colab (T4), giving val
  macro-F1 0.760. The re-run replaces it below; the collapsed run is not used.

## Binary results (mean ± std over 3 seeds)

| Encoder | Split | Specificity | FPR | Sensitivity |
|---|---|---|---|---|
| RoBERTa | val | 0.853 ± 0.000 | 0.147 ± 0.000 | 0.690 ± 0.005 |
| RoBERTa | test | 0.761 ± 0.000 | 0.239 ± 0.000 | 0.663 ± 0.054 |
| MentalBERT | val | 0.853 ± 0.000 | 0.147 ± 0.000 | 0.667 ± 0.011 |
| MentalBERT | test | 0.804 ± 0.023 | 0.196 ± 0.023 | 0.642 ± 0.020 |
| MentalRoBERTa | val | 0.853 ± 0.000 | 0.147 ± 0.000 | 0.705 ± 0.013 |
| MentalRoBERTa | test | 0.768 ± 0.020 | 0.232 ± 0.020 | 0.667 ± 0.046 |

Val: 95 non-distorted, 158 distorted. Test: 92 non-distorted, 161 distorted. Per-run counts (TN, FP) are in
`results/results_thresholds/specificity.csv`.

- Val specificity is 0.853 for every run **by construction**: each threshold is set to the same val FPR
  (14 of 95 flagged). Val sensitivity is what differs between encoders at this operating point.
- **Test FPR is above the target** for all encoders (0.196–0.239). Val has only 95 non-distorted texts, so a
  threshold fitted on it does not transfer exactly.

## Binary thresholds

| Encoder | Seed 42 | Seed 1337 | Seed 2024 |
|---|---|---|---|
| RoBERTa | 0.697 | 0.672 | 0.581 |
| MentalBERT | 0.710 | 0.885 | 0.787 |
| MentalRoBERTa | 0.509 | 0.493 | 0.694 |

## Multilabel thresholds (val-F1 per type)

Order: emotional reasoning, overgeneralization, mental filter, should statements, all-or-nothing, mind reading,
fortune telling, magnification, personalization, labeling.

| Encoder | Seed | Thresholds |
|---|---|---|
| RoBERTa | 42 | 0.70, 0.50, 0.70, 0.60, 0.70, 0.70, 0.75, 0.60, 0.65, 0.45 |
| RoBERTa | 1337 | 0.70, 0.45, 0.70, 0.70, 0.75, 0.70, 0.70, 0.55, 0.65, 0.65 |
| RoBERTa | 2024 | 0.75, 0.50, 0.65, 0.60, 0.70, 0.50, 0.65, 0.60, 0.65, 0.40 |
| MentalBERT | 42 | 0.50, 0.45, 0.45, 0.50, 0.65, 0.65, 0.55, 0.50, 0.50, 0.60 |
| MentalBERT | 1337 | 0.60, 0.50, 0.50, 0.65, 0.55, 0.70, 0.55, 0.50, 0.45, 0.60 |
| MentalBERT | 2024 | 0.70, 0.50, 0.50, 0.65, 0.65, 0.50, 0.65, 0.60, 0.55, 0.60 |
| MentalRoBERTa | 42 | 0.70, 0.40, 0.35, 0.55, 0.80, 0.60, 0.85, 0.55, 0.65, 0.65 |
| MentalRoBERTa | 1337 | 0.75, 0.50, 0.45, 0.85, 0.70, 0.60, 0.75, 0.80, 0.65, 0.60 |
| MentalRoBERTa | 2024 | 0.80, 0.55, 0.40, 0.70, 0.85, 0.60, 0.80, 0.60, 0.75, 0.50 |

Sources: `results/results_experiments/` (RoBERTa, MentalBERT) and `results/results_mentalroberta_frozensplits/`
(MentalRoBERTa). Per-type specificity for these multilabel runs was not recorded (they predate it in
`src/evaluate.py`) and would need retraining.

## Files

- `results/results_thresholds/`: eval JSON, per-class CSV and confusion PNG for each of the 9 binary runs,
  `paper_comparison.csv`, and `specificity.csv` (from `python -m src.specificity --results results/results_thresholds`).
