# D16 — Original vs relabelled labels

## What was run

- Model: MentalRoBERTa (`mental/mental-roberta-base`, D12 winner), T14 settings (lr 3e-5, 5 epochs,
  weight decay 0.1, max length 512). Seeds 42, 1337, 2024.
- Two training sets, same rows: `data/splits` (original labels) and `data/splits_relabelled` (D14).
- Each model is evaluated on both test sets. Binary threshold fitted on the model's own val split at
  FPR ≤ 0.15 (D13 rule); multilabel per-type thresholds fitted on val.
- Code: `izza-space` commit `7286d6d`, Kaggle notebook "D16_notebook" Version 2 (T4).
- All 12 training runs passed the failed-run check (binary val macro-F1 0.70–0.78).

Relabelling changes the class balance (`data/splits_relabelled/manifest.json`): on test, 61 of 253 rows changed
binary label (both directions), so non-distorted goes from 92 to 127 and distorted from 161 to 126. Every test
row was relabelled by one annotator (`label_changes_summary.md`).

These results are not comparable to the paper's published figures and are reported separately from them (Ch5).

## Test results (mean ± std over 3 seeds)

| Trained on | Tested on | Specificity | FPR | Sensitivity | Binary macro-F1 | Multilabel macro-F1 |
|---|---|---|---|---|---|---|
| Original | Original | 0.790 ± 0.020 | 0.210 ± 0.020 | 0.561 ± 0.033 | 0.642 ± 0.027 | 0.259 ± 0.014 |
| Original | Relabelled | 0.722 ± 0.004 | 0.278 ± 0.004 | 0.590 ± 0.033 | 0.654 ± 0.019 | 0.228 ± 0.006 |
| Relabelled | Original | 0.880 ± 0.023 | 0.120 ± 0.023 | 0.511 ± 0.084 | 0.644 ± 0.046 | 0.172 ± 0.017 |
| Relabelled | Relabelled | 0.787 ± 0.032 | 0.213 ± 0.032 | 0.526 ± 0.092 | 0.649 ± 0.037 | 0.182 ± 0.009 |

## What it shows

- **On the relabelled test, the relabelled model flags fewer non-distorted texts:** specificity 0.787 vs 0.722
  (FPR 0.213 vs 0.278). It catches fewer distorted texts: sensitivity 0.526 vs 0.590. Binary macro-F1 is about
  the same (0.649 vs 0.654).
- **The original-label model loses specificity on the relabelled test** (0.790 → 0.722): texts that relabelling
  moved to non-distorted are still flagged by it.
- **Multilabel macro-F1 is lower when training on relabelled labels** (0.182 vs 0.228 on the relabelled test).
- Test FPR is above the 0.15 target in three of four cells, as in D13 (val has few non-distorted texts).

## Caveats

- Relabelled seed 2024 binary has low sensitivity (val 0.473, test 0.405 vs 0.55–0.71 for the others). It passes
  the failed-run rule (val macro-F1 0.756), so it is kept; it drives the larger std in the relabelled rows.
- The original-label binary runs here are retrains of D13's with the same code (`src/` unchanged between commits
  `e130ddb` and `7286d6d`), but training is not deterministic on GPU, and their test sensitivity is lower (0.561 vs
  0.667 in `docs/thresholds.md`). Compare cells within this table.
- Most runs peaked at the last epoch (5 of 5), so a longer budget might help both label sets.

## Files

`results/results_relabelled/train_{orig|relab}__test_{orig|relab}/`: eval JSON, per-class CSV, confusion PNG,
`paper_comparison.csv` and `specificity.csv` per cell. 24 eval files: 12 for the relabelled-label model
(2 tasks × 3 seeds × 2 test versions) and 12 for the original-label model it is compared with.
