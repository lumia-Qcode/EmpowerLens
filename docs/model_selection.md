# D12 — Model selection

## Rule

Agreed 14 Nov 2026, before applying it. It is the example rule from the D3 task sheet's D12 entry, which was
written before the D09 and D13 results existed.

1. Only **validation** numbers are used. Test is never used to choose.
2. **Primary:** highest mean val specificity at the val threshold meeting the FPR target (≤ 0.15, D13),
   over seeds 42, 1337, 2024.
3. **Tie-break:** if encoders are within 1 std on (2), highest mean val multilabel macro-F1.
4. **Failed runs:** a *binary* run with val macro-F1 below 0.5 (chance level for two balanced-ish classes) is
   treated as a failed training run and re-run once with identical settings; the re-run replaces it. Applied
   to every encoder. It applied once: MentalRoBERTa binary seed 1337 (0.273), re-run → 0.760
   (`docs/thresholds.md`). *(Wording note: the rule as first agreed said "a run", but multilabel macro-F1 is
   0.24–0.33 for every healthy run, so 0.5 only marks failure for binary. This was corrected before applying it.)*

## Validation numbers used (mean ± std over 3 seeds)

| Encoder | Val specificity @ FPR ≤ 0.15 (step 2) | Val multilabel macro-F1 (step 3) |
|---|---|---|
| RoBERTa | 0.853 ± 0.000 | 0.275 ± 0.012 |
| MentalBERT | 0.853 ± 0.000 | 0.262 ± 0.015 |
| **MentalRoBERTa** | 0.853 ± 0.000 | **0.315 ± 0.011** |

Sources: `results/results_thresholds/specificity.csv` (D13); `results/results_mentalroberta_frozensplits/month1_summary_meanstd.csv` (D09).

Step 2 is a tie by construction: D13 fits every threshold to the same val FPR (14 of 95 non-distorted texts
flagged), so val specificity is identical. Step 3 decides.

## Winner

**MentalRoBERTa (`mental/mental-roberta-base`)**: highest val multilabel macro-F1 (0.315 vs 0.275 and 0.262),
ahead of the next encoder by more than 3 std.

## Caveats

- Only MentalRoBERTa used tuned settings (T14: lr 3e-5, 5 epochs, weight decay 0.1); RoBERTa and MentalBERT used
  the defaults (lr 2e-5, 4 epochs). Part of its lead may come from tuning rather than the encoder.
- At the 15% FPR threshold MentalRoBERTa also has the highest val binary sensitivity (0.705 vs 0.690 and 0.667),
  so the two measures agree. This was not part of the rule.

## Checkpoint for D25 and later

No weights from D09 or D13 were kept (checkpoints were deleted after evaluation; weights are never committed).
For D25, retrain the winner once with the same settings and **keep** the weights outside git:

- `checkpoints/binary_mental-roberta-base_42` — then run `python -m src.fit_binary_threshold --checkpoint
  checkpoints/binary_mental-roberta-base_42 --target-fpr 0.15`, because the threshold must be re-fitted for the
  retrained weights.
- `checkpoints/multilabel_mental-roberta-base_42` — per-type thresholds are fitted on val during training.

Seed 42 is used because it is the first seed, not because it scored best; picking the best seed would be
selecting on noise.
