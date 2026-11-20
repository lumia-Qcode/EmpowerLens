# D17 - Cascade inputs and the gap to the human answer

**Question.** Does pointing the classifier at the distorted part help, and how much of that gain survives when a trained selector, not a human, chooses the sentences?

**Setup.** MentalRoBERTa (`mental/mental-roberta-base`), multilabel over the 10 distortions, on the **distorted rows** of the frozen split, minus the 13 rows whose quoted span could not be located (train 1,268 / val 157 / test 159; the same rows in every version). Recipe: max length 512, batch 16, up to 12 epochs with early stopping (patience 3), `--deterministic`, seeds 42 / 1337 / 2024; per-label thresholds tuned on val at training time. Inputs are built by `src/make_splits_cascade_inputs.py` (`data/splits_cascade_inputs/`).

- **Selected sentences only**: the chosen sentences, in text order. **Marked**: the full text with each run of chosen sentences wrapped in `@ ... @`.
- **Gold span** = the sentences the human annotator quoted (T12 sentence labels). **Trained selector** = the D03 selector's picks for the same seed (rows where it picked nothing fall back to its top-scoring sentence; counts in `data/splits_cascade_inputs/manifest.json`).
- Each span classifier is **trained once on gold sentences** and evaluated twice: with gold and with selector sentences (the real cascade at use time). The selector rows reuse the thresholds tuned on gold-input val.

## Results (mean ± std over 3 seeds, std with n-1)

| Input | Val macro-F1 | Test macro-F1 | Test specificity (per-label) |
| --- | --- | --- | --- |
| full text | 0.381 ± 0.012 | 0.268 ± 0.020 | 0.843 ± 0.004 |
| selected sentences only - trained selector | 0.368 ± 0.015 | 0.336 ± 0.025 | 0.869 ± 0.016 |
| selected sentences only - gold span (oracle) | 0.519 ± 0.022 | 0.434 ± 0.016 | 0.891 ± 0.012 |
| full text, sentences marked - trained selector | 0.348 ± 0.018 | 0.269 ± 0.027 | 0.844 ± 0.004 |
| full text, sentences marked - gold span (oracle) | 0.376 ± 0.006 | 0.305 ± 0.022 | 0.849 ± 0.003 |

Macro-F1 is over the 10 distortion labels. **Specificity here is per-label**: for each label, the share of rows without that label that the model leaves unflagged, averaged over the 10 labels. It is *not* the binary headline specificity: this set has no No-Distortion rows, because the span versions only exist for distorted texts.

## Cost of automatic selection (gold span - trained selector, paired by seed)

| Version | Val macro-F1 | Test macro-F1 |
| --- | --- | --- |
| selected sentences only | +0.152 ± 0.010 | +0.098 ± 0.009 |
| marked full text | +0.028 ± 0.017 | +0.036 ± 0.011 |

Best non-oracle input on test: **selected sentences only - trained selector** (0.336 vs full text 0.268).

**Reference (different model).** The earlier oracle run with TF-IDF + logistic regression gave val macro-F1 0.402 on the gold span vs 0.322 on full text (`docs/FYP_PLAN.md` section 3.1, distorted-only, 10 labels). It is a ceiling from a different model, not a row of this table.

## Files

Every number above comes from the eval files in this folder (written by `src/evaluate.py`):

- `full/eval_mental-roberta-base_multilabel_42.json`
- `full/eval_mental-roberta-base_multilabel_1337.json`
- `full/eval_mental-roberta-base_multilabel_2024.json`
- `crop_sel/eval_mental-roberta-base_multilabel_42.json`
- `crop_sel/eval_mental-roberta-base_multilabel_1337.json`
- `crop_sel/eval_mental-roberta-base_multilabel_2024.json`
- `crop_gold/eval_mental-roberta-base_multilabel_42.json`
- `crop_gold/eval_mental-roberta-base_multilabel_1337.json`
- `crop_gold/eval_mental-roberta-base_multilabel_2024.json`
- `marked_sel/eval_mental-roberta-base_multilabel_42.json`
- `marked_sel/eval_mental-roberta-base_multilabel_1337.json`
- `marked_sel/eval_mental-roberta-base_multilabel_2024.json`
- `marked_gold/eval_mental-roberta-base_multilabel_42.json`
- `marked_gold/eval_mental-roberta-base_multilabel_1337.json`
- `marked_gold/eval_mental-roberta-base_multilabel_2024.json`

`summary.csv` (table), `summary_per_seed.csv` (one row per input x seed). Script: `python -m src.aggregate_cascade_inputs`.
