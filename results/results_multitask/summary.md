# D20 - Multi-task model vs flat vs cascade

**Status: pending Kaggle run.** The multi-task row fills in when `eval_multitask_<seed>.json` files are copied into `results/results_multitask/` and `python -m src.multitask_report` is re-run.

**Multi-task setup (planned).** `mental/mental-roberta-base`, D12 recipe (lr 3e-5, 5 epochs, weight decay 0.1, max length 512, batch 16), seeds 42 / 1337 / 2024, loss weights 1/1/1. One shared encoder, three heads; selection loss masked for No-Distortion and the 13 unlocated rows. Epoch chosen on val by the mean of val binary macro-F1, multilabel macro-F1 and selector F1. Binary threshold: lowest with val FPR <= 0.15; multilabel thresholds per type on val; selector threshold max val sentence F1. Test scored once.

## Test results (mean ± std over 3 seeds, ddof=0)

| Model | Binary specificity | Multilabel macro-F1 (all rows) | Multilabel macro-F1 (distorted rows) | Selector sentence F1 | Source |
| --- | --- | --- | --- | --- | --- |
| **Multi-task (D20)** | pending Kaggle run | pending Kaggle run | pending Kaggle run | pending Kaggle run | `results/results_multitask/eval_multitask_<seed>.json` |
| Flat MentalRoBERTa, separate binary + multilabel models (D12 recipe) | 0.768 ± 0.020 | 0.247 ± 0.016 | n/a | n/a (no selector) | `results/results_thresholds/specificity.csv`; `results/results_mentalroberta_frozensplits/eval_mental-roberta-base_multilabel_<seed>.json` |
| Flat multilabel, RUN 2 Exp 6 | n/a | 0.270 ± 0.013 | n/a | n/a | `results/results_RUN2/experiments.md` (Flat Baseline) |
| Cascade end-to-end, RUN 2 Exp 7 (binary gate -> multilabel) | 0.623 ± 0.059 (a) | 0.229 ± 0.045 | n/a | n/a (no selector) | `results/results_RUN2/results_cascade/eval_cascade_*.json`; stage 1 `results/results_RUN2/results_stage1/` |
| Cascade D17: trained selector -> selected sentences -> multilabel | see (b) | n/a | 0.336 ± 0.020 | 0.421 ± 0.012 | `results/results_cascade_inputs/summary_per_seed.csv` (crop_sel); `results/results_span_selector/eval_test_<seed>.json` |

(a) RUN 2 stage-1 specificity is at the argmax decision (P >= 0.5), **not** a val FPR <= 0.15 threshold, so it is not comparable with the other specificity numbers.
(b) The D17 cascade runs on distorted rows only and has no binary stage; deployed, its gate would be the flat binary model, i.e. the flat row's specificity.

## Where the comparison is not like-for-like

- **Binary specificity.** Multi-task and flat (D13) both use the lowest threshold with val FPR <= 0.15, same splits, same encoder and D12 recipe: comparable. RUN 2 stage 1 uses argmax, 12 epochs, lr 2e-5.
- **Multilabel, all rows vs distorted rows.** Multi-task and both flat rows score all test rows (253, No-Distortion rows have no labels). The D17 cascade scores distorted rows with a located span only (159 test rows), so compare it with the multi-task *distorted rows* column, which is over all distorted test rows (161; the 2 unlocated rows are included there).
- **Training budget.** Multi-task and the D12 flat rows: 5 epochs, lr 3e-5. RUN 2 Exp 6/7: 12 epochs, lr 2e-5. D17: up to 12 epochs with early stopping, deterministic.
- **RUN 2 flat discrepancy.** `experiments.md` Exp 6 (0.2568 / 0.2656 / 0.2886) does not match the `results_RUN2/results_multilabel_flat` eval JSONs (0.2730 / 0.2651 / 0.2494); the table uses `experiments.md` as the brief asks.
- **Selector.** The D03 selector sees one sentence with no context (max length 128); the multi-task selector scores sentences in context from the shared 512-token pass. Both are scored on the same T12 sentence labels with a val-chosen threshold; sentences truncated away in the multi-task pass get score 0 (counted in the eval files).

## Fallback

**If the run is not done by the freeze.** The multi-task model was not trained by the D3 freeze. What was built: `src/train_multitask.py` (one MentalRoBERTa encoder with a binary head, a 10-way multilabel head and a sentence-selection head that scores each sentence from the mean of its token states; the selection loss is masked to zero for No-Distortion rows and the 13 rows whose quote could not be located), its unit tests (`tests/test_multitask.py`, including B-06), the Kaggle runner (`notebooks/kaggle_runner_d20_multitask.ipynb`) and this report script. Chapter 9 therefore reports flat vs cascade from `results/results_RUN2/experiments.md`, and Chapter 10 lists the multi-task model as the first FYP-2 experiment.
