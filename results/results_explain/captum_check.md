# D22 — Captum cross-check (val)

> **Internal check only.** Captum scores are never shown to a trainer.

**Status: pending Kaggle run.** Run `notebooks/kaggle_runner_d22_captum.ipynb`, copy its `results_explain/` files into `results/results_explain/`, then `venv\Scripts\python.exe -m src.captum_check report` to fill this page.

## Setup

- Model: D12 winner, MentalRoBERTa multilabel, seed 42 (D12 recipe: lr 3e-5, 5 epochs, weight decay 0.1, max length 512, batch 16).
- Method: LayerIntegratedGradients over `model.roberta.embeddings` (as `src/explain.py`), 50 steps, baseline = `<pad>` tokens with `<s>`/`</s>` kept. Token score = attribution summed over the embedding.
- Label explained: the top-scoring distortion of each row (argmax of the 10 logits).
- Rows: val rows with a distortion (`y_bin == 1`). Test is not used.
- Sentences: the selector's own sentences (`results/results_span_selector/predictions_val_42.csv`), aligned by `Id_Number` + `sent_idx` using character offsets from `results/results_sentence_label/sentence_labels_val.csv`. Sentence score = signed sum of its tokens.

## 1. Agreement with the D03 selector (seed 42)

Agreement: **pending** (X of N rows (Y%)).

## 2. Cue or topic? (20 rows with absolute words)

Selection rule: val rows with `y_bin == 1` whose text contains at least one of `never`, `always`, `everyone`, `nobody`, `everything`, `nothing`, `completely`, `all`, `every`, `totally` (whole word, any case); eligible rows sorted by `Id_Number`, 20 drawn without replacement with `numpy.random.default_rng(42)`, shown sorted by `Id_Number`. Top 5 words = highest signed Captum score towards the attributed label.

Verdict: **cue** = top words are the distortion's wording (absolute words, "should", labels, ...); **topic** = top words are the subject (people, jobs, bodies, ...); **mixed** = both.

| Id_Number | Absolute words | Attributed label | Top 5 words (Captum) | Verdict |
|---|---|---|---|---|
| 131 | never, always, every | pending | pending |  |
| 165 | everyone, nothing, all | pending | pending |  |
| 219 | never, everything | pending | pending |  |
| 334 | never | pending | pending |  |
| 697 | always, everything | pending | pending |  |
| 920 | all | pending | pending |  |
| 1053 | never, everything, nothing, all, every | pending | pending |  |
| 1066 | never, every | pending | pending |  |
| 1172 | always | pending | pending |  |
| 1582 | all | pending | pending |  |
| 1696 | never, always, all | pending | pending |  |
| 1736 | always | pending | pending |  |
| 1756 | never, all | pending | pending |  |
| 1802 | never, always, every | pending | pending |  |
| 1904 | all | pending | pending |  |
| 1932 | never, always, nobody, every | pending | pending |  |
| 2417 | everything | pending | pending |  |
| 2532 | every | pending | pending |  |
| 4522 | always, all | pending | pending |  |
| 4549 | always, all | pending | pending |  |

Verdict counts: cue 0, topic 0, mixed 0, blank 20.

## Summary

<!-- summary:start -->
_To be written by the reviewer after filling the verdicts._
<!-- summary:end -->

## Files

`attributions_val.csv` (per sentence), `attributions_val_words.csv` (per word), `attributions_val_rows.csv` (per row: label, probability, top words), `attributions_val_config.json`, `captum_check_rows.csv` (the 20-row table). Fill the verdicts in that CSV or in the table above, and the summary between the markers; re-running `report` keeps both (the CSV wins if they differ).
