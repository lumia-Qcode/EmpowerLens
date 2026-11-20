# D19 — Justified-concern challenge set: verification and model scores

Pilot set: 50 matched pairs (100 texts), 5 pairs per distortion type, from `pairs_lumia.csv` and `pairs_izza.csv`. Evaluation only; the pairs are not edited.

## 1. Annotator check (blind)

**Pending.** `annotator_sheet.xlsx` (100 shuffled texts, no pair, type or intended label) has not been returned by the psychology annotator yet.

## 2. Leak check

TF-IDF (1-2 grams, sublinear tf) + logistic regression, 5-fold stratified CV, seed 42. F1 of 'distorted' (justified vs distorted) on all 100 texts: **0.184** (limit 0.75): **OK**.

Per file: `pairs_izza.csv` 0.042, `pairs_lumia.csv` 0.196.

Above 0.75 would mean the two halves of each pair differ in wording, not reasoning; those pairs would be rewritten and not scored.

## 3. Model scores (D12 winner, binary, D13 threshold)

**Pending Kaggle run** (`notebooks/kaggle_runner_d19_challenge.ipynb`): one `predictions_<seed>.csv` per seed 42 / 1337 / 2024 in `data/challenge_set/`.

