# D19 — Justified-concern challenge set: verification and model scores

Pilot set: 50 matched pairs (100 texts), 5 pairs per distortion type, from `pairs_lumia.csv` and `pairs_izza.csv`. Evaluation only; the pairs are not edited.

## 1. Annotator check (blind)

Agreement with the intended labels: **100 of 100** labelled texts (0 left blank).

Pairs with at least one disagreement: none.

## 2. Leak check

TF-IDF (1-2 grams, sublinear tf) + logistic regression, 5-fold stratified CV, seed 42. F1 of 'distorted' (justified vs distorted) on all 100 texts: **0.184** (limit 0.75): **OK**.

Per file: `pairs_izza.csv` 0.042, `pairs_lumia.csv` 0.196.

Above 0.75 would mean the two halves of each pair differ in wording, not reasoning; those pairs would be rewritten and not scored.

## 3. Model scores (D12 winner, binary, D13 threshold)

Model: `mental/mental-roberta-base`; seeds 42, 1337, 2024. Specificity = justified texts left unflagged; sensitivity = distorted texts flagged.

| | seed 42 | seed 1337 | seed 2024 | mean ± std |
|---|---|---|---|---|
| Specificity (justified) | 45/50 | 49/50 | 50/50 | 0.960 ± 0.043 |
| FPR | 0.100 | 0.020 | 0.000 | 0.040 ± 0.043 |
| Sensitivity (distorted) | 14/50 | 5/50 | 1/50 | 0.133 ± 0.109 |
| Threshold on P(distorted) | 0.8560 | 0.6609 | 0.8062 | |

### Per distortion type (5 justified + 5 distorted texts each)

| Type | spec s42 | spec s1337 | spec s2024 | spec mean ± std | sens s42 | sens s1337 | sens s2024 | sens mean ± std |
|---|---|---|---|---|---|---|---|---|
| All-or-nothing thinking | 4/5 | 5/5 | 5/5 | 0.933 ± 0.094 | 1/5 | 1/5 | 0/5 | 0.133 ± 0.094 |
| Emotional Reasoning | 2/5 | 4/5 | 5/5 | 0.733 ± 0.249 | 3/5 | 1/5 | 0/5 | 0.267 ± 0.249 |
| Fortune-telling | 5/5 | 5/5 | 5/5 | 1.000 ± 0.000 | 0/5 | 0/5 | 0/5 | 0.000 ± 0.000 |
| Labeling | 5/5 | 5/5 | 5/5 | 1.000 ± 0.000 | 5/5 | 2/5 | 0/5 | 0.467 ± 0.411 |
| Magnification | 5/5 | 5/5 | 5/5 | 1.000 ± 0.000 | 0/5 | 0/5 | 0/5 | 0.000 ± 0.000 |
| Mental filter | 4/5 | 5/5 | 5/5 | 0.933 ± 0.094 | 2/5 | 0/5 | 0/5 | 0.133 ± 0.189 |
| Mind Reading | 5/5 | 5/5 | 5/5 | 1.000 ± 0.000 | 0/5 | 0/5 | 0/5 | 0.000 ± 0.000 |
| Overgeneralization | 5/5 | 5/5 | 5/5 | 1.000 ± 0.000 | 1/5 | 0/5 | 0/5 | 0.067 ± 0.094 |
| Personalization | 5/5 | 5/5 | 5/5 | 1.000 ± 0.000 | 0/5 | 0/5 | 0/5 | 0.000 ± 0.000 |
| Should statements | 5/5 | 5/5 | 5/5 | 1.000 ± 0.000 | 2/5 | 1/5 | 1/5 | 0.267 ± 0.094 |

With 5 texts per cell, one text moves a rate by 0.2; read per-type differences as indicative only.

