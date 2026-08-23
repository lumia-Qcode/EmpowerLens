# No-Distortion contribution to weighted-F1 (11-class multiclass)

For each 11-class run, the share of the weighted-F1 attributable to the single `no_distortion` class — the large, easy majority category. A high share means the headline weighted-F1 mostly reflects detecting the *absence* of distortion, not discriminating *between* distortions.

| model | seed | split | no_distortion F1 | weighted_contribution | weighted_f1 | share of weighted_f1 |
|---|---|---|---|---|---|---|
| roberta-base | 42.0 | test | 0.565 | 0.205 | 0.281 | 73.2% |
| roberta-base | 1337.0 | test | 0.554 | 0.201 | 0.295 | 68.3% |
| roberta-base | 2024.0 | test | 0.552 | 0.201 | 0.295 | 68.0% |
| roberta-base | 42.0 | val | 0.660 | 0.248 | 0.352 | 70.4% |
| roberta-base | 1337.0 | val | 0.639 | 0.240 | 0.354 | 67.8% |
| roberta-base | 2024.0 | val | 0.568 | 0.213 | 0.330 | 64.7% |

## Mean share on **test** (across seeds)

- **roberta-base**: 69.8% ± 2.4% of the test weighted-F1 comes from `no_distortion` (n=3 seeds).
