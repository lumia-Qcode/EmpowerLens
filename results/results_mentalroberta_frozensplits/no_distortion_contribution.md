# No-Distortion contribution to weighted-F1 (11-class multiclass)

For each 11-class run, the share of the weighted-F1 attributable to the single `no_distortion` class — the large, easy majority category. A high share means the headline weighted-F1 mostly reflects detecting the *absence* of distortion, not discriminating *between* distortions.

| model | seed | split | no_distortion F1 | weighted_contribution | weighted_f1 | share of weighted_f1 |
|---|---|---|---|---|---|---|
| mental/mental-bert-base-uncased | 42.0 | test | 0.536 | 0.195 | 0.269 | 72.6% |
| mental/mental-bert-base-uncased | 1337.0 | test | 0.552 | 0.201 | 0.276 | 72.8% |
| mental/mental-bert-base-uncased | 2024.0 | test | 0.515 | 0.187 | 0.268 | 70.0% |
| mental/mental-bert-base-uncased | 42.0 | val | 0.604 | 0.227 | 0.339 | 67.0% |
| mental/mental-bert-base-uncased | 1337.0 | val | 0.591 | 0.222 | 0.337 | 65.9% |
| mental/mental-bert-base-uncased | 2024.0 | val | 0.630 | 0.236 | 0.337 | 70.2% |
| mental/mental-roberta-base | 42.0 | test | 0.565 | 0.205 | 0.276 | 74.4% |
| mental/mental-roberta-base | 1337.0 | test | 0.601 | 0.219 | 0.312 | 70.2% |
| mental/mental-roberta-base | 2024.0 | test | 0.577 | 0.210 | 0.320 | 65.6% |
| mental/mental-roberta-base | 42.0 | val | 0.615 | 0.231 | 0.350 | 66.0% |
| mental/mental-roberta-base | 1337.0 | val | 0.608 | 0.228 | 0.338 | 67.6% |
| mental/mental-roberta-base | 2024.0 | val | 0.577 | 0.217 | 0.346 | 62.6% |
| roberta-base | 42.0 | test | 0.565 | 0.205 | 0.281 | 73.2% |
| roberta-base | 1337.0 | test | 0.554 | 0.201 | 0.295 | 68.3% |
| roberta-base | 2024.0 | test | 0.552 | 0.201 | 0.295 | 68.0% |
| roberta-base | 42.0 | val | 0.660 | 0.248 | 0.352 | 70.4% |
| roberta-base | 1337.0 | val | 0.639 | 0.240 | 0.354 | 67.8% |
| roberta-base | 2024.0 | val | 0.568 | 0.213 | 0.330 | 64.7% |

## Mean share on **test** (across seeds)

- **mental/mental-bert-base-uncased**: 71.8% ± 1.3% of the test weighted-F1 comes from `no_distortion` (n=3 seeds).
- **mental/mental-roberta-base**: 70.1% ± 3.6% of the test weighted-F1 comes from `no_distortion` (n=3 seeds).
- **roberta-base**: 69.8% ± 2.4% of the test weighted-F1 comes from `no_distortion` (n=3 seeds).
