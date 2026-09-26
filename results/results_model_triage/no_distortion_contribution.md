# No-Distortion contribution to weighted-F1 (11-class multiclass)

For each 11-class run, the share of the weighted-F1 attributable to the single `no_distortion` class — the large, easy majority category. A high share means the headline weighted-F1 mostly reflects detecting the *absence* of distortion, not discriminating *between* distortions.

| model | seed | split | no_distortion F1 | weighted_contribution | weighted_f1 | share of weighted_f1 |
|---|---|---|---|---|---|---|
| mental/mental-roberta-base | 42.0 | test | 0.780 | 0.265 | 0.649 | 40.8% |
| mental/mental-roberta-base | 1337.0 | test | 0.742 | 0.252 | 0.623 | 40.5% |
| mental/mental-roberta-base | 2024.0 | test | 0.759 | 0.258 | 0.616 | 41.9% |
| mental/mental-roberta-base | 42.0 | val | 0.792 | 0.269 | 0.662 | 40.7% |
| mental/mental-roberta-base | 1337.0 | val | 0.766 | 0.260 | 0.662 | 39.4% |
| mental/mental-roberta-base | 2024.0 | val | 0.705 | 0.240 | 0.600 | 40.0% |

## Mean share on **test** (across seeds)

- **mental/mental-roberta-base**: 41.1% ± 0.6% of the test weighted-F1 comes from `no_distortion` (n=3 seeds).
