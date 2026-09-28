# No-Distortion contribution to weighted-F1 (11-class multiclass)

For each 11-class run, the share of the weighted-F1 attributable to the single `no_distortion` class — the large, easy majority category. A high share means the headline weighted-F1 mostly reflects detecting the *absence* of distortion, not discriminating *between* distortions.

| model | seed | split | no_distortion F1 | weighted_contribution | weighted_f1 | share of weighted_f1 |
|---|---|---|---|---|---|---|
| mental/mental-roberta-base | 42.0 | test | 0.596 | 0.217 | 0.345 | 62.8% |
| mental/mental-roberta-base | 1337.0 | test | 0.606 | 0.221 | 0.349 | 63.1% |
| mental/mental-roberta-base | 2024.0 | test | 0.543 | 0.197 | 0.320 | 61.6% |
| mental/mental-roberta-base | 42.0 | val | 0.619 | 0.232 | 0.381 | 61.1% |
| mental/mental-roberta-base | 1337.0 | val | 0.663 | 0.249 | 0.396 | 62.8% |
| mental/mental-roberta-base | 2024.0 | val | 0.575 | 0.216 | 0.343 | 62.9% |
| microsoft/deberta-v3-base | 42.0 | test | 0.634 | 0.231 | 0.344 | 67.1% |
| microsoft/deberta-v3-base | 1337.0 | test | 0.610 | 0.222 | 0.344 | 64.5% |
| microsoft/deberta-v3-base | 2024.0 | test | 0.577 | 0.210 | 0.321 | 65.3% |
| microsoft/deberta-v3-base | 42.0 | val | 0.652 | 0.245 | 0.413 | 59.3% |
| microsoft/deberta-v3-base | 1337.0 | val | 0.684 | 0.257 | 0.392 | 65.6% |
| microsoft/deberta-v3-base | 2024.0 | val | 0.635 | 0.238 | 0.365 | 65.2% |

## Mean share on **test** (across seeds)

- **mental/mental-roberta-base**: 62.5% ± 0.7% of the test weighted-F1 comes from `no_distortion` (n=3 seeds).
- **microsoft/deberta-v3-base**: 65.7% ± 1.1% of the test weighted-F1 comes from `no_distortion` (n=3 seeds).
