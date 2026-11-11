# D11 - Which list finds wrong labels best?

Three ranked lists are scored against the re-annotation verdicts: **neighbours** (stage 2: a row whose 10 nearest texts mostly carry another label ranks high), **model triage** (the A and B buckets of the re-annotation session: A before B, and inside a bucket the row whose label the model trusts least first), and **random** order (seed 42), the floor. Rows marked *extra* are the model's raw scores, for reference.

**Verdicts** use the same rule as D08 (`label_changes.py`): a vote counts only if Primary is filled, it was not filled in from the original label (81 CS-majors blanks), and Unsure is not ticked (never adjudicated); a row needs one 11-way label (calibrated pessimism = No Distortion) with more than half its valid votes. A row is a **confirmed error** when that label differs from the original dominant label. 1322 rows have a verdict; 1035 of them changed.

**Scores.** Average precision (AP) is one number for the whole list, high when real errors sit near the top. Precision@k = share of the first k rows that were real errors. Lift@100 = precision@100 / base rate (what random order gets on average). **Confirmed errors per hour** = precision@100 x 80 rows per hour, using the **45 s per row estimate** (no times were recorded). Tied scores are ordered by a fixed random draw.

## Table 1 - all three lists, on the rows each list can score

946 rows (buckets: {'A': 877, 'B': 69}). The neighbour queue covers distorted **train** rows only, and the only train rows that were re-annotated are ones the model had flagged, so **every row here is already in the model's A/B buckets** and the base rate is high (86.1%). Precision near the base rate means a list adds nothing; read lift and AP against random.

| list                              |     AP |   precision@50 |   precision@100 |   precision@200 |   lift@100 |   errors_per_hour@100 |   rows_scored |
|:----------------------------------|-------:|---------------:|----------------:|----------------:|-----------:|----------------------:|--------------:|
| neighbours (clustering)           | 0.8729 |           0.84 |            0.88 |           0.87  |       1.02 |                  70.4 |           946 |
| model triage (A+B buckets)        | 0.8878 |           0.94 |            0.91 |           0.885 |       1.06 |                  72.8 |           946 |
| random (seed 42)                  | 0.8432 |           0.84 |            0.84 |           0.835 |       0.98 |                  67.2 |           946 |
| extra: model self-confidence only | 0.8876 |           0.94 |            0.91 |           0.885 |       1.06 |                  72.8 |           946 |
| extra: model cleanlab quality     | 0.8775 |           0.88 |            0.88 |           0.895 |       1.02 |                  70.4 |           946 |
| extra: model entropy              | 0.8322 |           0.82 |            0.84 |           0.795 |       0.98 |                  67.2 |           946 |

## Table 2 - model triage vs random on the test rows

184 test rows with a verdict (base rate 54.9%): flagged changed 72 of 97, unflagged changed 29 of 87. This is the only part of the re-annotation that included rows the model did **not** flag, so it is the only fair test of the A/B selection itself. Neighbours cannot be scored here (its queue has no test rows).

| list                              |     AP |   precision@50 |   precision@100 | precision@200   |   lift@100 |   errors_per_hour@100 |   rows_scored |
|:----------------------------------|-------:|---------------:|----------------:|:----------------|-----------:|----------------------:|--------------:|
| model triage (A+B buckets)        | 0.8198 |           0.82 |            0.74 | n/a             |       1.35 |                  59.2 |           184 |
| random (seed 42)                  | 0.6125 |           0.6  |            0.59 | n/a             |       1.07 |                  47.2 |           184 |
| extra: model self-confidence only | 0.8246 |           0.82 |            0.77 | n/a             |       1.4  |                  61.6 |           184 |
| extra: model cleanlab quality     | 0.818  |           0.84 |            0.77 | n/a             |       1.4  |                  61.6 |           184 |
| extra: model entropy              | 0.6313 |           0.64 |            0.64 | n/a             |       1.17 |                  51.2 |           184 |

precision@200 is n/a on this table: it has fewer than 200 rows.

## Answer

On the shared rows the best of the three lists by AP is **model triage (A+B buckets)**: neighbours AP 0.873, model triage 0.888, random 0.843. On the test rows model triage scores AP 0.820 vs random 0.613 (precision@50 0.82 vs 0.60). Stage 1 shows why clustering struggles: clusters follow topic, not distortion type (ARI -0.011 to 0.027 across settings, `cluster_recovery.csv`).

**Caveat.** Ground truth exists only for re-annotated rows, and the model chose most of them, so Table 1 compares the lists *inside the model's own selection*. Table 2 is small and has one annotator per row. Neither table is an unbiased estimate over all 2,530 rows.

Files: `route_comparison.csv` (Table 1), `route_comparison_test.csv` (Table 2), `route_comparison_meta.json`. Script: `python -m src.cluster_label_errors --only compare`.
