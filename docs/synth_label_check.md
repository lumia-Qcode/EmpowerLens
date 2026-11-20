# S10 — Blind human check of flagged synthetic rows

**Labeller:** Izza. She labelled all 217 rows of `data/synthetic/review_sheet.xlsx` blind:
- the sheet shows only a review ID and the text;
- `review_key.csv` was opened only after every row was labelled.

20 rows are marked Unsure.

**Scoring:** `python -m src.synth_label_check --labels data/synthetic/review_sheet.xlsx --labeller Izza`.
- **Rule:** a row is "judged wrong" when her main label differs from the panel's label. The two No Distortion
  options count as the same.
- **Outputs:**
  - `results/results_synth_label_check/precision_table.csv`
  - `results/results_synth_label_check/removed_rows.csv`
  - `results/results_synth_label_check/summary.json`

**Earlier pass.** A disclosed model-assisted labelling pass was done first, as a provisional placeholder:
`data/synthetic/review_labels_model.csv`. It isn't used for any number below. Izza's main labels agree with it on
88% of the 217 rows.

## Precision of each check

| Check | Rows flagged | Rows judged wrong | Precision |
|---|---|---|---|
| Neighbours | 77 | 20 | **0.260** |
| Model (D12) | 52 | 5 | 0.096 |
| Close votes (2-to-1) | 132 | 14 | 0.106 |
| Any check | 187 | 25 | 0.134 |
| **Random rows (baseline error rate)** | 30 | 1 | **0.033** |

**Every check finds wrong labels more often than random reading does.** The baseline is 1 wrong in 30 random rows
(3.3%).

| Check | Precision | Times the random rate |
|---|---|---|
| Neighbours | 0.26 | about 8 |
| Close votes | 0.11 | about 3 |
| Model (D12) | 0.10 | about 3 |

The baseline rests on only 30 rows, so a 95% interval on 1/30 runs up to about 17%. Only neighbours is clearly
above that interval.

**Neighbours is the most useful check here.** That is the reverse of the benchmark warning (ARI 0.003–0.016).
Synthetic rows of one type share wording far more than real rows do (S07), so a row whose neighbours disagree is
more often a genuine mislabel.

**The other two checks.**
- *Close votes* is the costliest: it puts 132 rows on the sheet for 14 errors.
- *Model:* the D12 model, trained on the benchmark only, rarely disagrees in a way that turns out right.

This mirrors D11's route comparison for the benchmark, so Chapter 9 can set the two side by side.

## Rows judged wrong (removed)

**26 rows** are judged wrong and removed from the final file. The full list is in `removed_rows.csv`.

| Panel's label → Izza's main label | Rows |
|---|---|
| Fortune-telling → Magnification | 4 |
| Overgeneralization → Fortune-telling | 4 |
| Labeling → Mind Reading | 3 |
| Magnification → Fortune-telling | 3 |
| Overgeneralization → No Distortion (calibrated pessimism) | 2 |
| Labeling → Fortune-telling / Magnification | 1 each |
| Fortune-telling → Overgeneralization / Mind Reading | 1 each |
| Emotional Reasoning → Magnification | 1 |
| Personalization → No Distortion (calibrated pessimism) / Should statements | 1 each |
| Should statements → Personalization | 1 |
| Mind Reading → Emotional Reasoning / Labeling | 1 each |

- **Most removals sit on one border.** 14 of the 26 involve Fortune-telling and Magnification, in either direction.
  S06's critics struggled with the same border.
- **3 rows were read as sound reasoning:** two Overgeneralization rows and one Personalization row, labelled
  calibrated pessimism.
- **10 of the 26 are swaps of main and second label.** Izza kept the panel's label as the second distortion and
  disagreed only about which one is main. Under the brief's rule they count as wrong.
- **11 of the 26 were marked Unsure.**
- **Removed, not relabelled.** Wrong rows were thrown away, never relabelled, as the brief and D11 require.

## Final training file

`data/synthetic/synthetic_train_en_final.jsonl` = the S07 clean file (258) minus the 26 removed = **232 rows**.
S08 trains on this file.

| Type | Rows |
|---|---|
| No Distortion | 84 |
| Mind Reading | 19 |
| Personalization | 19 |
| Should statements | 18 |
| Labeling | 16 |
| All-or-nothing thinking | 16 |
| Emotional Reasoning | 16 |
| Fortune-telling | 15 |
| Overgeneralization | 14 |
| Mental filter | 11 |
| Magnification | 4 |
