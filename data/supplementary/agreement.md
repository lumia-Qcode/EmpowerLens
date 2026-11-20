# S03 - Agreement on the shared rows

Lumia and Izza each labelled the same 30 rows of `supplementary.csv` separately, before
discussing them (rows chosen by `src/supplementary_labels.py sheets`, seed 42, stratified by role;
IDs in `labelling/assignment.json`). Codebook: `reannotation_2026-09-27/EmpowerLens_Relabelling_Codebook.pdf`.

| Comparison | Raw agreement | Cohen's kappa |
|---|---|---|
| Type (11 labels + calibrated pessimism as its own label) | 76.7% (23/30) | 0.549 |
| Type, calibrated pessimism folded into No Distortion | 86.7% (26/30) | 0.669 |
| Binary (distorted vs not; calibrated pessimism = not distorted) | 93.3% (28/30) | 0.815 |

n = 30 rows. Kappa 0 = no better than chance, 1 = perfect. With only 30 rows the kappa is a
rough estimate; type kappa is unstable when some labels appear once or not at all.

## Label counts on the shared rows

| Label | Lumia | Izza |
|---|---|---|
| All-or-nothing thinking | 0 | 1 |
| Fortune-telling | 1 | 0 |
| Labeling | 2 | 2 |
| Mind Reading | 0 | 1 |
| No Distortion | 20 | 21 |
| No Distortion (calibrated pessimism) | 4 | 1 |
| Overgeneralization | 3 | 2 |
| Should statements | 0 | 2 |

## Rows to settle by discussion (10)

Dominant or secondary differs. Final labels go in `adjudication.csv`.

| item_id | Lumia | Izza |
|---|---|---|
| sup_049 | Overgeneralization | Overgeneralization + All-or-nothing thinking |
| sup_182 | Overgeneralization | Mind Reading |
| sup_206 | No Distortion (calibrated pessimism) | No Distortion |
| sup_207 | No Distortion | Should statements |
| sup_384 | No Distortion | Should statements |
| sup_385 | No Distortion (calibrated pessimism) | No Distortion |
| sup_413 | No Distortion (calibrated pessimism) | No Distortion |
| sup_417 | Fortune-telling + Overgeneralization | All-or-nothing thinking + Overgeneralization |
| sup_454 | Overgeneralization | Overgeneralization + Magnification |
| sup_481 | Labeling | Labeling + Fortune-telling |
