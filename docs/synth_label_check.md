# S10 — Label check of flagged synthetic rows

> **Status: PROVISIONAL.** These numbers come from a **disclosed model-assisted labelling pass**, not from the S10
> human check the brief asks for. **Izza** still has to label every row of `data/synthetic/review_sheet.xlsx` blind
> and re-run the command in the last section. That replaces every number below. Until then, don't report this file
> as a human check.

**Labels:** `data/synthetic/review_labels_model.csv`, 217 rows. Each row records `labelled_by = "model-assisted
(provisional; to be confirmed by Izza)"`.

**How the labels were made:**
- the reader used only the review sheet's ID and text, plus the codebook guide;
- `review_key.csv` was not read while labelling.

**The pass was not fully blind.** The same model had already read some of these synthetic texts and their panel
labels during the S05/S06 write-ups (`docs/synth_calibration.md`, `docs/synth_run.md`): all Magnification and
Mental filter rows and a few others. That can bias agreement upwards on those rows. Izza's pass doesn't have this
problem.

**Scoring:** `python -m src.synth_label_check --labels data/synthetic/review_labels_model.csv --labeller "model-assisted (provisional; to be confirmed by Izza)"`.
- **Rule:** a row is "judged wrong" when the reviewer's main label differs from the panel's label. The two No
  Distortion options count as the same.
- **Outputs:**
  - `results/results_synth_label_check/precision_table.csv`
  - `results/results_synth_label_check/removed_rows.csv`
  - `results/results_synth_label_check/summary.json`
  - `data/synthetic/synthetic_train_en_final.jsonl`

## Precision of each check (provisional)

| Check | Rows flagged | Rows judged wrong | Precision |
|---|---|---|---|
| Neighbours | 77 | 13 | **0.169** |
| Model (D12) | 52 | 1 | 0.019 |
| Close votes (2-to-1) | 132 | 10 | 0.076 |
| Any check | 187 | 15 | 0.080 |
| **Random rows (baseline)** | 30 | 0 | **0.000** |

The random rows' error rate is 0 of 30, so the baseline is 0. A 95% upper bound for 0/30 is about 10%. On this
pass:
- **Neighbours** has the best precision (0.17). That is the reverse of the benchmark warning, probably because
  synthetic neighbourhoods are tighter than real ones.
- **Close votes** is weaker (0.08), even though it is the largest list.
- **Model** almost never catches a wrong row (1 of 52). The D12 model's disagreements are mostly with the panel's
  choice of main versus second label. The reader, like the panel, accepted the panel's main label on those rows.

**With a baseline of 0 of 30, every check beats random.** But the sample is small, and only S10's human labels can
settle it.

## Rows judged wrong (removed)

**15 rows** are judged wrong and removed from the final file. Grouped by what the panel said and what the reader
said:

| Panel's label → reader's main label | Rows |
|---|---|
| Labeling → Mind Reading | 3 |
| Magnification → Fortune-telling | 3 |
| Labeling → Fortune-telling | 2 |
| Overgeneralization → Fortune-telling | 2 |
| Fortune-telling → Mind Reading / Overgeneralization / Magnification | 1 each |
| Mind Reading → Labeling | 1 |
| Personalization → Should statements | 1 |

**11 of the 15 are swaps of main and second label, not wrong distortions.** The reader kept the panel's label as
the second distortion and disagreed only about which one is main. The brief's rule judges main labels, so these
count as wrong here. When Izza does the human pass, the team should decide whether a main/second swap is "wrongly
labelled" or "acceptable". If it's acceptable, only **4 rows** are clearly wrong:

| Row | Panel's label | Reader's label |
|---|---|---|
| syn_00041 | Labeling | Fortune-telling (marked Unsure) |
| syn_00048 | Labeling | Mind Reading |
| syn_00269 | Personalization | Should statements |
| syn_00013 | Labeling | Fortune-telling, second Overgeneralization |

The list is in `results/results_synth_label_check/removed_rows.csv`.

**The Magnification ↔ Fortune-telling border** again accounts for the most removals. S06 already showed it for the
critic panel. The final file has only **4 Magnification rows**.

**Rows were removed, never relabelled.** This mirrors D11.

## Final training file (provisional)

`data/synthetic/synthetic_train_en_final.jsonl` = the S07 clean file (258) minus the 15 removed = **243 rows**.

| Type | Rows |
|---|---|
| No Distortion | 84 |
| Mind Reading | 20 |
| Personalization | 20 |
| Should statements | 19 |
| Fortune-telling | 18 |
| Overgeneralization | 18 |
| Emotional Reasoning | 17 |
| Labeling | 16 |
| All-or-nothing thinking | 16 |
| Mental filter | 11 |
| Magnification | 4 |

## How to finish S10 with the human labels

1. **Izza fills the "Label" tab of `data/synthetic/review_sheet.xlsx`** (all 217 rows, until Status = Done). She
   doesn't open `review_key.csv` or `review_labels_model.csv` until she's finished.
2. **Re-run** with her sheet:
   ```
   python -m src.synth_label_check --labels data/synthetic/review_sheet.xlsx --labeller "Izza"
   ```
   This overwrites the precision table, the removed list and `synthetic_train_en_final.jsonl`.
3. **Update this doc** with her numbers and remove the provisional banner.
   - Optional: report how often her main labels agree with the provisional pass, as a measure of how far the pass
     can be trusted.
4. **Only then** S08 trains on the final file.
