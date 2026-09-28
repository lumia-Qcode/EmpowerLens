# Task: Build relabelling files from model-triage output (TherapistQA)

Hand this file to Claude Code in the EmpowerLens repo. Work on a new branch off `main`
(`git checkout main && git pull && git checkout -b relabel-split`). Do not modify input files.

Tasks 1–3 run **before** the labelling session. Task 4 runs **after** it.

## Inputs (confirm paths before running)

| Name | Expected contents | Placeholder path |
| --- | --- | --- |
| Triage file | 1,370 flagged rows (disagree + ambiguous buckets), all triage columns: original labels, bucket, model predictions, confidence scores, etc. | `data/triage/flagged_rows.xlsx` |
| Original dataset | Full TherapistQA, 2,531 rows | `data/raw/therapistqa.csv` |
| Test split | The fixed 80/20 test split used in training (IDs or full rows) | `data/splits/test.csv` |

Expected TherapistQA columns: `Id_Number`, `Patient Question`, `Distorted part`,
`Dominant Distortion`, `Secondary Distortion (Optional)`. Check the real headers first and
adapt; stop and report if the ID column can't be matched across all three files.

Parameters (top of the script):

```python
SEED = 42
N_ANNOTATORS = 4            # set to 6 if all six annotators label
N_OVERLAP = 60              # shared rows for agreement metrics
SHOW_DISTORTED_PART = True  # False hides it (it can hint at the original label)
CP_LABEL = "No Distortion (calibrated pessimism)"
ANNOTATOR_DISCIPLINE = {1: "psych", 2: "psych", 3: "cs", 4: "cs"}  # fill in real mapping
```

Write every count and ID list to `outputs/relabel/manifest.json` so the split is reproducible.

## Task 1: Test sheet (all test rows, relabelled separately)

Output: `outputs/relabel/test_rows_all.xlsx`

1. Take every ID in the test split.
2. For test IDs present in the triage file: copy the row with **all** triage columns.
3. For test IDs **not** in the triage file (the agree bucket): pull the row from the original
   dataset, set `bucket = "agree (not flagged)"`, leave triage-only columns blank.
4. Add a column `in_triage_file` (TRUE/FALSE).

Discrepancy report (`outputs/relabel/test_discrepancies.md`), with counts and ID lists:
- Test IDs found in the triage file.
- Test IDs not in the triage file but found in the original dataset (expected: agree bucket).
- Test IDs missing from the original dataset entirely (**error**).
- Duplicate IDs in the test split or triage file (**error**).
- Rows where original labels differ between triage file and original dataset (**error**).

## Task 2: Annotator workbooks

Outputs: `outputs/relabel/annotator_1.xlsx` … `annotator_{N_ANNOTATORS}.xlsx`, one tab named `Label`.
No calibration tab: the session starts directly with the overlap set.

Each workbook has three shaded sections, in this order:

| Section | Rows | Source | Same in every workbook? | Fill colour |
| --- | --- | --- | --- | --- |
| 1. Overlap | `N_OVERLAP` | Training pool | Yes | `OVERLAP_FILL` |
| 2. Training | Share of remaining training pool | Training pool | No, split | `TRAIN_FILL` |
| 3. Test | Share of **all** test rows | Test sheet from Task 1 (every bucket, including agree) | No, split | `TEST_FILL` |

Training pool = triage rows whose ID is **not** in the test split.

Add to the parameters:

```python
OVERLAP_FILL = "DDEBF7"  # light blue
TRAIN_FILL   = "E2EFDA"  # light green
TEST_FILL    = "FFF2CC"  # light amber
```

Shade the whole row (all visible columns) in its section's colour. Keep colours light so text
stays readable when printed. Do not add section names or a legend in the annotator view.

**Overlap set (first `N_OVERLAP` rows, identical in every workbook).**
- Sample from the training pool, stratified by `bucket` and original `Dominant Distortion`,
  so every label appears at least 3 times where the pool allows. Report any label that can't.
- Same rows, same order, at the top of every workbook.

**Section 2: Training (the rest of the training pool).**
- Shuffle remaining training rows with `SEED`, then split evenly across annotators,
  stratified by `bucket` so each workbook gets a similar disagree/ambiguous mix.
- Each row appears in exactly one workbook. Report per-workbook row counts.

**Section 3: Test (all test rows, any bucket).**
- Take every row of `test_rows_all.xlsx` from Task 1, flagged or not.
- Shuffle with `SEED`, then split evenly across annotators, stratified by original
  `Dominant Distortion` so each workbook gets a similar label mix.
- Each test row appears in exactly one workbook. Report per-workbook row counts.
- Test rows never appear in sections 1 or 2.

Report total rows per workbook and estimated time at 45 seconds per row.

**Annotator-visible columns only:**

| Column | Notes |
| --- | --- |
| Row ID | `Id_Number` |
| Text | `Patient Question` |
| Distorted part | Only if `SHOW_DISTORTED_PART` |
| Primary | Dropdown, required |
| Secondary | Dropdown, optional |
| Unsure | Dropdown: `Yes` or blank |
| Notes | Free text |

- Do **not** include original labels, bucket, model predictions, confidence scores or any
  overlap marker.
- **Primary dropdown** = the dataset's exact label strings (unique values of
  `Dominant Distortion`, including `No Distortion`) **plus `CP_LABEL`**.
- **Secondary dropdown** = the 10 distortions only.
- Use Excel data validation (openpyxl `DataValidation`). Conditional formatting highlights
  a row red when:
  - Primary is blank,
  - Secondary equals Primary,
  - Secondary is set while Primary is `No Distortion` or `CP_LABEL`.
- Wrap text in the Text column; freeze the header row; lock read-only columns.

**Master key** (`outputs/relabel/annotator_master_key.xlsx`, not given to annotators):
Row ID, annotator number, discipline, position in workbook, `section` (overlap / training / test),
`split` (train / test), bucket,
original labels, all triage columns.

## Task 3: Training reannotation CSV

Output: `outputs/relabel/training_reannotated.csv`

- All training-pool rows (flagged rows not in the test split), **all columns of the
  triage file**, unchanged.
- Count check in `outputs/relabel/training_discrepancies.md`:
  - Expected = 1,370 − (flagged rows in test split). Report expected vs actual.
  - Flagged IDs in neither the train nor the test split (**error**).
  - Duplicates (**error**).

## Acceptance checks for Tasks 1–3 (run and print before finishing)

- Test sheet row count == test split size.
- Training CSV rows + flagged test rows == 1,370.
- Overlap rows identical and in the same order across all workbooks.
- Section 2 rows across workbooks: no duplicates; union == training pool minus overlap set.
- Section 3 rows across workbooks: no duplicates; union == all test IDs.
- No ID appears in more than one section.
- Every row's fill colour matches its section.
- No annotator workbook contains label, bucket, confidence or overlap columns.
- Every Primary dropdown value is in the dataset's label set, **except `CP_LABEL`**.

## Task 4 (after the session): Merge labels and compute agreement

Script: `scripts/relabel_agreement.py`. Input: the returned annotator workbooks + master key.

**Merge.**
- Validate each workbook (same rules as the red highlighting); list invalid rows, don't drop silently.
- Output `outputs/relabel/merged_labels.csv`: Row ID, annotator, `section`, `split`, Primary,
  Secondary, Unsure, Notes, plus two derived columns:
  - `primary_collapsed`: `CP_LABEL` mapped to `No Distortion` (use this to compare with
    original labels and for training).
  - `primary_binary`: `distorted` / `none`.
- List Unsure rows in `outputs/relabel/unsure_for_adjudication.csv`.

**Agreement on the overlap set only** (sections 2 and 3 are single-annotated) (`outputs/relabel/agreement_report.md`). Compute every metric
for three label views: 12-way (with `CP_LABEL`), 11-way collapsed, and binary.

| Metric | Definition |
| --- | --- |
| Percent agreement (pairwise) | Per row, share of annotator pairs that agree; averaged over rows |
| Percent agreement (full) | Share of rows where all annotators chose the same label |
| Fleiss' kappa | All annotators together (`statsmodels.stats.inter_rater.fleiss_kappa`) |
| Pairwise Cohen's kappa | Every pair (`sklearn.metrics.cohen_kappa_score`), plus min/max |
| Light's kappa | Mean of pairwise Cohen's kappas |
| Discipline breakdown | Mean pairwise Cohen's for psych–psych, cs–cs, psych–cs |

- 95% bootstrap confidence intervals (1,000 resamples of overlap rows, `SEED`) for percent
  agreement, Fleiss' and Light's kappa.
- Label distribution on the overlap set (to judge prevalence effects on kappa).
- Count of rows originally labelled as a distortion that the majority marked `CP_LABEL`.

Commit outputs and scripts on the branch; summarise all counts, metrics and any errors in the
final message.
