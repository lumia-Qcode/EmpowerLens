# Reannotation session — 27 Sep 2026

Relabelling of TherapistQA rows that the model flagged as possibly mislabelled.
Built from `RELABEL_SPLIT_TASK.md`; annotators work from
`EmpowerLens_Relabelling_Codebook.pdf`.

## Layout

```
inputs/triage_full.csv        model triage output, all 2,530 rows (input, do not edit)
scripts/build_relabel_files.py  Tasks 1-3: test sheet, workbooks, training CSV
scripts/relabel_agreement.py    Task 4: merge returned workbooks + agreement
outputs/relabel/              everything the scripts produce
```

Run from the repo root:

```
venv\Scripts\python.exe reannotation_2026-09-27/scripts/build_relabel_files.py
venv\Scripts\python.exe reannotation_2026-09-27/scripts/relabel_agreement.py   # after the session
```

## Decisions taken (differ from, or resolve gaps in, the task spec)

| Spec | What was used | Why |
| --- | --- | --- |
| Triage file = 1,370 flagged rows | `triage_full.csv` has all 2,530 rows; flagged = buckets **A + B = 1,373** | The delivered file is the full triage output; A/B are the flagged buckets, C (1,157) is agree |
| Dataset = 2,531 rows | 2,530 (`Annotated_data.csv`) | Measured; the public release is 2,530 |
| Test split 80/20 | `data/splits/test.csv`, 253 rows (repo split is 80/10/10) | The repo's frozen split |
| Val rows unaddressed | Val (253 rows) is part of the **training pool** | Model selection happens on val; cleaning train but not val would select against a noisier target |
| `data/triage/…`, `data/raw/…` paths | `reannotation_2026-09-27/inputs/`, repo-root `Annotated_data.csv` | Actual repo layout |
| One tab named `Label` | Tab named after the annotator; a hidden `_lists` tab holds the dropdown values | Inline dropdown lists exceed Excel's 255-character limit |

`Distorted part` is not in the triage file; it is joined from `Annotated_data.csv`.

## Counts

| | |
| --- | --- |
| Flagged rows (A + B) | 1,373 |
| Flagged rows inside the test split | 139 |
| Training pool (flagged, not in test) | 1,234 |
| Test sheet (all buckets) | 253 (139 flagged + 114 agree) |
| Overlap set | 60 rows, identical in every workbook |

Per workbook: 60 overlap + ~293 training + ~63 test ≈ **415 rows ≈ 5.2 h** at 45 s/row.

| Annotator | File | Discipline |
| --- | --- | --- |
| Laiba | `annotator_1_Laiba.xlsx` | psych |
| Reesha | `annotator_2_Reesha.xlsx` | psych |
| Hurema | `annotator_3_Hurema.xlsx` | psych |
| CS majors | `annotator_4_CS_majors.xlsx` | cs |

## Workbook format

Columns: Row ID, Text, Distorted part, Primary, Secondary, Unsure, Notes.
No original labels, bucket, model predictions or overlap marker.

Row shading: overlap `DDEBF7` (blue), training `E2EFDA` (green), test `FFF2CC` (amber).
Primary dropdown = the 11 dataset labels + `No Distortion (calibrated pessimism)`;
Secondary = the 10 distortions; both are inline lists, so there is no helper sheet.
A row turns red when Secondary repeats Primary, when Secondary is set while Primary
is a No-Distortion value, or when Primary is blank *on a row that was otherwise
started* (a blank Primary on an untouched row is not an error — flagging those
would paint the whole sheet red before the session begins).

`Unsure` is seeded with `FALSE` so it can be converted to real checkboxes in one
step: select the column's data rows and use **Insert > Checkbox** (Excel 365).
Without that it behaves as a `TRUE`/`FALSE` dropdown; the merge script accepts
either, plus `Yes`, `x` and `1`.

Sheets are **not** protected (`PROTECT_SHEET = False`): a protected sheet reads as
"broken" to an annotator whose Excel opens the file in Protected View.

`annotator_master_key.xlsx` maps every row to its annotator, section, split, bucket,
original labels and triage columns. **Not for annotators.**
