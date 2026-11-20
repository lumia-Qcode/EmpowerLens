# D27 — Unit, data-rule and performance test cases

Rows with Task = D27 on the *Test cases* tab of `D3/tasks_D3_2026-11-15.xlsx`. Run 20 Nov 2026 on branch
`lumia-space` (commit `d210137`), Windows 11 Pro, 12th Gen Intel Core i5-1245U, 15.6 GB RAM, Python 3.15 project
venv (CPU torch). Every command's full output is in `docs/test_evidence/<ID>.txt`.

## Summary

| ID | What is checked | Result | Pass / Fail |
|---|---|---|---|
| B-01 | `tests/test_data.py` | 13 passed | Pass |
| B-02 | `tests/test_input_repr.py` | 8 passed | Pass |
| B-03 | `tests/test_losses.py` | 9 passed | Pass |
| B-04 | `tests/test_specificity.py` | 4 passed | Pass |
| B-05 | whole suite, `pytest -q` | **83 passed, 0 failed, 0 errors** (34 when the sheet was written; tests added since) | Pass |
| C-01 | partitions add up, no shared IDs | `{'train': 2024, 'val': 253, 'test': 253} 0 0 0` | Pass |
| C-02 | split script refuses to overwrite | printed `REFUSING to overwrite existing split files`, exit 1; `git status data/splits` clean | Pass |
| C-03 | split IDs unchanged since frozen | identical Id_Number sets for train / val / test vs first commit `51cafef` (the only commit touching `data/splits/`) | Pass |
| C-04 | who reads the test split | 39 files: 18 (a), 11 (b), 6 (c), 4 comment only, **0 in (d)**; table below | Pass, with 2 notes |
| C-05 | fitted objects use train / val only | `src/baseline_classical.py:59` `vec.fit_transform(train[TEXT_COL])`; `src/train_transformer.py:291` `sweep_thresholds`, called at `:539` on val probabilities (`trainer.predict(val_ds)`, `:535`) | Pass |
| D-04 | every run reports specificity and FPR | 255 eval files: binary 66/66 and multiclass 51/51 have them; multilabel 27/129 | Fail (102 older multilabel runs, named below) |
| E-01 | 50 reflections, one GPU, < 300 s | *pending: `notebooks/kaggle_runner_d27_tests.ipynb` after the D25 run* | pending |
| E-02 | same 50 on CPU | finished in **71.4 s** (load 2.6 s, report 68.8 s) on 12th Gen Intel Core i5-1245U, 15.6 GB RAM | Pass |
| E-03 | same pass count on Windows and Linux | Windows: 83 passed. Linux: *pending (Kaggle notebook)* | pending |
| E-04 | encoder swap without code change | `--model distilroberta-base` smoke run finished (val macro-F1 0.231 on 100 rows); no existing file under `src/` changed | Pass |
| E-05 | data + metrics load without the app | printed `ok`; no fastapi / uvicorn / backend / empowerlens / torch / transformers module loaded | Pass |

## Full pytest output (B-05, Windows)

```
$ venv\Scripts\python.exe -m pytest -q   (Windows 11, branch lumia-space, commit d210137)
........................................................................ [ 86%]
...........                                                              [100%]
83 passed, 1 warning in 10.05s
```

The one warning is a third-party deprecation (`anyio` pytest plugin importing `CallSpec2`), not from project code.
The count is higher than the sheet's 34 because tests were added for the truncation report (D15), the
supplementary-label tooling (S03), label changes, cascade inputs, span baselines, route scoring, the relabelled
splits and the specificity audit (D-04).

## C-01 output line

```
{'train': 2024, 'val': 253, 'test': 253} 0 0 0
```

## C-04 audit: every script that mentions the test split

`git grep -l -E "\"test\"|'test'|test\.csv" -- src notebooks` → 39 files (27 on 4 Nov).
Groups: **(a)** scores a trained model, or reports scores `evaluate.py` already computed; **(b)** writes the splits;
**(c)** only counts or describes rows; **(d)** fits, tunes or chooses something; **(–)** mentions test only in a
comment that says it is not read.

| File | Group | Reason |
|---|---|---|
| src/evaluate.py | a | the test reader: loads test.csv once, scores a checkpoint with val-fitted thresholds |
| src/evaluate_cascade.py | a | scores the two-stage model on val and test with fixed thresholds |
| src/eval_two_exams.py | a | shells out to `src.evaluate`, then reads its eval JSON |
| src/eval_thresholds.py | a | scores one checkpoint four ways via `evaluate.py`; **prints a "best" row picked on test** (see note 1) |
| src/train_span_selector.py | a | after training, scores test sentences once with the val-chosen epoch and threshold |
| src/span_baselines.py | a | scores first / last / random sentence baselines on val and test (nothing fitted) |
| src/predict.py | a | prints predictions vs truth for N rows; `--split` defaults to test (display only) |
| src/tutorial_distilbert.py | a | its own numbers are val; test numbers only via `src.evaluate` |
| src/aggregate.py | a | rolls eval JSONs into summary tables |
| src/aggregate_cascade_inputs.py | a | D17 summary from eval JSONs; never opens test.csv |
| src/compare_models.py | a | pivots eval JSONs by split; no re-scoring |
| src/compile_results.py | a | compiles eval JSONs; flags leaked-split runs `valid=False` |
| src/make_results_table.py | a | results table from eval JSONs |
| src/make_rerun_table.py | a | rerun tables from eval JSONs; test section only after choosing on val |
| notebooks/cascade_bootstrap.py | a | reads test metrics from eval JSONs for a printed summary |
| notebooks/sequential_bootstrap.py | a | reads test metrics from eval JSONs for a printed summary |
| notebooks/kaggle_runner_cascade.ipynb | a | runs `evaluate_cascade` |
| notebooks/wellally_distilbert_tutorial.ipynb | a | test read only in its last section, through `src/evaluate.py` |
| src/make_splits.py | b | writes the frozen 80/10/10 splits (refuses to overwrite, C-02) |
| src/make_splits_cascade.py | b | distorted-only splits derived from the frozen ones, same IDs per partition |
| src/make_splits_cascade_inputs.py | b | D17 input versions of val/test (gold or selector sentences), labels unchanged |
| src/make_splits_clean.py | b | copies val.csv / test.csv byte-identical; de-duplicates train against them |
| src/make_splits_codipas.py | b | group split of the CODIPAS corpus (separate dataset) |
| src/make_splits_codipas_classification.py | b | group split of CODIPAS for classification |
| src/make_splits_codipas_transfer.py | b | copies Annotated test.csv unchanged as the transfer yardstick |
| src/make_splits_combined.py | b | merges extra data into train only; val/test are the frozen files |
| src/make_splits_patternreframe.py | b | PatternReframe train sets; eval splits copied from the frozen ones |
| src/make_splits_relabelled.py | b | relabelled labels on the same frozen IDs per partition (D14) |
| src/generate_clean_splits.py | b | **re-splits `data/splits_combined` with `train_test_split`** (see note 2) |
| src/audit_data.py | c | per-split label counts and truncation shares |
| src/codipas_agreement.py | c | all partitions pooled to measure CODIPAS-vs-annotation agreement |
| src/cluster_label_errors.py | c | describes re-annotation verdicts on test rows (triage list vs random); nothing chosen |
| src/experiments_flat_mentalroberta.py | c | data audit counts rows per split; test scores delegated to `src.evaluate` |
| notebooks/kaggle_runner_patternreframe.ipynb | c | prints row counts per split folder |
| notebooks/kaggle_runner_sequential.ipynb | c | prints row counts per split folder |
| src/fit_binary_threshold.py | – | docstring: only val.csv is read |
| src/specificity.py | – | docstring: test.csv is never opened |
| src/span_check.py | – | docstring: reads Annotated_data.csv only |
| src/synth/README.md | – | says test.csv stays real and is read only by evaluate.py |

**Note 1 — `eval_thresholds.py`.** It prints the threshold-mode / cap combination with the best *test* macro-F1.
No reported setting was taken from that line: the 2-label cap rests on the data (no row carries more than 2
labels, stated in its own docstring), the D13 binary threshold is fitted on val by `src/fit_binary_threshold.py`,
and multilabel thresholds by `sweep_thresholds` on val. The only other mention is a pointer in `docs/TODO.md`.

**Note 2 — `generate_clean_splits.py`.** It pools train + val + test of `data/splits_combined` and re-splits them
with `train_test_split`. It never touches the frozen `data/splits/`, but it breaks the project rule that only
`make_splits.py` splits. `compile_results.py` already marks `results_combined` as leaked (`valid=False`: "194/253 val
+ 189/253 test rows also in train"), so those numbers must not be compared with anything.

**For Ch6.** `src/predict.py` and several scoring scripts read test themselves, so "only the evaluation module reads
the test partition" is too strong. The sheet's wording fits what the audit found: *"Only evaluation and descriptive
scripts read the test partition; nothing is fitted or chosen on it."*

## D-04: specificity and FPR in every evaluation run

`python -m src.specificity_audit --out docs/test_evidence/D-04_specificity_audit.csv` reads every `eval_*.json`
under `results/` and `results_RUN2/` (255 files; the 6 span-selector files are not classifier evals and are
skipped). Binary specificity = recall of `non_distorted`; 11-class specificity = recall of `no_distortion`;
multilabel = `macro_specificity_10` (per-label specificity averaged over the 10 types). FPR = 1 − specificity.
The per-run table (510 rows: each file × val/test) is in the CSV.

| Task | Files with specificity + FPR |
|---|---|
| binary | 66 / 66 |
| multiclass (11 classes) | 51 / 51 |
| multilabel | 27 / 129 |

The 102 multilabel files without it were written before `src/evaluate.py` recorded per-label specificity (added
for D17), and their predictions were not saved, so it cannot be recomputed from the files. They are in
`results/results_RUN1/*`, `results/results_RUN2/*`, `results/results_experiments` (+ `exp6`),
`results/results_mentalroberta_frozensplits`, `results/results_model_triage` and
`results_RUN2/results_tutorial_distilbert/*` (full list in `docs/test_evidence/D-04.txt`). Per the sheet, Ch9 names
these multilabel runs as missing specificity. Every binary run, which is where specificity is the headline
(FR-10), has it.

## E-01 / E-02: 50-reflection cohort report

Script: `python -m src.time_cohort_report --models <D25 app_models> --n 50` — loads the three D25 models (binary +
multilabel MentalRoBERTa, D03 selector), then on the first 50 rows of `data/splits/val.csv` runs the binary decision,
the types, sentence selection on flagged texts and the aggregation (counts per type, one example sentence per
shown type, suppression below the cohort threshold of 5). Model loading is included in the total.

| Row | Hardware | Load (s) | Report (s) | Total (s) | Target |
|---|---|---|---|---|---|
| E-01 | *pending (Kaggle GPU)* | | | | < 300 |
| E-02 | CPU only: 12th Gen Intel Core i5-1245U, 15.6 GB RAM, Windows 11 Pro, torch 2.15.0+cpu | 2.6 | 68.8 | **71.4** | none (must finish): Pass |

E-02 ran on 20 Nov with the D25 models (`mentalroberta-d12-s42-20261120`): 20 of the 50 texts were flagged, 192
sentences scored by the selector, and the report kept one type above the cohort threshold of 5 (Mind Reading, 9).
Evidence: `docs/test_evidence/E-02_cpu_timing.json` (its `cpu` field holds Windows' generic processor string; the
model name above is from `Get-CimInstance Win32_Processor`).

## E-03: pass count on Windows and Linux

| OS | Command | Result |
|---|---|---|
| Windows 11 Pro | `venv\Scripts\python.exe -m pytest -q` | 83 passed, 0 failed |
| Linux (Kaggle) | `python -m pytest -q` | *pending (`notebooks/kaggle_runner_d27_tests.ipynb`)* |

## Still to do

1. After the D25 Kaggle run finishes: open `notebooks/kaggle_runner_d27_tests.ipynb` on Kaggle, add the D25
   notebook's output as input, run it (≈10 min), and bring back `d27.zip` → fills E-01 and the Linux half of E-03.
(E-02 is done: see above.)
