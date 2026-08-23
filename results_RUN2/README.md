# RUN 2 — the rerun

Every result produced after the August 2026 fixes lands here. Folder names match
RUN 1 (`results_stage1/`, `results_experiments/`, …) so the same folder name means
the same experiment in either root and one registry serves both.

## What is different from RUN 1

| | RUN 1 | RUN 2 |
|---|---|---|
| determinism | off — same-seed reruns drift up to 0.047 macro-F1 | **on** — verified 0.000 difference |
| splits | `data/splits_combined` leaked 396 rows into Annotated val/test | leak-audited dirs only (`*_clean`, `*_matched`) |
| `--max-length` | 256 on the cascade track, 512 elsewhere | **512 everywhere** (2.4% truncation) |
| `--epochs` | 4, never validated | **8**, with `epoch_history.csv` so the peak is visible |
| model selection | `metric_for_best_model` was loss | task-appropriate F1 (`macro_f1_10` / `macro_f1`) |
| thresholds | fixed 0.5 | swept on **val**, reported both ways |
| test-set scoring | own test set only | **two exams** — own test set *and* the Annotated yardstick |

## Who writes here

| notebook | folder |
|---|---|
| `notebooks/wellally_distilbert_tutorial.ipynb` | `results_tutorial_distilbert/` |
| `notebooks/kaggle_runner_model_selection.ipynb` | `results_model_selection/` |
| `experiments/kaggle_runner_flat_experiments.ipynb` | `results_experiments/` |
| `notebooks/kaggle_runner_cascade.ipynb` | `results_stage1/`, `results_stage2/`, `results_cascade/`, `results_multilabel_flat/`, `results_multiclass_v2/` |

Overriding the root: the cascade bootstrap reads `EMPOWERLENS_RUN_ROOT`; the
other notebooks have a single `OUT`/`OUT_ROOT` constant in their config cell.

## Compiling

```
venv\Scripts\python.exe -m src.compile_results                     # both roots, `run` column
venv\Scripts\python.exe -m src.compile_results --roots results_RUN2  # this run only
```
