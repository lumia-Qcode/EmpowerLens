# CLAUDE.md — EmpowerLens

Guidance for AI/dev sessions on this repo. Read before changing anything.

## What this project is

EmpowerLens detects CBT **cognitive distortions** in short written self-reflections.
It replicates and extends Shreevastava & Foltz (2021), *"Detecting Cognitive
Distortions from Patient-Therapist Interactions."* Two questions per input:
**binary** (is a distortion present?) and **fine-grained** (which of 10 types?).

Two parallel bodies of code live here:
- **`cd_pipeline.py`** — a teammate's classical-ML replication (offline embedding
  substitutes). **Frozen** (see below).
- **`src/`** — the clean, no-leakage transformer pipeline. Active development.

## FROZEN FILES — do not modify, refactor, move, reformat, or delete

These are a submitted artifact. Treat as read-only reference:

- `cd_pipeline.py`
- `README.md`
- `requirements.txt`
- `binary_f1_results.csv`, `multiclass_f1_results.csv`, `labeled_data_sample.csv`
- `Annotated_data.csv` (the source data — never edit)

New work goes in `src/`, `tests/`, `data/splits/`, `results/`, `notebooks/`, and
`requirements-transformer.txt`. Do **not** edit `requirements.txt`; the
transformer stack has its own `requirements-transformer.txt`.

## Measured dataset facts (do not re-derive)

`Annotated_data.csv` — 2,530 rows, columns: `Id_Number`, `Patient Question`,
`Distorted part`, `Dominant Distortion`, `Secondary Distortion (Optional)`.

- **Encoding is UTF-8** (read with `utf-8-sig`). The file is **NOT latin-1** —
  an earlier spec claimed latin-1, but that mojibake-corrupts the text in ~86%
  of rows (curly quotes / em-dashes become `â\x80\x99`). `src/data.py` reads
  UTF-8; keep it that way.
- `Dominant Distortion` distribution: No Distortion 933, Mind Reading 239,
  Overgeneralization 239, Magnification 195, Labeling 165, Personalization 153,
  Fortune-telling 143, Emotional Reasoning 134, Mental filter 122,
  Should statements 107, All-or-nothing thinking 100.
- 1,597 rows distorted; 933 No Distortion. 416 rows carry a secondary
  distortion (26.0% of distorted rows); no secondary duplicates its dominant,
  and no No-Distortion row has a secondary. The `load_raw` filters (null / <10
  chars) drop **zero** rows on the shipped data.
- Label strings have inconsistent casing/hyphenation; `LABEL_CANON` in
  `src/data.py` maps every raw string explicitly (no fuzzy matching — unmapped
  values raise).

## Non-negotiable conventions

1. **Splits are immutable.** `data/splits/{train,val,test}.csv` +
   `split_manifest.json` are generated **once** by `src/make_splits.py`
   (80/10/10, `random_state=42`, multi-label stratified) and **committed**.
   `make_splits.py` refuses to overwrite without `--force`. Every script reads
   these files. `data/splits/` is intentionally **not** gitignored. Do not call
   `train_test_split` anywhere except `make_splits.py`.
2. **No leakage.** Any vectorizer / embedding / scaler / threshold is fit on
   **train only**. (This is the specific bug in the frozen `cd_pipeline.py`,
   which trains Word2Vec/Doc2Vec on the full corpus before splitting — do not
   reproduce it.)
3. **`test.csv` is read only by `src/evaluate.py`.** All epoch selection,
   threshold tuning, and model comparison use val.
4. **Multi-class is 11 classes** (`no_distortion` + 10) over all 2,530 rows —
   never a 10-class distorted-only variant. Always also report `macro_f1_10`
   (macro over the 10 distortion classes, `no_distortion` dropped, same
   predictions).
5. **Never report accuracy as a headline.** Headlines are F1 variants
   (`weighted_f1`, `macro_f1`, `macro_f1_10`, `micro_f1`, `positive_class_f1`).
6. **Three seeds** (42, 1337, 2024), report mean ± std.
7. **Never commit model weights** (`checkpoints/`, `*.safetensors`, etc. are
   gitignored). No synthetic/random labels — fail loudly if real labels missing.

## `src/` layout

| module | role |
|--------|------|
| `data.py` | `load_raw`, `make_targets` (`y_bin`, `y_mc` 11-class, `y_ml` 10-col), `LABEL_CANON` |
| `make_splits.py` | once-only frozen split generator + manifest |
| `metrics.py` | shared metric bundle + `paper_comparison.csv` row shape + upsert |
| `baseline_classical.py` | TF-IDF + balanced LogReg floor (all 3 tasks, val) |
| `train_transformer.py` | HF `Trainer` fine-tune; `--task`, weighted CE / BCE, head/head_tail truncation, `--smoke` |
| `evaluate.py` | only test-set reader; per-class CSV, `paper_comparison.csv`, confusion PNGs, `eval_*.json` |
| `aggregate.py` | rolls `eval_*.json` into `month1_summary*.csv` + `no_distortion_contribution.md` |

## Running (Windows)

Use the project venv: `venv\Scripts\python.exe`. Examples:

```
venv\Scripts\python.exe -m pytest -q
venv\Scripts\python.exe -m src.make_splits
venv\Scripts\python.exe -m src.baseline_classical
venv\Scripts\python.exe -m src.train_transformer --task multiclass --smoke --max-length 128
venv\Scripts\python.exe -m src.evaluate --checkpoint checkpoints/multiclass_roberta-base_42
venv\Scripts\python.exe -m src.aggregate
```

Real transformer training is impractical on CPU — use `notebooks/kaggle_runner.ipynb`
on a free Kaggle GPU, then copy `results/` back.

## Environment gotchas

- **Windows + torch CPU** hits an OpenMP crash (`0xC0000005`); `train_transformer.py`
  sets `KMP_DUPLICATE_LIB_OK=TRUE` before importing torch. Keep that.
- Installed stack is on the newest majors (torch 2.13, **transformers 5.x**,
  numpy 2.x, pandas 3.x). transformers 5.x removed
  `tokenizer.build_inputs_with_special_tokens` (we add special tokens manually)
  and uses `eval_strategy` / `processing_class` / `compute_loss(..., num_items_in_batch)`.
- `head_tail` truncation keeps the first 128 + last (budget−128) tokens; it
  degrades to plain head truncation when `--max-length` is too small.
- `Path.write_text()` on Windows defaults to cp1252 — pass `encoding="utf-8"`
  for any file that may contain non-ASCII (e.g. the generated markdown).
