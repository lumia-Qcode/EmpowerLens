# EmpowerLens app

Code: **https://github.com/izza14/empowerlens-demo**

## Install

Needs Python 3.10–3.12 and Node.js 18+.

```bash
git clone https://github.com/izza14/empowerlens-demo.git
cd empowerlens-demo
py -3.12 -m venv backend/.venv
backend\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
backend\.venv\Scripts\pip install -r backend/requirements.txt
npm install
```

## Model

Loads the **D12 winner: MentalRoBERTa (`mental/mental-roberta-base`)**, chosen by the rule in
`docs/model_selection.md`, with the **D13 thresholds** (`docs/thresholds.md`):

- `binary_mental-roberta-base_42`: flags a reflection as distorted when P(distorted) is at or above the lowest
  val threshold with false-positive rate ≤ 0.15;
- `multilabel_mental-roberta-base_42`: names the distortion types (per-type val thresholds, at most 2, only for
  flagged reflections);
- the D03 sentence selector (seed 42): picks the sentences shown as evidence.

All three were trained on `data/splits`, seed 42, lr 3e-5, 5 epochs, weight decay 0.1, max length 512. The version
(`mentalroberta-d12-s42-<date>`) and every threshold are in `MODEL_CARD.json`; the researcher view shows the model
name, version and checkpoints.

Weights are not in git. Get them by running `notebooks/kaggle_runner_d25_app_models.ipynb` (this repo) on Kaggle,
then unzip `app_models.zip` into the app's `models/` folder and rename `app_models` to `d12`. The app's earlier
multi-task model is still available with `EMPOWERLENS_ARCH=multitask`.

The app gives the same prediction as the evaluation code: compare with
`python -m src.app_parity --models <app>/models/d12 --val-row <n>` (record in `docs/app_model_swap.md`).

## Start

```bash
backend\.venv\Scripts\python -m backend.seed_demo --reset
backend\.venv\Scripts\python -m uvicorn backend.app:app --port 8000
npm run dev
```

Run the last command in a second terminal, then open http://localhost:5173.
Demo logins: trainer `trainer@empowerlens.demo` / `trainer123`, researcher `researcher@empowerlens.demo` / `research123`.

## Data

Stored in the SQLite file `backend/empowerlens.db` (change with `EMPOWERLENS_DB`). Predictions are not stored.

## Tested by

| Name | Date | Worked from this README alone? |
|---|---|---|
| | | |
