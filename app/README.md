# EmpowerLens app

Code: **https://github.com/izza14/empowerlens-demo**

## Install

Needs Python 3.10–3.12 and Node.js 18+.

```bash
git clone https://github.com/izza14/empowerlens-demo.git
cd empowerlens-demo
py -3.11 -m venv backend/.venv
backend\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
backend\.venv\Scripts\pip install -r backend/requirements.txt
npm install
```

## Model

Loads **MentalRoBERTa (`mental/mental-roberta-base`), multi-task version** from `models/multitask/`, with the
thresholds stored beside it. Weights are not in git: unzip `models.zip` from the Kaggle run into `models/`.

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
