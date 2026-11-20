# EmpowerLens — Cognitive Distortion Detection

An NLP research pipeline that detects **cognitive distortions** (CBT-defined
irrational thought patterns) in patient-written text. It replicates the
methodology of:

> Shreevastava & Foltz (2021), *"Detecting Cognitive Distortions from
> Patient-Therapist Interactions."*

Given a patient's message, the pipeline answers two questions:

1. **Binary** — does the text contain a cognitive distortion at all?
2. **Multi-class** — if so, which of 10 distortion types is dominant
   (Emotional Reasoning, Overgeneralization, Mental Filter, Should Statements,
   All-or-Nothing, Mind Reading, Fortune Telling, Magnification,
   Personalization, Labeling)?

This is an experiment/replication repo — there is no web app, server, or UI.

## How it works

The entire program lives in [`cd_pipeline.py`](cd_pipeline.py), organized in
five sections:

1. **Load data** — reads `Annotated_data.csv`, keeps entries with real text,
   normalizes label spellings, and derives a binary label and an 11-class
   label from the `Dominant Distortion` column.
2. **Feature extraction** — turns each message into a numeric vector using
   four methods. Because the original paper's tools require internet access
   (GloVe, BERT, licensed LIWC), this repo uses fully offline, self-trained
   substitutes:

   | Feature set | What it is | Substitutes |
   |-------------|-----------|-------------|
   | `SIF` | Word2Vec trained on this corpus + Smooth Inverse Frequency weighting | pretrained GloVe |
   | `S-BERT(sub)` | Doc2Vec (Distributed Memory) | a pretrained transformer / S-BERT |
   | `LIWC` | hand-built lexicon (pronouns, negations, negative emotion, absolutes, "should" words, etc.) | licensed LIWC dictionary |
   | `POS` | spaCy part-of-speech tags → Word2Vec embeddings | — (real POS tagging) |
   | `S-BERT+LIWC` | hybrid: Doc2Vec vectors concatenated with LIWC features | — |

3. **Classification** — an 80/20 stratified split, standard scaling, then five
   classifiers: Logistic Regression, SVM, Decision Tree, k-NN (k=15), and MLP.
   Scored with weighted F1.
4. **Main** — runs every feature set through both tasks and writes result
   tables.

## Files

| File | Role |
|------|------|
| [`cd_pipeline.py`](cd_pipeline.py) | The complete pipeline (input → features → models → results) |
| `Annotated_data.csv` | **Input.** ~2,530 hand-labeled patient entries (`Patient Question`, `Dominant Distortion`, …) |
| `labeled_data_sample.csv` | **Output.** Cleaned text plus the derived binary / type labels actually used |
| `binary_f1_results.csv` | **Output.** F1 scores for distorted vs. non-distorted |
| `multiclass_f1_results.csv` | **Output.** F1 scores for distortion type |

## Setup

Python 3.9+ recommended.

```bash
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

```bash
python cd_pipeline.py
```

This prints the label distribution and F1 tables to the console and refreshes
`binary_f1_results.csv`, `multiclass_f1_results.csv`, and
`labeled_data_sample.csv`.

## Results (current run)

**Binary — distorted vs. non-distorted** (weighted F1):

| Classifier | SIF | S-BERT(sub) | LIWC | POS | S-BERT+LIWC |
|------------|-----|-------------|------|-----|-------------|
| Log. Reg. | 0.71 | 0.68 | 0.71 | 0.54 | **0.73** |
| SVM | **0.74** | 0.73 | 0.71 | 0.56 | 0.72 |
| Decision Tree | 0.58 | 0.62 | 0.62 | 0.55 | 0.62 |
| k-NN (k=15) | 0.61 | 0.66 | 0.71 | 0.57 | 0.65 |
| MLP | 0.69 | 0.67 | 0.71 | 0.60 | **0.73** |

**Multi-class — distortion type** (weighted F1):

| Classifier | SIF | S-BERT(sub) | LIWC | POS | S-BERT+LIWC |
|------------|-----|-------------|------|-----|-------------|
| Log. Reg. | 0.29 | 0.29 | 0.28 | 0.20 | 0.30 |
| SVM | 0.24 | 0.22 | 0.26 | 0.20 | 0.23 |
| Decision Tree | 0.27 | 0.25 | 0.26 | 0.18 | 0.26 |
| k-NN (k=15) | 0.30 | 0.23 | 0.29 | 0.22 | 0.23 |
| MLP | 0.28 | 0.26 | **0.33** | 0.22 | 0.30 |

**Takeaway:** detecting *whether* a distortion is present is workable (~0.74
F1), but identifying *which* of 10 types is much harder (~0.20–0.33 F1) — in
line with the original paper's findings.

## Notes

- `Secondary Distortion (Optional)` in the annotations is not used; only the
  dominant distortion is treated as ground truth, matching the paper's primary
  classification setup.
- `make_synthetic_labels()` in `cd_pipeline.py` is retained for reference only
  (an earlier random-label fallback) and is not called during a real run.
- Results may shift slightly across environments due to library versions, even
  with a fixed random seed.
