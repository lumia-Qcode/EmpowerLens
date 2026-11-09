# D03 — Span selector: what we did and how it performed

## What it is
A sentence-level selector. For each sentence of a reflection it outputs a score (0–1) for "this sentence carries the distortion", and a sentence is **selected** when the score is above a threshold. It is the first half of the cascade (selector, then classifier) tested in D17.

## What we did
- **Data:** the T12 sentence labels (`results/results_sentence_label/`): distorted rows with a located quote only (13 rows excluded because the quote couldn't be found; No-Distortion rows have no span and are not included). Splits follow `data/splits`, so all sentences of a text stay in one split.
  - val: 157 rows, 1,586 sentences, 270 gold (17.0%)
  - test: 159 rows, 1,591 sentences, 290 gold (18.2%)
- **Model:** MentalRoBERTa (`mental/mental-roberta-base`) with one score output. Each sentence is read **on its own** (no surrounding text), max length 128.
- **Training:** `python -m src.train_span_selector --seed <seed>`; learning rate 3e-5, up to 5 epochs, batch 32, weight decay 0.1, binary cross-entropy weighted for the ~17% positives (`pos_weight = negatives / positives`). Kaggle GPU, seeds 42, 1337, 2024. Full settings: `config.json`.
- **Choices made on val only:** the epoch (best val sentence F1) and the threshold (the one maximising val sentence F1). The same threshold was then applied unchanged to test, which was scored once.

## Results (mean ± std over 3 seeds, std with n−1)

| Split | Precision | Recall | Sentence F1 | Row hit rate* |
|---|---|---|---|---|
| val (157 rows, 1,586 sentences) | 0.373 ± 0.016 | 0.542 ± 0.025 | 0.441 ± 0.010 | 0.705 ± 0.013 |
| test (159 rows, 1,591 sentences) | 0.354 ± 0.019 | 0.521 ± 0.015 | 0.421 ± 0.015 | 0.665 ± 0.010 |

*Share of rows where at least one selected sentence is a gold sentence.

Per seed:

| Seed | Best epoch | Threshold (val) | Val F1 | Test F1 | Test selected / gold |
|---|---|---|---|---|---|
| 42 | 2 | 0.711 | 0.451 | 0.412 | 418 / 290 |
| 1337 | 3 | 0.469 | 0.431 | 0.413 | 456 / 290 |
| 2024 | 2 | 0.715 | 0.442 | 0.438 | 409 / 290 |

## How to read this
- **Modest, not strong.** Test sentence F1 is 0.42. Precision (0.35) is about double the share of gold sentences (0.18), so the selector is clearly better than picking at random, but about two in three selected sentences are not the annotator's quoted sentence. It selects ~1.4× as many sentences as there are gold ones.
- **Recall about 0.52:** roughly half of the gold sentences are found; in about two of three rows (66.5%) at least one gold sentence is among the picks.
- **Val to test gap is small** (F1 0.441 to 0.421, about 1.3 std), so there is no sign of overfitting to val; the threshold carried over.
- **Threshold is unstable across seeds** (0.47 for seed 1337 vs ~0.71 for the others). Precision and recall trade off, so the F1 barely changes, but any later step that uses the selected sentences (D17, D22) should use each seed's own threshold, not an average.
- **Not yet compared with simple rules.** Whether 0.42 beats "first sentence", "last sentence" or "random sentence" is D10's question; until then, don't claim the selector is good, only that it works above chance.
- **Caveats:** the quote can be several sentences or edited, so "gold" is a strict target; the selector sees one sentence without its context, which likely limits it (a context-aware version is an obvious FYP-2 variant).

## Files
`config.json`; `eval_{val,test}_<seed>.json` (precision, recall, F1, row hit rate, counts, threshold, best epoch); `predictions_{val,test}_<seed>.csv` (`Id_Number`, `sent_idx`, `sentence`, `score`, `selected`, `gold`). The checkpoints stay on Kaggle and are not committed. The 13 unlocated rows are excluded as in T12.
