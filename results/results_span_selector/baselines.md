# D10 - Span selector vs simple baselines

Does the trained sentence selector (D03, MentalRoBERTa, one sentence at a time) beat simple rules? Rules: always pick the **first** sentence, always the **last**, or one **random** sentence per row (seed 42). All methods are scored with the same code as D03 (`score_part` in `src/train_span_selector.py`) against the T12 sentence labels in `results/results_sentence_label/`: sentence-level precision, recall and F1 over all sentences, and **row hit rate** = share of rows where at least one picked sentence is a gold sentence.

**13 rows were excluded** because their quoted distorted part couldn't be found in the text (`excluded_rows.csv`); No-Distortion rows have no gold sentence and are not scored.

The selector is the mean ± std (n−1) over seeds 42, 1337, 2024, each with its own threshold chosen on val; the rules need no tuning and are run once. The selector picks about 2.6 sentences per row while every rule picks exactly one, which raises its recall and lowers its precision; the **extra** top-1 row restricts the selector to its single highest-scoring sentence per row, so it can be compared with the rules on an equal budget.

## val

Scored: **157 rows, 1,586 sentences** (270 gold, 17.0%).

| Method | Precision | Recall | Sentence F1 | Row hit rate | Sentences picked |
| --- | --- | --- | --- | --- | --- |
| trained selector (mean ± std, 3 seeds) | 0.373 ± 0.016 | 0.542 ± 0.025 | 0.441 ± 0.010 | 0.705 ± 0.013 | 393 |
| first sentence | 0.146 | 0.085 | 0.108 | 0.146 | 157 |
| last sentence | 0.293 | 0.170 | 0.215 | 0.293 | 157 |
| random sentence (seed 42) | 0.248 | 0.144 | 0.183 | 0.248 | 157 |
| extra: trained selector, top-1 sentence per row | 0.537 ± 0.027 | 0.312 ± 0.015 | 0.395 ± 0.020 | 0.537 ± 0.027 | 157 |

On sentence F1, the best rule is **last sentence** (0.215); the selector is 0.226 above it (selector std 0.010), so it **does** beat the best rule by more than its std.
On row hit rate, the best rule is **last sentence** (0.293); the selector is 0.412 above it (selector std 0.013), so it **does** beat the best rule by more than its std.
Equal budget (one sentence per row): the selector's top-1 pick is gold in 0.537 ± 0.027 of rows vs 0.293 for the best rule (last sentence), so the advantage is not only an effect of picking more sentences.

Random rule, analytic expectation over all draws: precision 0.253, recall 0.147, F1 0.186, row hit rate 0.253 (the seed-42 row above is one draw). "Last sentence" is gold in 0.293 of rows, +0.040 vs that chance level, and "first sentence" -0.106.

## test

Scored: **159 rows, 1,591 sentences** (290 gold, 18.2%).

| Method | Precision | Recall | Sentence F1 | Row hit rate | Sentences picked |
| --- | --- | --- | --- | --- | --- |
| trained selector (mean ± std, 3 seeds) | 0.354 ± 0.019 | 0.521 ± 0.015 | 0.421 ± 0.015 | 0.665 ± 0.010 | 428 |
| first sentence | 0.164 | 0.090 | 0.116 | 0.164 | 159 |
| last sentence | 0.258 | 0.141 | 0.183 | 0.258 | 159 |
| random sentence (seed 42) | 0.252 | 0.138 | 0.178 | 0.252 | 159 |
| extra: trained selector, top-1 sentence per row | 0.539 ± 0.057 | 0.295 ± 0.031 | 0.382 ± 0.040 | 0.539 ± 0.057 | 159 |

On sentence F1, the best rule is **last sentence** (0.183); the selector is 0.238 above it (selector std 0.015), so it **does** beat the best rule by more than its std.
On row hit rate, the best rule is **last sentence** (0.258); the selector is 0.407 above it (selector std 0.010), so it **does** beat the best rule by more than its std.
Equal budget (one sentence per row): the selector's top-1 pick is gold in 0.539 ± 0.057 of rows vs 0.258 for the best rule (last sentence), so the advantage is not only an effect of picking more sentences.

Random rule, analytic expectation over all draws: precision 0.251, recall 0.138, F1 0.178, row hit rate 0.251 (the seed-42 row above is one draw). "Last sentence" is gold in 0.258 of rows, +0.007 vs that chance level, and "first sentence" -0.088.

Source: `baselines.csv` (same numbers, unrounded std). Script: `python -m src.span_baselines`.
