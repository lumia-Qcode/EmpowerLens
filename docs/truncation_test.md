# D15 — Truncation test: head vs head_tail

## Question

Texts longer than the model's limit are cut. **head** keeps the first tokens; **head_tail** keeps the first 128
tokens and the end, where the conclusion usually is. Does keeping the end help?

## Setup

- Model: MentalRoBERTa (`mental/mental-roberta-base`, the D12 winner), **multiclass** (11 classes), frozen splits
  `data/splits`.
- Settings: T14 / D09 (lr 3e-5, 5 epochs, weight decay 0.1, batch 16) except **max length 256**, so that enough
  texts are cut for a difference to show. `--deterministic`, seeds 42 / 1337 / 2024, one Kaggle GPU.
- Only the truncation option differs between the two arms. The best epoch is chosen on val macro-F1 over the 10
  distortions (`macro_f1_10`), as in every multiclass run.
- **Validation only.** Test was not read: this compares two settings, and settings are chosen on val.
- Code: `notebooks/kaggle_runner_truncation.ipynb` → `results/results_truncation/`; table from
  `python -m src.truncation_report --results results/results_truncation` (`summary.csv`).

## How many texts get cut

Token counts with the MentalRoBERTa tokenizer, 2 special tokens included in the budget.

| Texts | n | Median tokens | Cut at 256 | Cut at 512 |
|---|---|---|---|---|
| All (corpus) | 2,530 | 155 | 27.5% | 3.2% |
| Train | 2,024 | 156 | 28.5% | 3.3% |
| Val | 253 | 142 | 22.9% | 2.8% |

## Results (val, mean ± std over 3 seeds)

| Truncation (max length 256) | Macro-F1 (11 classes) | Macro-F1 (10 distortions) | Best epoch |
|---|---|---|---|
| head | 0.215 ± 0.025 | 0.176 ± 0.027 | 5, 5, 5 |
| head_tail | 0.223 ± 0.012 | 0.185 ± 0.016 | 4, 5, 4 |
| *Reference: head at 512 (D09)* | *0.223 ± 0.014* | *0.185 ± 0.017* | |

Per seed (macro-F1 11 classes / 10 distortions):

| Seed | head | head_tail |
|---|---|---|
| 42 | 0.233 / 0.192 | 0.227 / 0.188 |
| 1337 | 0.227 / 0.190 | 0.233 / 0.199 |
| 2024 | 0.187 / 0.145 | 0.209 / 0.168 |

The reference row is D09's MentalRoBERTa multiclass run (`results/results_mentalroberta_frozensplits/`, same
settings at max length 512, not deterministic), shown for scale only.

## Conclusion

At 256 tokens, where 23% of val texts are cut, head_tail is 0.008–0.009 higher than head on both macro-F1 views,
less than one standard deviation of either arm, so the truncation choice makes **no measurable difference**. At
the 512 tokens we actually use, only 3.2% of texts are cut, so the default head truncation is kept.

## Caveats

- Three seeds per arm; head's spread comes mostly from seed 2024. With this spread a gap below ~0.03 cannot be
  told apart from seed noise.
- Head peaked at the last epoch in all three runs, head_tail in one of three, so a longer budget could change the
  numbers slightly; the conclusion would only change if it opened a gap larger than the seed spread.
- Val has 253 rows (95 No Distortion, about 16 per distortion type), so per-class F1 is noisy.
