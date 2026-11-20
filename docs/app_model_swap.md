# D25 — The demo app now runs the D12 winner

## What the app loads

The demo app (`izza14/empowerlens-demo`, commit `570b82a`) serves the encoder chosen in `docs/model_selection.md`
(D12), **MentalRoBERTa (`mental/mental-roberta-base`)**, with the D13 thresholds (`docs/thresholds.md`).

- **Version:** `mentalroberta-d12-s42-20261120`, from `MODEL_CARD.json` in the app's `models/d12/`. Trained on Kaggle
  with `notebooks/kaggle_runner_d25_app_models.ipynb` at main-repo commit `cb4786a`.
- **Training data and settings:** `data/splits`, seed 42, lr 3e-5, 5 epochs, weight decay 0.1, max length 512, head
  truncation.
- **Where the weights live:** not in git. They are in the app's `models/d12/` folder, which is gitignored.

| Model | Checkpoint | Decision rule |
|---|---|---|
| Binary | `binary_mental-roberta-base_42` | distorted when P(distorted) ≥ **0.9767**, the lowest val threshold with FPR ≤ 0.15 (val FPR 0.147, 14 of 95 non-distorted texts flagged) |
| Multi-label | `multilabel_mental-roberta-base_42` | per-type val thresholds (0.45–0.80); at most 2 types; types reported only for texts judged distorted |
| Sentence selector (D03) | `span_selector/model.pt` | a sentence is selected when its score ≥ 0.7165 (max val sentence F1) |

The binary run went through the D12 failed-run rule once: the first run's val macro-F1 was below 0.5, so it was
re-run with identical settings and the re-run kept (`failed_run_reruns: 1` in the model card).

### Scores of these exact weights (seed 42 only)

| Model | Split | Macro-F1 | Other |
|---|---|---|---|
| Binary | val | 0.710 | specificity 0.853, sensitivity 0.627, ROC-AUC 0.828 |
| Binary | test | 0.695 | specificity 0.793, sensitivity 0.646, ROC-AUC 0.801 |
| Multi-label | val | 0.277 | micro-F1 0.293 |
| Multi-label | test | 0.247 | micro-F1 0.263 |
| Selector | val | sentence F1 0.451 | precision 0.378, recall 0.559, row hit rate 0.720 |
| Selector | test | sentence F1 0.411 | precision 0.349, recall 0.500, row hit rate 0.660 |

These are single-seed numbers for the deployed weights. The reported 3-seed results are in `docs/thresholds.md`
(binary, D13) and `results/results_mentalroberta_frozensplits/` (multi-label, D09). Test was read only by
`src/evaluate.py`, inside the Kaggle run.

## Researcher view

`/api/health` on the running backend (20 Nov 2026) returned:

```
architecture: classifier_pair      model_name: MentalRoBERTa
model_version: mentalroberta-d12-s42-20261120      encoder: mental/mental-roberta-base
checkpoints: binary_mental-roberta-base_42, multilabel_mental-roberta-base_42, span_selector/model.pt
thresholds.binary: 0.9767      selection_threshold: 0.7165      main_repo_commit: cb4786a
```

The researcher view's Model panel shows the name, version, architecture, encoder, training date, binary threshold
and checkpoints from this response. Saved reviews record `model_version`.

## Same prediction in the app and in the evaluation script

**The two sides:**
- **Evaluation script:** `python -m src.app_parity --models <app>/models/d12 --val-row N`, run in this repo's venv
  (torch 2.15.0+cpu, transformers 5.23.0). It uses `src/evaluate.py`'s code path: `encode_texts` →
  `predict_logits` → `predictions_from_logits`, with the D13 threshold, the per-type thresholds and the 2-label cap.
- **App:** `empowerlens.inference.Predictor("models")` in the app's own venv (torch 2.15.0+cpu, transformers
  4.57.6). The live-analysis endpoint was also called once (last column).

**Texts:** val rows only, never test.
- Rows 1 and 6 are judged distorted.
- Row 15 sits just under the binary threshold (0.9743 vs 0.9767), which makes it a strict check of the decision rule.

| Val row (Id_Number) | Evaluation script | App (`Predictor`) | Largest type-score difference | App live endpoint |
|---|---|---|---|---|
| 1 (4522) | P = 0.9840, distorted; Should statements + Personalization | P = 0.9840, distorted; Personalization + Should statements | 0.0000 | – |
| 6 (4559) | P = 0.9915, distorted; Mind Reading + Personalization | P = 0.9915, distorted; Mind Reading + Personalization | 0.0000 | P = 0.9915, distorted; Mind Reading + Personalization |
| 15 (4600) | P = 0.9743, not distorted | P = 0.9743, not distorted | 0.0000 | – |

Every probability and all 10 type scores are identical to 4 decimals on all three rows, across two transformers
versions, and every decision is the same. Only the order of the types differs: the script lists them in label
order, the app by score.

On row 15, the multi-label model alone would fire Overgeneralization. The evaluation script scores that model on
its own, while the app reports types only for texts the binary model flags. The app's behaviour is the intended
one (README §4.4).

**Selected sentences:**
- Row 1: the app selected 3 sentences.
- Row 6: the app selected 1 sentence.
- Row 15: none, since the text isn't flagged.

## Done-when check (D3 brief)

- The app loads the checkpoint named in `docs/model_selection.md` with the thresholds in `docs/thresholds.md`.
  **Yes:** the two seed-42 checkpoints, retrained as that file instructs, with the D13 rule.
- The researcher view shows the model name and version. **Yes:** see the health output above.
- `app/README.md` names the model. **Yes:** updated in commit `cb4786a`.
- One test reflection gives the same prediction in the app as in the evaluation script. **Yes:** val row 6 (and two
  more), table above.
