# Cascade run — results, 2026-08-16

First end-to-end cascade run on verified leakage-free data.
Backbone: `mental/mental-roberta-base` (single backbone for every stage).
Three seeds (42, 1337, 2024), mean ± std, **test** split.

Splits: `data/splits` — train 2,024 / val 253 / test 253, verified
**train-in-val = 0, train-in-test = 0**.
Stage 2: `data/splits_stage2` — 1,278 / 158 / 161, derived by *filtering*
`data/splits` to `y_bin == 1` (no re-splitting, so the row-level boundary is
inherited unchanged).

Training config: `--max-length 256 --truncation head_tail --batch-size 32`;
Stage 2 adds focal loss (γ=2.0) + layer-wise LR decay (0.9) at lr 3e-5.

---

## 1. Valid results

| Model | Task | weighted_f1 | macro_f1 | macro_f1_10 |
|---|---|---|---|---|
| **Cascade** (Stage 1 → Stage 2) | multilabel | **0.229 ± 0.022** | **0.225 ± 0.027** | — |
| Flat multiclass (`results_multiclass_v2`) | multiclass (11) | 0.280 ± 0.022 | — | **0.125 ± 0.008** |
| Stage 1 (binary) | binary | 0.746 † | — | — |

† Stage 1: `positive_class_f1 = 0.791`, seed 2024 only — seeds 42 and 1337 were
not recorded from the console. **Re-read them from `results_stage1/eval_*.json`
before citing**, and report mean ± std like everything else.

### Notes on the cascade number

- **Seed spread is wide**: 0.250 / 0.229 / 0.197 for seeds 42 / 1337 / 2024 —
  ±12% of the mean. Report it as-is rather than smoothing; with 1,278 training
  rows across 10 classes (~128/class, rarest = 101), that instability is itself
  evidence that **data volume is the binding constraint**.
- **Stage 1 is the ceiling.** Any distorted row Stage 1 misses never reaches
  Stage 2 and is permanently wrong, whatever Stage 2's quality.
- Stage 2 in isolation scored **val macro_f1 0.343 / 0.331** (seeds 42 / 1337).
  That figure is *not* comparable — it drops the 92 No-Distortion test rows (36%
  of the test set) and hides every Stage 1 false negative. It is tagged
  `[stage2-isolated]` in `paper_comparison.csv` for exactly that reason.

---

## 2. Leaked results — reference only, NOT a baseline

Everything below was trained on `data/splits_combined`, where **194/253 val rows
(76.7%) and 189/253 test rows (74.7%) also appear in train**. See §4.

| Model | Task | weighted_f1 | macro_f1 | macro_f1_10 |
|---|---|---|---|---|
| mental-roberta-base | multilabel | 0.288 ± 0.009 | 0.279 ± 0.005 | — |
| deberta-v3-base | multilabel | 0.209 ± 0.011 | 0.193 ± 0.012 | — |
| mental-roberta-base | multiclass | 0.338 ± 0.016 | — | 0.188 ± 0.012 |
| deberta-v3-base | multiclass | 0.336 ± 0.013 | — | 0.172 ± 0.005 |

---

## 3. The comparison that is still open

The cascade (0.225) sits **below** the flat multilabel figure (0.279) — but that
flat number is leaked, so the comparison cannot be read in either direction.

**A valid flat multilabel baseline on `data/splits` does not yet exist.** It is
~5 minutes of GPU time and it is the single missing piece for a defensible
"is the cascade better?" claim.

```python
for seed in SEEDS:
    run_and_report("multilabel", PARENT_SPLITS, "results_multilabel_flat", seed,
                   ckpt_dir="/kaggle/working/checkpoints_flat",
                   extra_flags="--max-length 256 --truncation head_tail --batch-size 32")
```

`ckpt_dir` is **mandatory** — checkpoints are named `{task}_{TAG}_{seed}`, so
without it this overwrites Stage 2.

> **Do not extrapolate.** Applying the multiclass deflation factor to the flat
> multilabel number gives ~0.186, which would put the cascade ahead. That is
> arithmetic on an assumption and must not appear in the thesis as evidence.

---

## 4. Leakage, quantified — a result in its own right

Same model, same metric, leaked splits vs clean:

| Metric | Leaked | Clean | Relative inflation |
|---|---|---|---|
| multiclass `macro_f1_10` | 0.188 | 0.125 | **+50%** |
| multiclass `weighted_f1` | 0.338 | 0.280 | **+21%** |

**Macro inflates ~2.4× more than weighted.** Memorising a rare test example lifts
macro-F1 far more than weighted-F1, so leakage flatters precisely the metric that
matters most for the rare distortion classes — the ones this project is trying to
improve.

**Caveat to state:** the clean run also trained on fewer rows (2,024 vs 4,645), so
these figures mix the leakage effect with reduced training data. They are an
upper bound on the leakage contribution, not a clean isolation of it.

### Root cause

`make_splits_combined.py` is written correctly — its docstring says val/test stay
frozen and the code does that. The bug is its *input assumption*: it treats
CODIPAS as a disjoint dataset. It isn't. **1,937 of CODIPAS's 2,621 rows (74%)
are already in `Annotated_data.csv`** — 1,554 in train, 194 in val, 189 in test.
Only 684 rows are genuinely new.

---

## 5. Context for the write-up

From `CLAUDE.md`:

- Paper headline binary **F1 = 0.79** (SVM + S-BERT). Your Stage 1 is **0.791**
  positive-class / 0.746 weighted. **Confirm which variant the paper reports**
  before claiming a match — it decides which of your two numbers is comparable.
- Leakage-fixed classical replication: best binary test F1 **0.69**. Stage 1 at
  0.79 clears it.
- Paper multi-class: all classifiers **below F1 = 0.30**; leakage-fixed classical
  best **0.30 test / 0.27 val**.
- **~33.7% inter-annotator agreement on distortion type** — the real ceiling.
  Also flagged as unverified against the source; check before citing.

**On targets.** A macro-F1 of 0.70 on the 10-class task is not a realistic goal:
humans agree with each other about a third of the time on *which* distortion is
present, so a model exceeding that is far more likely to indicate a bug than a
breakthrough. This run has already demonstrated the mechanism — leakage alone
moved macro_f1_10 from 0.125 to 0.188. The realistic ceiling from here is
~0.30–0.35, and the honest levers are: **more training data** (the synthetic
pipeline), and **merging classes the confusion structure shows are not
separable** — which doubles as the entrepreneurial-taxonomy contribution.

The binary task already provides a headline above 0.70.

---

## 6. Known gaps

- [ ] Flat multilabel baseline on `data/splits` (§3)
- [ ] Stage 1 `positive_class_f1` for seeds 42 and 1337
- [ ] Stage 2 isolated **test** macro_f1 (only val recorded)
- [ ] Confirm the paper's binary F1 variant, and the 33.7% agreement figure
- [ ] Per-class breakdown — which of the 10 distortions the cascade fails on
      (`per_class_cascade_multilabel_*.csv` has it, unread)
