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

## 1. Valid results — complete, all three seeds

| Model | Task | weighted_f1 | macro_f1 | micro_f1 | macro_f1_10 | positive_class_f1 |
|---|---|---|---|---|---|---|
| **Stage 1** | binary | 0.738 ± 0.012 | 0.720 ± 0.010 | 0.735 ± 0.014 | — | **0.785 ± 0.018** |
| **Cascade** (S1 → S2) | multilabel | **0.229 ± 0.023** | **0.225 ± 0.027** | 0.235 ± 0.025 | — | — |
| Flat multiclass | multiclass (11) | 0.280 ± 0.022 | — | 0.275 ± 0.028 | **0.124 ± 0.008** | — |
| *Stage 2 isolated* ‡ | multilabel | *0.273 ± 0.017* | *0.262 ± 0.023* | *0.278 ± 0.015* | — | — |

‡ **Not a cascade result.** Scored on distorted-only inputs, which removes the 92
No-Distortion test rows (36% of the test set) and hides every Stage 1 false
negative. Tagged `[stage2-isolated]` in `paper_comparison.csv`. Listed here only
to compute the cascade tax below.

### The cascade tax is smaller than expected

```
Stage 2 isolated  macro_f1 = 0.262
Cascade end-to-end macro_f1 = 0.225
                    drop   = 0.037   -> retains 86% of isolated performance
```

Composing the two stages costs only **14%** of Stage 2's standalone quality. That
is because Stage 1 has strong recall on the distorted class
(`positive_class_f1 = 0.785`), so most distorted rows do reach Stage 2 — the
pipeline loses far less than the "Stage 1 is a hard ceiling" framing suggests.

This is a genuinely favourable result for the two-stage design and worth stating
explicitly: **the cascade's weakness is not error propagation from Stage 1.**

### Stage 2 overfits: 22% val→test drop

```
val macro_f1  0.343 / 0.331   (seeds 42 / 1337)   mean 0.337
test macro_f1 0.287 / 0.258 / 0.242               mean 0.262
```

A 22% relative drop on 1,278 training rows across 10 classes (~128 per class,
rarest = 101). Together with the wide seed spread below, this is the clearest
evidence in the run that **training data volume is the binding constraint** — and
therefore the strongest internal argument for the synthetic-data track.

### Seed spread is wide — report it, don't smooth it

Cascade macro_f1 by seed: **0.250 / 0.229 / 0.197** — a range of 0.053 on a mean
of 0.225, i.e. ±12%. Same story on Stage 2 isolated (0.287 / 0.258 / 0.242).
Instability of that size at this data volume is a finding, not noise to hide.

### Stage 1 vs the literature

`positive_class_f1 = 0.785 ± 0.018` against the paper's headline binary
**F1 = 0.79**, and the leakage-fixed classical replication's **0.69**. Stage 1
clears the classical baseline comfortably and effectively matches the paper — on
clean splits, which the paper's own pipeline did not have.

**Confirm which F1 variant the paper reports** before claiming the match: if it is
weighted rather than positive-class, the comparable number is **0.738**, not 0.785.

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

- [ ] **Flat multilabel baseline on `data/splits`** (§3) — the only thing blocking
      a defensible "is the cascade better than flat?" claim. ~5 min.
- [ ] Confirm the paper's binary F1 variant (weighted vs positive-class), and the
      33.7% agreement figure, against the source
- [ ] Per-class breakdown — which of the 10 distortions the cascade fails on.
      `per_class_cascade_multilabel_*.csv` has it and is still unread; it feeds
      directly into the "which classes aren't separable" taxonomy argument
- [x] ~~Stage 1 `positive_class_f1` for seeds 42 and 1337~~ — recorded, §1
- [x] ~~Stage 2 isolated **test** macro_f1~~ — recorded, §1
