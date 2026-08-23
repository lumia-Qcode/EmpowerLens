# Re-run plan — Experiments 1–8

Plain-language plan for re-running the experiment suite. Every number quoted was
measured on the data currently in this repo, not estimated.

**The one-sentence version:** the old suite trained on data containing 77% of its
own test set, scored each experiment on a different test set, and could not
reproduce its own results run-to-run — so almost none of its numbers can be
compared to each other. This fixes all three.

---

## Part 1 — What was wrong

### 1. The model was shown the answers

`data/splits_combined/train.csv` contains **396 rows whose text also appears in
the Annotated val or test set** — covering **195 of the 253 test rows, 77% of the
test set**. Every Experiment 3–8 cell pointed at `--splits data/splits_combined`.

Worse than ordinary leakage: on those 195 shared texts the two corpora give the
**same label only 36% of the time**, so the model mostly saw the test passage
with the *wrong* answer attached. That can push a score down as easily as up —
the numbers are wrong by an unknown amount in an unknown direction.

`data/splits_codipas_cls` carries the same 396 rows. `data/splits_stage2` was
derived from the leaked `splits_combined` by default, so it inherits the leak too.

### 2. Duplicate rows silently re-weighted training

4,645 train rows, only **3,019 unique texts** — 1,626 duplicates. A text
appearing three times counts three times in the loss. Nobody chose that.

### 3. Every experiment sat a different exam

E2 scored each corpus on **its own** test set. E3–E8 scored on **Combined**.
Month-1 scored on **Annotated**. Three different exams, so a higher number in one
says nothing about another.

For E2 this is fatal to the conclusion. The recorded test truncation rates differ
per arm (0.0237 / 0.0122 / 0.0383), which is only possible with different test
sets. CODIPAS's labels come from an aggregation rule, and rule-generated labels
are more self-consistent — hence easier to predict — regardless of data volume.
So E2's ranking may be measuring **which test set is easiest**.

### 4. Results did not reproduce run-to-run

E6 and E7 were the *same* configuration — same backbone, splits, loss, seeds,
only the output directory differed. They disagree:

| seed | E6 | E7 | gap |
|---|---|---|---|
| 42 | 0.316 | 0.363 | **0.047** |
| 1337 | 0.307 | 0.315 | 0.008 |
| 2024 | 0.302 | 0.275 | 0.027 |

Cause: a GPU adds thousands of numbers in whatever order its threads finish, and
floating-point addition is order-dependent. Adding the same 100,000 floats in
three orders gives `-90.825134` / `-90.825070` / `-90.825070`. One step's
difference is ~1e-5; over ~500 steps/epoch it compounds.
`torch.use_deterministic_algorithms` is set nowhere in the repo.

**Consequence:** run-to-run variance at a *fixed* seed (0.047) exceeds the
seed-to-seed SD being reported (0.007). Every ± figure in the supervisor doc
understates uncertainty, and any difference under ~0.05 is currently unreadable.

### 5. Nobody ever checked the epoch budget

`--epochs 4` is a default someone typed. There is no sweep, no justification, and
**no past run saved a per-epoch curve** — so it cannot even be checked
retrospectively. Every recorded run stopped at `epoch=4.0`, i.e. used the whole
budget.

With `load_best_model_at_end=True`, "4 epochs" means *"best of the first 4."* If
the true peak is later, it is never seen. The DistilBERT replication is exactly
that case: epoch 4 scored **0.0000**, epoch 8 scored **0.0561**. A 4-epoch budget
would have reported zero.

### 6. E1's model-based half was never run

`--checkpoint-mc` / `--checkpoint-ml` were commented out, so no per-class
precision/recall/F1 and no confusion matrices came from E1.

### 7. Scattered and empty output

Results split across `results/expN` and `result_experiment/expN`. Of
`results_experiments/`'s 84 files, **34 are 0 bytes** — including every top-level
`per_class_*.csv`. The rest are byte-identical copies of `results/`.

### 8. Two copies of the script, wrong branch

`experiments_flat_mentalroberta.py` exists byte-identical at `src/` and
`experiments/`. The old notebook cloned `lumia-space` and asserted `src/`.

---

## Part 2 — What we change

### Fix 1: clean splits, frozen originals

`python -m src.make_splits_clean` writes **new** directories; the originals are
never edited so old results stay traceable. Drops any train row whose normalized
text appears in the yardstick's val/test, collapses duplicates, copies val/test
byte-identical, then re-audits itself and fails if anything survives.

| directory | train before | after | leaked |
|---|---|---|---|
| `data/splits_combined_clean` | 4,645 | **2,623** | 396 → **0** |
| `data/splits_codipas_clean` | 2,621 | **2,224** | 396 → **0** |

Matching is on normalized text, **not `Id_Number`** — the same passage carries
different ids across corpora, which is how the leak got in.

### Fix 2: every run scored on two exams

`src/eval_two_exams.py` scores each checkpoint twice:

| exam | test set | answers |
|---|---|---|
| **home** | the test set of what it trained on | within-dataset performance |
| **yardstick** | always `data/splits/test.csv`, the same 253 rows | comparable across every experiment |

`transfer_gap = home − yardstick`. The script refuses to run if the training data
contaminates the yardstick.

### Fix 3: determinism, forced and verified

`src/determinism.py` sets `torch.use_deterministic_algorithms(True,
warn_only=True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False` and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. Verified locally: two runs at the same seed
gave bit-identical loss and gradients.

`warn_only=True` means an op without a deterministic kernel warns instead of
crashing — so the run completes, but reproducibility is *claimed* rather than
guaranteed. Hence a **verification run**: one config trained twice at the same
seed, asserting the gap is exactly 0.000. Until that passes, the error bars are
not trustworthy and the notebook says so.

Cost: roughly 10–30% slower training.

### Fix 4: every run saves its epoch curve

The experiments script gains `epoch_history_*.csv` (the DistilBERT runner already
does this). Budget moves to **8 epochs** so the peak has room to appear.

Then the budget question answers itself: if the best epoch clusters at 7–8, the
budget is still too small; if it clusters at 2–4, four was fine and we can say so
with evidence rather than assumption.

### Fix 5: metrics that cannot flatter

Two changes, both because a metric that hides a failure is worse than no metric.

**Multiclass now selects on `macro_f1_10`, not `macro_f1`.** Plain macro-F1 over
11 classes averages in `no_distortion` — 36.9% of rows and by far the easiest
class — so selecting on it partly rewards the model for the one thing that was
never hard. Measured on a real run: `macro_f1` 0.053 against `macro_f1_10`
0.008, six times apart. `train_transformer.py` already did this; the experiments
script did not.

**`roc_auc` is recorded as a diagnostic column.** F1 cannot distinguish "the
model never learned this class" from "it ranks the class correctly but the
threshold is wrong", and those need opposite fixes. AUC ignores the threshold, so
a low F1 beside a high AUC localises the fault to calibration. Already decisive
once in this project: a DistilBERT run scored macro-F1 **0.029** with ROC-AUC
**0.709** — the model had learned, it just never crossed 0.5.

It is never a headline. The headline stays macro-F1, which averages the ten
labels equally so a class the model never predicts drags it down. Micro-F1 pools
every label into one count and lets frequent labels dominate: on this val set, a
model that perfectly predicts the 3 commonest labels and ignores the other 7
scores **micro 0.582, macro 0.300**.

### Fix 6: one script, one branch, one results root

Single copy at `experiments/experiments_flat_mentalroberta.py` (the `src/` twin
is deleted). Notebook clones `nayab-space` and checks the file exists first.
Everything writes under `results_rerun/expN/`.

---

## Part 3 — Experiment by experiment

| | question | trains on | reported on | change from before |
|---|---|---|---|---|
| **E1** | what is the data like? | — | — | run the model half too |
| **E2** | does CODIPAS help, hurt, or nothing? | 4 arms | **yardstick + home** | new matched arm; one exam |
| **E3** | weighted CE vs plain CE? | Annotated | yardstick | moved off leaked splits |
| **E4** | focal / class-balanced vs E3's winner? | Annotated | yardstick | moved off leaked splits |
| **E5** | weighted sampling vs E3/E4's winner? | Annotated | yardstick | moved off leaked splits |
| **E6** | which labels fail; do per-label thresholds help? | Annotated | yardstick | 512 tokens, 8 epochs |
| **E7** | flat vs cascade | — | — | **owned by Izza** — see `docs/E7_PROTOCOL.md` |
| **E8** | are we losing information to truncation? | — | — | already correct |

### E1 runs twice

The audit has a data half and a model half, and the model half needs checkpoints
that do not exist yet.

1. **Before any training** — class frequencies, multilabel prevalence per split,
   label co-occurrence (counts + Jaccard), train/val/test distributions. This is
   the gate that decides whether E3–E5 are worth GPU time.
2. **After E6** — re-run with `--checkpoint-mc` and `--checkpoint-ml` for
   per-class precision/recall/F1 and confusion matrices. This is the step that
   was commented out before.

### E2 gets a fourth arm, and it is the important one

Already built and verified clean:

```
data/splits_codipas_transfer_matched   train=2,024   test == Annotated test   leak=0
```

Downsampled to exactly 2,024 rows — the same as Annotated — so **volume is not a
second confound**, and its test set *is* the yardstick. This is the arm that
actually isolates the label-convention effect.

**Five arms**, all scored on the yardstick:

| arm | train rows | labels |
|---|---|---|
| `annotated_only` | 2,024 | human |
| `codipas_clean` | 2,224 | rule |
| `codipas_matched` | **2,024** | rule |
| `combined_clean` | 2,623 | mixed |
| `combined_matched` | **2,024** | mixed |

The two matched arms are what make E2 answerable. Comparing `annotated_only`
(2,024, human) against `combined_clean` (2,623, mixed) varies **volume and label
convention at once**, so whichever way it lands it cannot say which caused it.
Holding volume fixed splits that into two clean questions:

* `annotated_only` vs `combined_matched` → **label convention**, volume fixed
* `combined_matched` vs `combined_clean` → **volume**, convention fixed
* `codipas_matched` vs `codipas_clean` → volume, rule labels only

`combined_matched` is built by `make_splits_clean --match-to`, stratified on
`y_mc` with a fixed seed so the class balance is preserved (verified to within
0.1%) and the draw is reproducible. An unstratified sample would add a third
confound.

**Per-seed results are reported alongside mean ± SD**, as specified. The script
already writes `exp2_all_seed_results.csv`; the old notebook only printed the
aggregate.

**Expected outcome: merging hurts the fine-grained task.**
`docs/codipas_agreement.md` measured the two label sets over 2,520 shared texts —
binary agreement 66.5% (κ = 0.321), 11-class 36.8% (κ = **0.199**), and among rows
*both* call distorted they disagree on which one **73%** of the time. Per class,
CODIPAS reproduces the human label as rarely as 5.7%. Two schemes agreeing at
κ = 0.199 are not labelling the same thing. **A clear negative result on a fixed
test set is a real finding** and more defensible than a marginal win.

### E7 is Izza's, and needs a contract

Both arms — flat and cascade — will be run by Izza so they share one protocol.
`docs/E7_PROTOCOL.md` pins the exact configuration both sides must use.

The old comparison was not valid: cascade ran at `max_length 256` where **24.9%
of test rows truncate**, against E6's 512 (2.4%). Part of the 0.237-vs-0.308 gap
was truncation, not architecture.

Required outputs: Stage 1 binary, Stage 2 multilabel *in isolation*, and
end-to-end — reported separately. Stage 2's isolated number must never be quoted
as a cascade result; it is measured only on rows already known to be distorted,
so it hides every Stage 1 false negative.

### E8 needs no change

The script already emits median, p90, p95, max and percent-truncated at each
`max_length`. It fully meets the spec.

---

## Part 4 — Order to run

1. **Preflight** — `make_splits_clean --check`. Must report no leaks.
2. **Determinism verification** — one config twice at the same seed; gap must be
   0.000. Everything after this is only as trustworthy as this step.
3. **E1 pass 1** — the data audit. No GPU. Gates E3–E5.
4. **E8** — truncation analysis. No training. Confirms 512 is the right budget.
5. **E6** — per-label + thresholds on Annotated. Produces the checkpoints E1
   pass 2 needs.
6. **E1 pass 2** — per-class tables and confusion matrices.
7. **E2** — the four-arm ablation. Heaviest: 4 arms × 3 seeds.
8. **E3 → E4 → E5** — the imbalance chain, in order; each reads the previous
   winner.
9. **E7** — Izza, per the protocol contract.

### Budget

39 training runs: E6 (3), E2 (15), E3 (6), E4 (9), E5 (6). At roughly 12–14 min
per run on a T4 — 8 epochs × 127 steps at 512 tokens, plus ~10–30% determinism
overhead — that is **about 8–9 hours**, over Kaggle's 9-hour session cap. Two
sessions: preflight→E2 (~4 hr), then E3→E5 and the tables (~4 hr).

**Estimated, not measured.** Time the first E6 run and scale from it.

If GPU time is short, cut E4 and E5 first — both are conditional on E3's outcome,
and E1's audit says upfront whether the imbalance justifies the chain at all. Do
not cut E6 (it produces the checkpoints E1 pass 2 needs) or E2's first three arms.

---

## Part 5 — What "comparable" will mean

Two numbers may sit in the same column only if both are:

- from the **yardstick** exam (`data/splits/test.csv`, the same 253 rows),
- on the **same task** (2, 11 and 10 classes are three different exams),
- marked `leaked = False`,
- and produced **after** the determinism verification passed.

All four are recorded per row, so this is checkable rather than remembered.

**Never comparable, by design:** home-exam scores from different corpora; val
against test; anything from `results_experiments/` or `results_combined/`, which
came from the contaminated splits.
