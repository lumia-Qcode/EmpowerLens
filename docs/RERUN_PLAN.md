# Re-run plan — what was broken, what we change, and why

Plain-language plan for re-running Experiments 1–8. Every number quoted here was
measured on the data currently in this repo, not estimated.

**The one-sentence version:** the old suite trained on data that contained 77% of
its own test set, and each experiment used a different test set, so no two
results could be compared. We fix the contamination and score every run on one
fixed exam.

---

## Part 1 — What went wrong

### Problem 1: the model was shown the answers

`data/splits_combined/train.csv` contains **396 rows whose text also appears in
the Annotated val or test set**. Those cover **195 of the 253 test rows — 77% of
the test set**.

Every Experiment 3–8 cell in the old notebook pointed at
`--splits data/splits_combined`. So every one of those results was produced by a
model that had already studied three-quarters of its own exam.

It is worse than ordinary leakage. Of those 195 shared texts, the two corpora
give the **same label only 36% of the time**. So the model was shown the test
passage with the *wrong* answer attached in most cases. That can push a score
**down** as easily as up — we cannot even say which direction the numbers are
wrong in, only that they are not trustworthy.

`data/splits_codipas_cls/train.csv` carries the same 396 rows.

### Problem 2: duplicate rows quietly re-weighted the training data

`data/splits_combined/train.csv` has **4,645 rows but only 3,019 unique texts** —
**1,626 duplicates**. A text appearing three times counts three times in the
loss, so the model is pulled towards whatever happened to be duplicated. Nobody
chose that weighting; it is an artefact of how the corpora were concatenated.

### Problem 3: every experiment sat a different exam

- Experiment 2 trained on three corpora and scored each on **its own** test set.
- Experiments 3–8 scored on the **Combined** test set.
- The Month-1 baselines (`roberta-base`, `mental-bert`) scored on the
  **Annotated** test set.

Three different test sets, so a higher number in one experiment says nothing
about another. `SUPERVISOR_EXPERIMENTS.md` already warns about this — *"They are
different exams. A higher number on Combined is not a better model."* — but the
warning does not repair the results, it only labels them unusable for comparison.

### Problem 4: two copies of the experiment script, and the wrong branch

`experiments_flat_mentalroberta.py` exists **twice**, byte-identical, at
`src/` and `experiments/`. The old notebook cloned the **`lumia-space`** branch
and asserted the file was at `src/`. Nothing keeps the two copies in step, and a
fix applied to one silently does not apply to the other.

### Problem 5: results scattered across directories

The old notebook wrote Experiment 1 to `results/exp1`, Experiment 2 to
`result_experiment/exp2`, and the rest to `results/expN`. There are now 84 files
under `results_experiments/` that no longer correspond to any current run.
Nothing tells you which results came from contaminated splits and which did not.

---

## Part 2 — What we change

### Fix 1: build clean splits, never edit the frozen ones

`python -m src.make_splits_clean` writes **new** directories and leaves the
originals untouched, so old results stay traceable to the data that produced
them.

For each source it drops any train row whose text appears in the Annotated
val/test (or its own val/test), collapses duplicate texts, and copies val and
test **byte-identical** — the exam does not change, only what the model may
study. It then re-audits its own output and fails loudly if anything is left.

| directory | train before | train after | leaked rows |
|---|---|---|---|
| `data/splits_combined_clean` | 4,645 | **2,623** | 396 → **0** |
| `data/splits_codipas_clean` | 2,621 | **2,224** | 396 → **0** |

Matching is on normalized text (whitespace collapsed, casefolded), **not on
`Id_Number`** — the same passage carries different ids in different corpora,
which is exactly how the leak got in.

> **Why not the approach on `izza-space`?** `src/generate_clean_splits.py` pools
> train+val+test, dedups, and **re-splits 80/10/10 from scratch**. That does
> remove the leak — but it creates a brand-new test set, and only about **25 of
> the 253 Annotated test rows** survive into it. Every result would again be
> incomparable to the Month-1 baselines and to the other experiments. Her method
> is right for "a clean Combined benchmark in its own right"; ours is right for
> "one fixed exam so everything is comparable." That script also overwrites the
> immutable splits in place and calls `train_test_split` outside
> `make_splits.py`, both of which `CLAUDE.md` forbids.

### Fix 2: every run is scored on two exams

This is the change that makes results comparable. `src/eval_two_exams.py` scores
each checkpoint twice:

| exam | test set | answers |
|---|---|---|
| **home** | the test set of whatever it trained on | *How well did it learn this corpus?* — within-dataset performance |
| **yardstick** | always `data/splits/test.csv` — the same 253 human-annotated rows | *Does it work on our real task?* — comparable across every experiment |

`transfer_gap = home − yardstick`. A big positive gap means the model learned its
own corpus but it did not carry over.

When a model trains on the Annotated data itself, home and yardstick are the same
exam; the script detects that, scores once, and flags the row.

The script **refuses to run** if the training data contaminates the yardstick,
and names the fix. Contamination becomes impossible to record by accident.

### Fix 3: one script, one branch, one results root

- Single copy at `experiments/experiments_flat_mentalroberta.py`.
- The notebook clones **`nayab-space`** and checks the file is present before
  doing anything.
- Everything writes under **`results_rerun/expN/`**. One root, one naming scheme.

### Fix 4: the same protocol everywhere

Same model, same three seeds (42, 1337, 2024), same epochs, same max length,
same selection rule for every experiment. If two experiments differ, the *only*
thing differing should be the thing under test.

---

## Part 3 — What each experiment becomes

| | question it answers | trains on | reported on |
|---|---|---|---|
| **E1** | What is the data actually like? | — | — |
| **E2** | Does adding CODIPAS help, hurt, or do nothing? | Annotated / CODIPAS-clean / Combined-clean | **yardstick + home** |
| **E3** | Does class-weighted CE beat plain CE? | Annotated | yardstick |
| **E4** | Does focal or class-balanced loss beat E3's winner? | Annotated | yardstick |
| **E5** | Does weighted sampling beat E3/E4's winner? | Annotated | yardstick |
| **E6** | Which labels fail, and do per-label thresholds help? | Annotated | yardstick |
| **E7** | The flat multilabel headline result | Annotated | yardstick |
| **E8** | Are we losing information to 512-token truncation? | Annotated | yardstick |

E3–E8 now train on **Annotated** rather than Combined. Two reasons: it is the
corpus whose labels we trust, and it makes home == yardstick, so those
experiments compare directly to each other and to Month 1.

### Experiment 2 deserves a note

We expect combining to **hurt** the fine-grained task, and that is a legitimate
result to report rather than a failure to hide.

`docs/codipas_agreement.md` measured the two label sets directly over the 2,520
shared texts:

| comparison | agreement | Cohen's κ | reading |
|---|---|---|---|
| binary — distorted or not? | 66.5% | 0.321 | fair |
| 11-class — which distortion? | 36.8% | **0.199** | **slight** |
| which type, among rows *both* call distorted | **27.5%** | — | — |

Even where both sources agree a distortion is present, they disagree about which
one **73% of the time**. Per class, CODIPAS reproduces the human label as rarely
as **5.7%** (`mental_filter`) and never above **31%** for any of the ten
distortions. CODIPAS also under-detects distortion roughly **2:1**.

So E2 is not "does more data help" — two schemes agreeing at κ = 0.199 are not
labelling the same thing. E2 asks whether merging them costs you accuracy, and
we predict it does for the 10-class task, less so for binary. **A clear negative
result, measured on a fixed test set, is more defensible than a marginal win**
and is worth stating plainly in the write-up.

---

## Part 4 — Order to run

1. **Preflight** — `make_splits_clean --check`. Must report no leaks. Everything
   after depends on it.
2. **E1** — the audit. No GPU. Tells you whether E3/E4/E5 are worth running.
3. **E7** — the headline flat multilabel result on Annotated. Run this early;
   it is the number most likely to be quoted.
4. **E2** — the dataset ablation. Heaviest experiment, three corpora × 3 seeds.
5. **E3 → E4 → E5** — the imbalance chain. Each reads the previous winner, so
   they must run in order.
6. **E6** — per-label analysis and thresholds.
7. **E8** — truncation check. Cheap; only launches Longformer if the truncation
   rate justifies it.

---

## Part 5 — What "comparable" now means

After the re-run, two numbers from any two experiments can be compared **if and
only if** both are:

- from the **yardstick** exam (`data/splits/test.csv`, the same 253 rows),
- on the **same task** (2, 11 and 10 classes are three different exams),
- and marked `leaked = False`.

The generated results table carries all three fields on every row, so this is
checkable rather than remembered.

Numbers that stay **non-comparable by design**, and should never be put in one
column: home-exam scores from different corpora, val vs test, and anything from
the pre-fix `results_experiments/` runs.
