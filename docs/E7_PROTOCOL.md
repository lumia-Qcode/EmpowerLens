# Experiment 7 — flat vs cascade: the protocol contract

**Owner: Nayab.** Both arms — flat *and* cascade — are run by **one person, in
one session, on one GPU**. This document is the contract; if the two arms differ
in anything listed here, the comparison is not valid and the numbers cannot be
put side by side.

> **Why one person and one session.** The gap being measured is tiny — the last
> (confounded) comparison gave **+0.003**. Determinism pins the *algorithm
> choice*, which makes a re-run on the **same** machine reproducible; it does not
> make two different GPUs agree. A T4 and a P100 select different kernels and
> different mixed-precision paths (`fp16=True` on CUDA), and Kaggle hands out
> both. Splitting the arms across sessions would leave architecture and hardware
> varying together, with no way to attribute a 0.003 difference to either.
>
> Izza's E6 run produces a flat multilabel number too. Treat it as a
> **replication check**, not as the comparator. If it lands close to yours that
> is reassuring; if it diverges, that is itself worth reporting as evidence of
> cross-machine variance.

## Why the previous comparison did not count

The existing result — cascade 0.240 ± 0.012 vs flat 0.237 ± 0.030 — came from two
different runs with **different truncation budgets**:

| | max_length | test rows truncated |
|---|---|---|
| cascade track (E7) | 256 | **24.9%** |
| experiment suite (E6) | 512 | 2.4% |

A quarter of the cascade's evaluation set was losing content. Part of any gap
between those runs is truncation, not architecture. `head_tail` truncation softens
it — it keeps the first 128 tokens and the last N, and a reflection's conclusion
often carries the distortion — but it does not remove it.

Second problem: `data/splits_stage2` was derived from `data/splits_combined`,
which contains **396 rows that also appear in the Annotated val/test** (77% of the
test set). Stage 2 inherits that leak.

## The contract

Every number below is fixed. Do not vary any of it between the two arms.

| setting | value | why |
|---|---|---|
| backbone | `mental/mental-roberta-base` | matches the rest of the suite |
| splits | `data/splits` | the yardstick; leak-free (verified 0) |
| Stage 2 splits | **regenerate** from `data/splits` — see below | the shipped `splits_stage2` came from the leaked dir |
| `--max-length` | **512** | 2.4% truncation instead of 24.9% |
| `--truncation` | `head_tail` | keeps the conclusion, where distortions often sit |
| `--head-keep` | 128 | the project default |
| `--epochs` | **8** | 4 was never validated; 8 lets the peak appear |
| `--batch-size` | 16 | project default |
| `--lr` | 2e-5 | project default |
| seeds | `42, 1337, 2024` | project convention, 3 seeds |
| determinism | **on** | see below |
| `--max-labels` | 2 | no row in the corpus carries more than 2 |

### Regenerate the Stage 2 splits first

```bash
python -m src.make_splits_cascade --source data/splits --out data/splits_stage2_annotated
python -m src.make_splits_clean --check --source data/splits_stage2_annotated --dest /tmp/x
```

The second command must report **0 leaked rows**. `make_splits_cascade` only
filters an existing dir to `y_bin == 1`; it never re-splits, so the train/val/test
boundary is inherited unchanged from `data/splits`.

### Determinism must be on

Same-seed runs currently differ by up to **0.047 macro-F1** — larger than the
seed-to-seed SD being reported. Until this is on, a flat-vs-cascade gap smaller
than ~0.05 is unreadable, and the real gap is expected to be small.

```python
from src.determinism import enable_determinism
enable_determinism()      # call BEFORE importing torch-dependent training code
```

Verify once before the real runs: train one config twice at the same seed and
confirm the test macro-F1 gap is exactly 0.000.

## What to report

Three separate numbers, never merged:

1. **Stage 1 (binary)** — `positive_class_f1` on test. Reference point: the paper
   reports 0.79 and the previous run matched it at 0.794 ± 0.012.
2. **Stage 2 (multilabel, isolated)** — `macro_f1` on the distorted-only test set.
   **This is not a cascade result.** It is measured only on rows already known to
   be distorted, so it hides every Stage 1 false negative. `src/evaluate.py`
   refuses distorted-only splits without `--allow-distorted-only` precisely to
   stop this number being quoted as end-to-end.
3. **End-to-end** — `macro_f1` on the full test set, Stage 1 gating Stage 2. This
   is the only number comparable to the flat model.

Plus, for both arms: **per-seed results as well as mean ± SD**, and per-label F1.

## The comparison that counts

| system | metric | test |
|---|---|---|
| Flat multilabel | macro_f1 | ? |
| Cascade end-to-end | macro_f1 | ? |

Same backbone, same splits, same preprocessing, same budget, same seeds, same
determinism setting. If those all match, the difference is attributable to the
architecture. If any differ, it is not.

**Prior expectation:** the previous (confounded) comparison gave +0.003 in the
cascade's favour, well inside the noise, and a replication on CODIPAS gave
cascade 0.259 ± 0.007 vs flat 0.270 ± 0.027 — overlapping, if anything favouring
flat. So the likely finding is **"indistinguishable at this data scale"**, and
with determinism on that becomes a statement you can actually defend rather than
one the noise floor swallows.

## Commands

```bash
# Stage 1 — binary, on the full splits dir
python -m src.train_transformer --task binary --splits data/splits \
    --model mental/mental-roberta-base --seed 42 --epochs 8 \
    --max-length 512 --truncation head_tail --batch-size 16

# Stage 2 — multilabel, on the distorted-only dir
python -m src.train_transformer --task multilabel --splits data/splits_stage2_annotated \
    --model mental/mental-roberta-base --seed 42 --epochs 8 \
    --max-length 512 --truncation head_tail --batch-size 16

# End-to-end
python -m src.evaluate_cascade \
    --stage1-checkpoint checkpoints/binary_mental-roberta-base_42 \
    --stage2-checkpoint checkpoints/multilabel_mental-roberta-base_42 \
    --splits data/splits --out results_rerun/exp7 --max-labels 2

# Flat, the comparator — identical settings, SAME session, SAME GPU.
# Not optional: Izza's copy of this was produced on different hardware.
python -m src.train_transformer --task multilabel --splits data/splits \
    --model mental/mental-roberta-base --seed 42 --epochs 8 \
    --max-length 512 --truncation head_tail --batch-size 16
python -m src.evaluate --checkpoint checkpoints/multilabel_mental-roberta-base_42 \
    --splits data/splits --out results_rerun/exp7 --max-labels 2
```

Repeat for seeds 1337 and 2024 — **9 training runs total** (3 seeds × {Stage 1
binary, Stage 2 multilabel, flat multilabel}), plus 3 cascade evaluations that
train nothing. Budget roughly **2–2.5 hours** on a T4.

## Record these alongside the results

A discrepancy that surfaces later is unattributable without them:

```python
import torch, transformers, subprocess
print("gpu         :", torch.cuda.get_device_name(0))
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("commit      :", subprocess.run(["git", "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip())
```

## Reading the result

Report the gap **against the noise floor**, never on its own:

    flat macro_f1 − cascade end-to-end macro_f1    vs    the seed SD of each

If the gap is smaller than the SDs, the honest finding is **"indistinguishable at
this data scale"** — a result, not a failure. The previous comparison gave +0.003
against SDs of 0.030 and 0.012, and a CODIPAS replication gave cascade
0.259 ± 0.007 vs flat 0.270 ± 0.027 — overlapping, if anything favouring flat.
With determinism on and both arms on one machine, that conclusion becomes
defensible rather than something the noise swallows.
