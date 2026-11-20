# S06 — Synthetic generator: full run (300 specs)

**Run:** 20 Nov 2026, Kaggle T4 ×2, `notebooks/kaggle_runner_synth.ipynb` with `MODE = "full"`.
**Command:** `python -m src.synth.batch_generate --n 300 --chunk 75 --out .../synthetic_train_en.jsonl`.
**Code:** commit `229d1d8` on `lumia-space`, the same code and models as the S05 calibration
(`docs/synth_calibration.md`).

**Outputs** (all in `data/synthetic/`):

| File | What it holds |
|---|---|
| `synthetic_train_en.jsonl` | the training rows |
| `synthetic_train_en.rejected.jsonl` | every abandoned spec, with what the panel read instead |
| `synthetic_train_en.summary.json` | all counts |
| `synthetic_train_en.run.log` | the log; the vLLM weight-loading and progress-bar lines are removed |

**Models**, four families so no model judges its own family's writing:

| Role | Model |
|---|---|
| Generator | `Qwen/Qwen3-14B-AWQ`, used for all rows. The default `Qwen3-32B-AWQ` ran out of GPU memory and fell back each time (bug 1 in the S05 doc), exactly as in the calibration. |
| Critic 1 | `casperhansen/mistral-small-24b-instruct-2501-awq` (temperature 0.3) |
| Critic 2 | `stelterlab/phi-4-AWQ` (0.6) |
| Critic 3 | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (0.9) |

All three critics voted on every accepted row.

## Results

| | Count |
|---|---|
| Rows requested (planner specs) | 300 |
| **Accepted** | **258** (86%) |
| Revised once | 80 |
| Rejected by the critic panel (after a revision) | 41 |
| Dropped by a filter | 1 (length) |
| Critics said "can't judge" (Unknown) | 0 |

**Cost:** $0, with no API calls. The brief estimated PKR 920–1,110 for the paid-API version. The run took 125 min of
Kaggle GPU time (about 0.5 GPU-minutes per accepted row):
- generating: 53 min;
- critics: 43 min;
- revision round: 30 min.

About 25 min of that went on the five failed 32B load attempts (about 5 min each in the S05 log).

### Accepted rows per distortion type

| Type | Accepted | Attempted | Abandoned |
|---|---|---|---|
| No Distortion | 84 | 90 | 7% |
| Mind Reading | 21 | 21 | 0% |
| Personalization | 21 | 21 | 0% |
| Fortune-telling | 21 | 21 | 0% |
| Labeling | 21 | 21 | 0% |
| Overgeneralization | 20 | 21 | 5% |
| Should statements | 19 | 21 | 10% |
| Emotional Reasoning | 17 | 21 | 19% (one of the 4 was the length drop) |
| All-or-nothing thinking | 16 | 21 | 24% |
| Mental filter | 11 | 21 | 48% |
| **Magnification** | **7** | 21 | **67%** |
| **Total** | **258** | **300** | 14% |

- **Secondary label:** 172 rows (67%) have one. The labels are capped at 2, as in the benchmark.
- **Length:** 184 rows are 30–50-word snippets and 74 are ~150-word reflections.

### What the panel read when it rejected a type (taxonomy signal)

| Asked for | Panel read instead |
|---|---|
| Magnification (14 rejected) | Fortune-telling 6, Mind Reading 3, Labeling 3, Overgeneralization 1, Emotional Reasoning 1 |
| Mental filter (10) | Mind Reading 5, Overgeneralization 2, Labeling 1, Magnification 1, Fortune-telling 1 |
| No Distortion (6) | Labeling 2, Mind Reading 2, Personalization 2 |
| All-or-nothing thinking (5) | Overgeneralization 2, Fortune-telling 2, Personalization 1 |
| Emotional Reasoning (3) | Fortune-telling 2, Mind Reading 1 |
| Should statements (2) | Mind Reading 2 |
| Overgeneralization (1) | Fortune-telling 1 |

**Magnification is the type the panel least often confirms in founder writing.** Asked to blow up a setback, the
generator writes a catastrophic *forecast* ("the whole business will collapse"), and the critics call that
Fortune-telling. Even 4 of the 7 accepted Magnification rows got 3 Fortune-telling votes against 2 for
Magnification. This fits the benchmark, where Magnification and Fortune-telling are often confused. It's the
clearest case for the entrepreneurial-taxonomy argument.

### Calibrated pessimism

- 84 of 90 No-Distortion specs were accepted.
- Only **27 of the 84 had a unanimous 3/3 vote**; 57 passed 2/3.
- The 6 rejected were read as Labeling, Mind Reading or Personalization, 2 each.

So the panel usually accepts sound reasoning about setbacks, but one critic often finds a distortion in it. The
`no_distortion_votes` field keeps that margin on every row.

## Spot-read (Claude's reading; Lumia/Izza to confirm)

**What was read:**
- all 7 accepted Magnification rows;
- all 11 accepted Mental filter rows;
- 2 random accepted rows for each other type;
- 4 Magnification / Mental filter rejects.

**Labels are mostly right.** Most sampled rows clearly show their label, for example:
- syn_00111 Mind Reading: "her silence spoke volumes … she still thinks it's not enough";
- syn_00024 All-or-nothing: "either you're a success or you're a failure";
- syn_00184 Should statements: "a real founder should never rest";
- syn_00260 No Distortion: checks the contract buffer and adjusts the schedule.

**Labeling also covers other people.** syn_00277 labels a supplier ("he's not just unreliable — he's
unprofessional"). That is a valid use of the type, and good variety.

**Magnification rows are really catastrophising forecasts.** syn_00040: "us ending up homeless". syn_00114:
"our entire vision collapsing". They are distorted, but sit on the Fortune-telling border, as the votes show.

**Mental filter rows describe the pattern out loud.** 9 of the 11 tell the reader they are filtering: "I fixated
on…", "ignoring the…", "I replayed that comment", "the only thing that matters". A classifier could learn those
words instead of the thinking, which is exactly the leak S07's keyword check is meant to catch.

**A few weak rows.** For example, syn_00170 (Labeling): "I guess I'm just not cut out for this industry" is
self-doubt more than a fixed label. It's the same phrase flagged in S05.

**Seeds and rejects.** No row was edited. Rejected specs were thrown away, as the brief requires; they're kept in
`.rejected.jsonl` only as evidence.

## Diversity: weaker than the calibration suggested (flag for S07)

**Openers have collapsed:**
- only **12 distinct first words in 258 rows**;
- 54 distinct two-word starts;
- the most common: "When my" ×52, "Last week" ×37, "Our first" ×33.

The opener nudge in the batch runner only updates between chunks of 75 rows, so within a chunk the generator falls
back to the same openings.

**Repeated phrases** (number of rows):

| Phrase | Rows |
|---|---|
| "I keep thinking" | 19 |
| "I feel like" | 19 |
| "not cut out for this" | 12 |

**What S07 should do:**
- add these phrases and the Mental filter words (*fixate, ignore, replay, only thing that matters*) to the keyword
  check;
- report how predictable the openers are.

**If the keyword check scores much higher on synthetic than on real text,** there are two fixes:
1. re-run with smaller chunks (`--chunk 15`), so the nudge updates five times as often;
2. ban the top 2-word starts outright.

## Field check (S06 "done when")

Every row has its text, label, `critic_votes` and the seeds it was prompted with:

```
$ python -c "import json; rows=[json.loads(l) for l in open('data/synthetic/synthetic_train_en.jsonl',encoding='utf-8')]; print('missing:', sum(not(r['text'] and r['dominant_distortion'] and r['critic_votes'] and r['seed_ids']) for r in rows), 'of', len(rows))"
missing: 0 of 258
```

- **Seeds:** first-attempt rows carry the 4 seed ids shown in their prompt. Revised rows carry the seeds of both
  prompts.
- **Training rows only.** These are training rows, never test rows. No seed is a supplementary-evaluation row (S04
  role check).

## What it means for training

- **Balance.** The data is balanced except for Magnification (7) and Mental filter (11). Those two types get little
  synthetic help; report that alongside any per-type gains.
- **Gate before use.** Use this data in training only after S07's leak and overlap checks pass. The opener collapse
  and the Mental filter wording are the specific risks.
