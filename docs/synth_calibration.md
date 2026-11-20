# S05 — Synthetic generator: 20-row calibration run

**Run:** 20 Nov 2026, Kaggle T4 ×2, `notebooks/kaggle_runner_synth.ipynb` with `MODE = "calibrate"`.
**Command:** `python -m src.synth.batch_generate --n 20 --out .../calibrate.jsonl`.
**Code:** commit `229d1d8` on `lumia-space`.
**Outputs** (all in `data/synthetic/`):
- `calibrate.jsonl` (17 accepted rows)
- `calibrate.rejected.jsonl` (3 rejected rows)
- `calibrate.summary.json`
- `calibrate.run.log` (the vLLM weight-loading lines are removed)

## Setup: why it isn't the setup in the brief

The brief planned Claude (generator) and Gemini (critics) on paid APIs. The team didn't use those keys.

| Attempt | What happened |
|---|---|
| NVIDIA NIM free tier | Worked, but slow: a 2-row trial took ~30 min because of queueing, timeouts and overloads. That puts S06 at 15–30 h. |
| **Open models on Kaggle's GPUs** (this run) | Same pipeline: same planner, prompts, filters, 3 blind critics with a 2-of-3 vote, and one revision. Only one model is on the GPUs at a time. |

The models used, four families so no model judges its own family's writing:

| Role | Model | Temperature |
|---|---|---|
| Generator | `Qwen/Qwen3-14B-AWQ` (Qwen; thinking off) | 0.9 |
| Critic 1 | `casperhansen/mistral-small-24b-instruct-2501-awq` (Mistral) | 0.3 |
| Critic 2 | `stelterlab/phi-4-AWQ` (Microsoft) | 0.6 |
| Critic 3 | `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4` (Meta) | 0.9 |

The planned generator was `Qwen3-32B-AWQ`. It loaded but ran out of memory: see bug 1 below. The pipeline then fell back
to the slot's backup, `Qwen3-14B-AWQ`, and all 20 rows come from the 14B.

## Results

| | Count |
|---|---|
| Requested | 20 |
| **Accepted** | **17** (15 on the first attempt, 2 after one revision) |
| Revised | 5 (2 then accepted: syn_00004, syn_00009; 3 still rejected) |
| Rejected by the critic panel | 3 (spec_00007, spec_00011, spec_00017, each after a revision) |
| Dropped by filters (length, label word, not English, duplicate) | 0 |
| Critics said "can't judge" (Unknown) | 0 |

**Accepted by type:** No Distortion 5, then Labeling, Mind Reading and Fortune-telling 2 each, then Emotional
Reasoning, Magnification, Overgeneralization, Mental filter, Personalization and Should statements 1 each.

**Rejected specs** (asked for X, the panel read Y):
- All-or-nothing thinking → Overgeneralization;
- Magnification → Fortune-telling;
- No Distortion → Personalization.

**Calibrated pessimism.** 5 of the 6 No-Distortion specs were accepted. Only 1 of those 5 had a unanimous 3/3
No-Distortion vote; the other 4 passed 2/3. The panel is quick to find a distortion in sound reasoning.

**Cost:** $0 (no API calls). Kaggle GPU time was **47 min** for 20 rows:
- generating: 13 min;
- critics: 16 min;
- revision round: 18 min.

About two-thirds of that time is loading models (4–6 min per load), not writing. Cost per accepted row is $0, or
about 2.8 GPU-minutes.

## Do the accepted rows show their label? (Claude's reading, confirmed by the team)

Read all 17 accepted rows and all 3 rejected rows.

| Row | Label | Shows it? | Note |
|---|---|---|---|
| syn_00000 | Emotional Reasoning | **No (weak)** | "Feeling like I'm wasting everyone's time … maybe I'm just not cut out for this" reads more as Labeling / Fortune-telling. There's no clear "I feel it, so it's true". |
| syn_00001 | Labeling | Yes | "I wonder if I'm one of those people who can't build anything real … I feel like a failure, and that feeling is like proof I am one." |
| syn_00002 | Mind Reading | Yes | "They must think I'm unprofessional, even though I didn't do anything wrong." |
| syn_00003 | No Distortion | Yes | Loan rejection, reasoned through calmly with a plan. |
| syn_00004 | Magnification (revised) | Borderline yes | Four sales in three days becomes "a sign we were on the wrong path". The span the critics picked is really Mental filter. |
| syn_00005 | Mind Reading | Yes | "I'm sure they think I'm useless", plus Labeling ("I'm a failure"). |
| syn_00006 | Fortune-telling | Yes | "No one will invest, and this idea is doomed before it starts." |
| syn_00008 | Labeling | **No (weak)** | "Maybe I'm just not cut out for this" is self-doubt, not a fixed label. |
| syn_00009 | Overgeneralization (revised) | **No** | "No matter what we do, we can't close the gap" is a forecast. All 3 critics voted Fortune-telling; Overgeneralization got 2. |
| syn_00010 | No Distortion | Yes | Cancelled customer → reads the feedback → plans a fix. |
| syn_00012 | No Distortion | Yes | "One critique doesn't define my whole project." |
| syn_00013 | Mental filter | Yes | "I can't see past that one request. It feels like everything else doesn't matter." |
| syn_00014 | Fortune-telling | Yes | "If this keeps up, the whole thing will fall apart", with clear Mind Reading too. |
| syn_00015 | No Distortion | Yes | "A common hurdle … a realistic next step." |
| syn_00016 | Personalization | Yes | "This is on me." |
| syn_00018 | No Distortion | Yes | Supplier delay: "a setback, but not the end of the project." |
| syn_00019 | Should statements | Yes | "A real founder would have pushed harder", a Should statement without the word *should*. |

**Count: 14 yes, 3 no** (syn_00000, syn_00008, syn_00009).

The 3 rejected rows were rightly rejected:
- spec_00007 reads as Overgeneralization ("if he doesn't support me, no one will").
- spec_00017 reads as Fortune-telling.
- spec_00011 is sound reasoning, so the No-Distortion target was arguably met. The panel's 2/3 Personalization vote
  is the over-eager pattern described under calibrated pessimism.

**Patterns to watch in S06:**
- **A repeated phrase.** "Maybe I'm just not cut out for this" opens or carries 3 of 17 rows (00000, 00001, 00008).
  The word-overlap dedup doesn't catch a repeated clause inside different texts. Only 6 of 17 rows start with
  different first words: almost all start "I …", "The …" or "We …". S07's keyword check should look for this clause.
- **The panel spreads its votes.** Most accepted rows have votes on 4–9 labels. `y_soft` is spread thin, and the
  secondary label is often a weak call.
- **Blended rows.** The weak rows are Emotional Reasoning, Labeling and Overgeneralization rows that blend into
  their neighbours. These three types are worth reading closely in the S06 output.

## Bugs found and fixed

| # | Problem | Fix |
|---|---|---|
| — | NIM free tier too slow for S06 (2 rows in 30 min) | New batch runner on a Kaggle GPU: `src/synth/batch_generate.py` + `notebooks/kaggle_runner_synth.ipynb` (`229d1d8`). |
| — | Rows didn't record the seeds in their prompt, which S06 requires | `seed_ids` on every row (`30b16b7`). Every row in this run has 4 seeds; revised rows have 7–8. |
| 1 | **Not fixed yet.** `Qwen3-32B-AWQ` doesn't fit 2 × T4 at a 6,144-token context with CUDA graphs on: the KV cache left is 0.24 GiB and 0.75 GiB is needed. It fell back to the 14B both times, wasting about 5 min per attempt. | For a re-run: load the 32B with `enforce_eager=True` and a 4,096-token context, or drop it from the default. Left alone for S06 so it uses the same generator as this calibration. |
| 2 | **Not fixed yet.** The Mistral critic's tokenizer warns of "an incorrect regex pattern … incorrect tokenization" (a known issue with this checkpoint). Its votes look sensible, but its input is tokenized slightly wrongly. | For a re-run: load with `fix_mistral_regex=True`, or use a Mistral checkpoint with a fixed tokenizer. |

## Decision for S06: **GO**, with the same setup

- **Rates:** 17 of 20 accepted, 0 unjudgeable, 0 filter drops. 14 of 17 accepted rows show their label on a read.
- **Calibrated pessimism works:** 5 of 6 passed.
- **Same configuration:** the S06 run (`MODE = "full"`, started 20 Nov) uses the same code and models, including the
  14B fallback. These 20 rows are a fair preview of it.
- **After S06:**
  - read the Emotional Reasoning, Labeling and Overgeneralization rows first;
  - run S07's keyword and overlap checks with "not cut out for this" in mind;
  - the yes/no column above was confirmed by the team on 21 Nov 2026, with no changes.
