# S07 — Synthetic leak + overlap checks

**Run:** 20 Nov 2026 with `python -m src.synth_checks`, on `data/synthetic/synthetic_train_en.jsonl` (258 rows from
S06).

**Outputs:**

| File | What it holds |
|---|---|
| `data/synthetic/synthetic_train_en_clean.jsonl` | the clean copy |
| `data/synthetic/overlap_flags.csv` | flagged overlap pairs (empty: none) |
| `results/results_synth_checks/keyword_scores.csv` | keyword macro-F1 scores |
| `results/results_synth_checks/keyword_hits_per_type.csv` | keyword hit rate per type |
| `results/results_synth_checks/summary.json` | all numbers |

## 1. Keyword check

### The word list

The list was written before looking at the S06 rows (`KEYWORDS` in `src/synth_checks.py`). It uses the brief's
examples plus the lexical markers of FYP_PLAN §3.5. Matching is case-insensitive at word boundaries. Each row gets
the type whose words appear most often; no hits gives No Distortion.

| Type | Give-away words |
|---|---|
| Should statements | should, shouldn't, should have, must, have to, has to, ought, supposed to, need to |
| All-or-nothing | always, never, completely, totally, entirely, total failure, perfect, nothing, everything, all or nothing |
| Mind Reading | they think, he thinks, she thinks, must think, they must, thinks i, think i'm, think i am, judging me, see me as, they'll think |
| Fortune-telling | will never, never going to, going to fail, will fail, won't work, doomed, it's over, probably won't, will be a disaster |
| Overgeneralization | every time, everyone, nobody, no one, all of them, keeps happening, again and again, this always |
| Labeling | i'm a failure, i am a failure, i'm such a, loser, idiot, stupid, worthless, useless, fraud, failure |
| Personalization | my fault, because of me, blame myself, on me, i caused, i let, i'm responsible, i am responsible |
| Emotional Reasoning | i feel, feel like, feels like, i felt, my gut, feeling |
| Magnification | disaster, catastrophe, ruined, end of everything, the end, collapse, terrible, awful, worst, huge |
| Mental filter | only, just one, all i, focus on, can't see past, can't stop thinking |

### Result

The keyword list predicts the panel's labels far better on synthetic text than it predicts the gold labels on
real text.

| Keyword list | Rows | Macro-F1 (11 classes) | Macro-F1 (10 distortions) | Chance macro-F1 (11 / 10) |
|---|---|---|---|---|
| **A-priori list, synthetic rows** | 258 | **0.254** | **0.239** | 0.091 / 0.068 |
| A-priori list, benchmark val rows | 253 | 0.108 | 0.081 | 0.093 / 0.065 |
| Post-hoc list, synthetic rows | 258 | 0.300 | 0.291 | 0.091 / 0.068 |
| Post-hoc list, benchmark val rows | 253 | 0.108 | 0.081 | 0.093 / 0.065 |

- **The "chance" column** is the true labels shuffled, averaged over 100 shuffles, so it keeps the same class mix.
- **On real text the list is barely above chance:** 0.108 against 0.093.
- **On synthetic text it scores 2.3 times higher:** 0.254, almost 3 times chance.
- **The post-hoc list is not the headline.** It adds the S06 phrases (fixate / ignore / replay / "not cut out" / "fall
  apart" …) and is fitted to what we saw. It pushes the synthetic score to 0.300 and changes nothing on val. Those
  phrases simply don't occur in real text.

**Rows whose own type's give-away words appear** (a-priori list):

| Type | Synthetic | Benchmark val |
|---|---|---|
| Personalization | **76%** (n=21) | 13% (n=16) |
| Overgeneralization | **70%** (20) | 17% (24) |
| Mental filter | **45%** (11) | 8% (13) |
| Mind Reading | 33% (21) | 15% (20) |
| Should statements | 79% (19) | 64% (11) |
| Emotional Reasoning | 94% (17) | 85% (13) |
| All-or-nothing | 75% (16) | 80% (10) |
| Fortune-telling | 19% (21) | 13% (16) |
| Labeling | 19% (21) | 19% (16) |
| Magnification | 14% (7) | 16% (19) |

**Openers.**

| | Synthetic | Benchmark val |
|---|---|---|
| Distinct first words | 12 of 258 | 46 of 253 |
| Distinct 2-word starts | 54 | 114 |
| Share of rows on the top-3 starts | 47% ("when my", "last week", "our first") | 32% |

### What it means

**The brief's warning applies.** The list scores well above chance on synthetic rows and close to chance on real
ones, so the generator often writes the obvious word.

The problem is concentrated in three types:
- **Personalization:** "my fault", "on me", "because of me".
- **Overgeneralization:** "everyone", "no one", "every time".
- **Mental filter:** "only", "can't stop thinking", plus the post-hoc "fixated / ignoring / replayed".

Their synthetic rows carry the cue words 4 to 6 times as often as real rows of the same type. Should statements,
Emotional Reasoning and All-or-nothing are lexical in real text too ("should", "I feel", "always/never"), so their
high rates are not a synthetic artefact.

**For S08:** a model trained on these rows can learn "spot the word" for those three types. Two things to report:
1. Per-class F1 on the supplementary-evaluation rows, not just the overall score. If synthetic data helps
   Personalization or Overgeneralization on val but not on the scraped evaluation rows, that is this leak.
2. If S08 shows harm, a re-generation should first:
   - ban the cue words for the structural types (as `STRUCTURAL_MARKER_BANS` in `src/synth/attributes.py` already
     does for some);
   - use a smaller `--chunk`, so the opener nudge updates more often.

**One source of a repeated phrase is the seed file itself.** seed_001 reads "I guess I'm just not cut out to run a
company", and the generator copied that clause into 12 rows.

## 2. Overlap check (word Jaccard > 0.6)

Each synthetic row was compared with every text in each set.

| Set | Texts compared | Synthetic rows > 0.6 | Row IDs | Highest Jaccard seen |
|---|---|---|---|---|
| Generator seeds (`entrepreneurial_seeds.jsonl`) | 44 | **0** | — | 0.279 |
| Supplementary-evaluation rows (S03, role = evaluation) | 376 | **0** | — | 0.213 |
| Benchmark test split (`data/splits/test.csv`, texts only) | 253 | **0** | — | 0.235 |
| Survey responses | — | **not checked yet** | — | — |

**No synthetic row is a near-copy of anything:** the closest pair in any set is 0.28, well under 0.6. The clean
copy, `synthetic_train_en_clean.jsonl`, therefore holds **all 258 rows**, with the same counts per type as S06:

| Type | Rows |
|---|---|
| No Distortion | 84 |
| Mind Reading | 21 |
| Personalization | 21 |
| Fortune-telling | 21 |
| Labeling | 21 |
| Overgeneralization | 20 |
| Should statements | 19 |
| Emotional Reasoning | 17 |
| All-or-nothing | 16 |
| Mental filter | 11 |
| Magnification | 7 |

**Survey.** The survey responses are participant data and are not in the repo (D04). The check is ready:
`python -m src.synth_checks --survey <export.csv>`. It reads the free-text answers of 10+ words and reports only
anonymous row/question numbers. The survey CSV is never committed. This section is updated when the export is
available.

**test.csv exception.** CLAUDE.md reserves `data/splits/test.csv` for `src/evaluate.py`. The brief asks for this
overlap check, and the team agreed to it:
- the script reads only the `Id_Number` and text columns;
- it never reads test labels;
- it can only remove synthetic *training* rows that copy a test row;
- it never selects a model, threshold or setting.
