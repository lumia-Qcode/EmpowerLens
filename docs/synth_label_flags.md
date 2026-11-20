# S09 — Suspect synthetic labels, flagged three ways

**Run:** 20 Nov 2026, on all 258 rows of `data/synthetic/synthetic_train_en.jsonl`. S07 removed nothing, so all
258 are also in the clean file. Two commands:

```
python -m src.cluster_label_errors --only neighbours --input data/synthetic/synthetic_train_en.jsonl --out results/results_synth_neighbours
python -m src.synth_label_flags --models ../empowerlens-demo/models/d12
```

**Outputs:**

| File | What it holds |
|---|---|
| `data/synthetic/flags.csv` | one row per synthetic row: yes/no for each check, plus the scores behind it |
| `data/synthetic/review_sheet.xlsx` | the blind S10 sheet |
| `data/synthetic/review_key.csv` | the key; **not to be opened until S10 is finished** |
| `results/results_synth_label_flags/model_scores.csv` | the model's predictions |
| `results/results_synth_label_flags/summary.json` | all counts |
| `results/results_synth_neighbours/unsupervised_queue.csv` | neighbour scores |

## The three checks

| Check | Rule | Rows flagged (of 258) |
|---|---|---|
| **Neighbours** | 8 or more of a row's 10 nearest rows (TF-IDF + SVD, cosine) carry a different label | **77** (30%) |
| **Model** | The D12 model confidently disagrees with the panel label (rule below) | **52** (20%) |
| **Close votes** | The panel accepted the row 2-to-1 | **132** (51%) |
| Any check | | **187** (72%) |

**Neighbours: the cut.** The brief's literal rule, "most neighbours disagree" (> 0.5), flags 142 rows (55%). Counts
at each cut:

| Cut | Rows flagged |
|---|---|
| > 0.5 | 142 |
| > 0.6 | 114 |
| ≥ 0.7 | 114 |
| ≥ 0.8 | 77 |
| ≥ 0.9 | 44 |

As the brief warned (ARI 0.003–0.016 on the benchmark), neighbourhoods follow topic more than label. Every type
has 21 rows or fewer among 258, so most neighbours of almost any row carry another label. The cut used is ≥ 0.8.
The neighbours are computed among the synthetic rows only, No Distortion rows included.

**Model: the rule.** The model is MentalRoBERTa, D12 winner, seed 42, version `mentalroberta-d12-s42-20261120`. It
was trained on the benchmark only, never on synthetic rows. Predictions use the same code path as `src/evaluate.py`,
run on CPU. A row is flagged when:
- **No Distortion row:** P(distorted) ≥ the D13 binary threshold (0.977). This flagged **0 rows**.
- **Distorted row:** either of these:
  - P(distorted) ≤ 0.10 (4 rows); or
  - the model's top type has probability ≥ 0.5, is neither the panel's main nor second label, and the panel's main
    label is below its own D13 threshold (48 rows).

**Model flags by panel label.**

| Panel label | Model-flagged rows |
|---|---|
| Overgeneralization | 13 |
| All-or-nothing | 10 |
| Labeling | 7 |
| Emotional Reasoning | 7 |
| Should statements | 6 |
| Mental filter | 3 |
| Fortune-telling | 2 |
| Mind Reading | 2 |
| Personalization | 1 |
| Magnification | 1 |

**Close votes: the rule.** For a distorted row, `critic_votes[main label] == 2`. For a No Distortion row,
`no_distortion_votes == 2`. 57 of the 132 close-vote rows are No Distortion: as `docs/synth_run.md` notes, the panel
usually passes sound reasoning only 2 to 1. Of the rest, Should statements has 15, Fortune-telling 12, and the other
types 2–8 each.

### How much the checks agree

| | Rows |
|---|---|
| Flagged by exactly 1 check | 122 (neighbours alone 21, model alone 18, close votes alone 83) |
| Flagged by 2 checks | 56 |
| Flagged by all 3 checks | 9 |
| Neighbours and model | 25 |
| Neighbours and close votes | 40 |
| Model and close votes | 18 |
| Flagged by none | 71 |

The 9 rows flagged by all three are syn_00022, 00026, 00106, 00171, 00203, 00206, 00254, 00264 and 00296. Most are
cases where the model reads a neighbouring type:
- **Fortune-telling** for Magnification, Should statements and Emotional Reasoning rows;
- **Mind Reading** for an Overgeneralization row.

## Review sheet for S10

**What's in it:** `review_sheet.xlsx` holds **217 rows**, shuffled with seed 42:
- all **187 flagged rows** (every one is in the clean file);
- **30 random rows** from the 71 unflagged rows.

**What the "Label" tab shows:**
- only `Review ID` and the text;
- empty columns for main distortion, second distortion, distorted words, Unsure and Notes, with dropdowns;
- a Status column.

It shows no route, no panel label and no flag. The "Start here" tab gives the instructions, and the "Label guide"
tab is the same guide as the S03 workbooks.

**The key:** `review_key.csv` maps `review_id` to the synthetic id, the panel label, `random` yes/no and each
check's flag. Its first line says not to open it until every row is labelled.

**Note for S10:** this is a big sheet. The close-vote check alone puts 132 rows in, because the brief's "2-to-1" rule
catches half the data. S10's precision table will show whether close votes are worth that cost:
- if their precision is near the random rows' error rate, a later run can drop that check;
- or it can use the close-vote flag only for distorted rows.

S10 needs every flagged row labelled to measure precision, so the sheet keeps them all, as the brief says.

## `--input` option (committed)

`src/cluster_label_errors.py` now takes `--input <jsonl|csv>` and `--out <folder>` with `--only neighbours`:
- the input is any file with `id`, `text` and `dominant_distortion`, as in the synthetic schema;
- every row is used, No Distortion included;
- with no `--input`, the script behaves as before: the benchmark run reproduces `unsupervised_queue.csv`
  byte-for-byte.
