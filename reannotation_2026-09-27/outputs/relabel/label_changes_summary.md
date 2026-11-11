# D08 - Errors vs redefinitions, and the label change rate

Source: `merged_labels.csv` (1665 label rows, 1486 distinct re-annotated rows). Script: `scripts/label_changes.py`. Per-row output: `label_changes.csv` (one row per changed label).

**How the groups are decided.** This split is a *rule applied to the pair of labels* (original dominant label, our majority label), not a reading of the original annotators' minds. Our workbooks offered two no-distortion options, plain `No Distortion` and `No Distortion (calibrated pessimism)`, so the option an annotator picked tells us which kind of "not distorted" they meant.

| Original -> our majority | Group |
| --- | --- |
| a distortion -> calibrated pessimism | redefinition |
| a distortion -> plain No Distortion | genuine error |
| No Distortion -> a distortion | genuine error |
| type A -> type B, B = the original secondary | primary swap |
| type A -> any other type B | genuine error |

The majority label is on the 11-way scale (calibrated pessimism counted as No Distortion) and must hold **more than half** of a row's valid votes. When the majority is No Distortion, the row is a redefinition if calibrated-pessimism votes are at least as many as plain No-Distortion votes (a tie goes to redefinition), otherwise a genuine error. Counting the calibrated-pessimism group as redefinition is the cautious choice: it keeps us from claiming more errors than we can show.

**Annotator count.** The 60 overlap rows have 3-4 annotators; every other row, including all 253 test rows, was labelled by **one** annotator, so its "majority" is that one label. `unanimous` is `yes`/`no` only for rows with 2+ valid votes and `n/a` otherwise.

## Count per group

| group         |   majority rows (all) |   of which single-annotator |   multi-annotator majority |   unanimous (2+ annotators, all agree) |
|:--------------|----------------------:|----------------------------:|---------------------------:|---------------------------------------:|
| genuine error |                   826 |                         809 |                         17 |                                      7 |
| redefinition  |                   160 |                         159 |                          1 |                                      0 |
| primary swap  |                    49 |                          47 |                          2 |                                      0 |
| total changed |                  1035 |                        1015 |                         20 |                                      7 |

Rows with a usable majority label: **1322** of 1486 (26 of them with 2+ annotators). Rows with valid votes but no majority (split vote): **34** [8, 60, 129, 246, 320, 328, 414, 663, 692, 733, 783, 787, 795, 799, 884, 1121, 1203, 1286, 1482, 1507, 1618, 1744, 1817, 1919, 1936, 1971, 2129, 2224, 2331, 2365, 2383, 4572, 4604, 4646].

## Change rate on the test split

Flagged = triage buckets A + B; unflagged = bucket C ("agree"), the rows the model did not flag. The unflagged rate shows how often labels change under review anyway, so the flagged rate only means something compared with it.

| Test rows | In split | Usable | Excluded (filled / Unsure / blank) | Label changed (11-way type) | Binary changed (distorted <-> not) |
| --- | --- | --- | --- | --- | --- |
| Flagged (A+B) | 139 | 97 | 42 (35 / 2 / 5) | 72 of 97 (74.2%) | 39 of 97 (40.2%) |
| Unflagged (C) | 114 | 87 | 27 (21 / 3 / 3) | 29 of 87 (33.3%) | 22 of 87 (25.3%) |

Fisher's exact test, flagged vs unflagged: type change p = 2.82e-08; binary change p = 0.0413.

Changed test rows by group:

| bucket    |   genuine error |   primary swap |   redefinition |
|:----------|----------------:|---------------:|---------------:|
| flagged   |              57 |              3 |             12 |
| unflagged |              25 |              1 |              3 |

## Rows and votes left out

- **Unsure: 28 labelled votes dropped** (on 28 rows; 27 rows lost all their votes this way). Why: the annotator ticked Unsure and the adjudication file (`unsure_for_adjudication.csv`) was never filled in, so there is no settled label to count. Rows lost: [39, 47, 100, 134, 275, 288, 454, 638, 654, 721, 820, 925, 984, 1052, 1187, 1211, 1214, 1463, 1544, 1714, 1720, 1819, 1960, 2329, 2514, 4588, 4629]. (`agreement_report.md` lists 45 Unsure ticks in total; the other 17 sit on blank Primary cells, which carry no vote anyway.)
- **Filled from the original label: 81 votes excluded** (81 rows). Why: these were CS-majors blanks that `relabel_agreement.py` filled with the original dataset label; nobody reviewed them, so counting them would make labels look unchanged when they were never checked. 56 of these rows are in the test split.
- **Blank Primary** (no vote, already excluded in `agreement_report.md`): 24 cells; rows left with no vote at all because of a blank: [257, 296, 423, 600, 695, 781, 957, 1057, 1138, 1449, 1589, 1688, 1762, 2084, 2163, 2222, 2318, 2326, 2401, 2546, 4575, 4670].

## Reconciliation with `agreement_report.md`

`agreement_report.md` counts **161** rows originally labelled as a distortion where calibrated pessimism held a majority on the 12-way scale, counting every non-blank vote (Unsure and filled votes included). Here the redefinition group has **160** rows. In the old count but not here: 2 (2 because the only vote was Unsure and is dropped here; the rest change group under the >half-of-valid-votes rule): [275, 638]. Here but not in the old count: 1 (rows where CP and plain ND votes together form a No-Distortion majority and CP is at least tied with plain ND): [4643].

## Optional check: are the redefinitions really redefinitions?

`redefinition_check_20.xlsx` holds 20 rows from the redefinition group (sampled with seed 42: every multi-annotator redefinition row (1) is included, the rest are single-annotator rows; order shuffled), with the text and distorted part only, no labels. Give it to one psychology annotator with the question: *"If calibrated pessimism were not an option, would you call this distorted?"* Mostly **yes** means these rows really are redefinitions; mostly **no** means they were errors all along, and Chapter 9 should say so.

**Status: pending** (sheet generated, not yet answered).
