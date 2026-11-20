# D3 freeze audit: Lumia's and Izza's tasks

**Date:** 2026-11-20
**Audited:** `origin/lumia-space` at `54a366e5ae4cdf0cf3f495a02f4cbc4f19435692` ("S09 done …"), plus `origin/main` at
`47018ded166eabd817e9901365d1c249fd542858` where needed. `origin/izza-space` (tip `4c89cbe`) is fully contained in
`origin/lumia-space`, so its work is audited there.
**Method:** read-only. `git fetch origin`, then `git ls-tree` / `git show` / `git log` on the remote refs. Uncommitted
files in the working tree were ignored. `data/splits/test.csv` was only hashed, never parsed.
**Sources:** `D3/TASK_BRIEFS_D3.md` ("Done when" lists) and a copy of `D3/tasks_D3_2026-11-15.xlsx` (Tasks and Test
cases tabs). Both files sit in `D3/`, which is **not tracked** in git.

Status values used per task: **Ready** (every "Done when" item is met), **Gaps** (one or more items missing or partial),
**Moved to FYP-2**, **In other session** (S07–S10, being finished in a parallel session right now; listed by what is
committed so far, not counted as failures), **Not started**.

## Summary

| Overall | Count | Tasks |
|---|---:|---|
| Ready | 26 | D03, D04, D07, D08, D09, D10, D11, D12, D13, D14, D15, D16, D17, D18a, D18b, D19, D21, D22, D25, D27, C02, S01, S02, S03a, S03b, S06 |
| Moved to FYP-2 | 1 | D23 |
| Gaps | 6 | D02, D05, D20, D26, S04, S05 |
| In other session | 4 | S07, S08, S09, S10 |
| Not started | 1 (+ D24, Team) | D28 |
| **Total** | **38 Lumia/Izza tasks + D24** | |

Cross-checks: 1 (weights) **pass**, with a note about RNG-state files. 2 (splits) **pass**. 3 (frozen files) **pass for
lumia-space = main**, but 6 of 7 frozen files changed after their first commit. One of those changes (README.md, 9 Oct)
came after the freeze rule was added. 4 (planning/) **pass**. 5 (citations) **2 real dangling references**. 6 (test
cases) **5 Lumia/Izza rows with empty Pass / Fail**.

---

## Report admin

D06 belongs to Nayab and is outside this audit.

## Getting results into the repo

| Task | Owner | Overall | Gaps / evidence |
|---|---|---|---|
| D02 | Lumia | **Gaps** | OK: the 4 filled workbooks (`reannotation_2026-09-27/outputs/relabel/annotator_{1_Laiba,2_Reesha,3_Hurema,4_CSmajors}.xlsx`). OK: `merged_labels.csv` (1,665 rows; has `Row ID`, `Primary`, `Secondary`, `primary_collapsed`, `original_dominant`). OK: `agreement_report.md`, where the 24 invalid rows are listed and explained (blank Primary). **Partial:** `unsure_for_adjudication.csv` has 45 rows with `final_label` and `dropped_reason` **both empty in every row**. The only explanation is collective, in `label_changes_summary.md` ("the adjudication file … was never filled in", 28 votes dropped). OK: pushed (commit `42e6cf2`). |
| D03 | Lumia | Ready | `src/train_span_selector.py` (run command in its docstring). `results/results_span_selector/eval_{val,test}_{42,1337,2024}.json`: P/R/F1, `n_rows`, `n_sentences`. `predictions_{val,test}_<seed>.csv` with columns `Id_Number, sent_idx, sentence, score, selected, gold`. `config.json`. No weights (cross-check 1). |
| D04 | Izza | Ready | `docs/collected_data_counts.md` (commit `493d8dc`) gives survey totals, women, non-English excluded (3), labelled (0), scraped (500, HN 330 / Failory 170), synthetic, and the D23 sentence. Note: dated 17 Nov, so "scraped labelled 0" and "synthetic 0" are now out of date (S03 labelled 500; S06 made 258). |
| D05 | Izza | **Gaps** | OK: `app/README.md` links to `github.com/izza14/empowerlens-demo` and lists install and start steps, the model and the data location (SQLite `backend/empowerlens.db`). OK: pushed. **Missing:** the README's "Tested by" table is empty, so there is no record that someone other than the builder started the app from the README alone. |

## Re-annotation

| Task | Owner | Overall | Gaps / evidence |
|---|---|---|---|
| D07 | Lumia | Ready | `outputs/relabel/agreement_tables.csv` (+ `.md`): 16 rows = (all four, psychology only, 6 pairs) × (type, binary). Columns are comparison, label level, kappa type, kappa value, raw % agreement and number of rows. The blank-row note is in `agreement_tables.md` (3 of 60 rows: 2057, 414, 1924). |
| D08 | Lumia | Ready | `label_changes.csv` (1,035 rows; `Id_Number, original_label, new_majority_label, n_annotators, unanimous, group`). `label_changes_summary.md` gives counts per group (majority / unanimous), change rate flagged 72/97 (74.2%) vs unflagged 29/87 (33.3%), and the Unsure votes dropped with the reason. The optional 20-row redefinition check is "pending" (optional item). |
| D14 | Lumia | Ready | `data/splits_relabelled/{train,val,test}.csv`: 2,024 / 253 / 253 rows. Train and val have the same `Id_Number` sets as `data/splits` (test.csv not parsed; the manifest records the source hashes, which match the committed split blobs). `label_provenance` is filled in every row. `manifest.json` gives changed rows per file and the source files. |

## Clustering

| Task | Owner | Overall | Gaps / evidence |
|---|---|---|---|
| D11 | Lumia | Ready | `results/results_cluster_label_errors/route_comparison.{md,csv}` (+ `_test.csv`, `_meta.json`): rows for neighbours, model triage and random; AP, P@50/100/200, rows scored, and errors per hour (45 s estimate stated). `cluster_recovery.csv`, `unsupervised_queue.csv` and `near_duplicates.csv` are committed. |
| C02 | Lumia | Ready | `results/results_cluster_label_errors/duplicates_report.md`: 16-pair table, the 253-row vs 251-row headline numbers (mean ± std, 3 seeds) and a size sentence. The 251-row figures are **bounds** (best and worst case from confusion counts), because per-row test predictions were not saved; the report says so. `test.csv` not read. |

## Models, thresholds and evaluation

| Task | Owner | Overall | Gaps / evidence |
|---|---|---|---|
| D09 | Izza | Ready (deviation) | 9 `eval_mental-roberta-base_{binary,multiclass,multilabel}_{42,1337,2024}.json` files, all with `meta.splits = data/splits` and lr 3e-5 / 5 ep / 512 / wd 0.1, plus a three-encoder `month1_summary_meanstd.csv`. They are in **`results/results_mentalroberta_frozensplits/`**, not in `results/results_experiments/` as the brief says. `results/results_experiments/month1_summary_meanstd.csv` is still a 0-byte file (since `2fa7bea`, 26 Sep). `docs/t14_hpo_results.md` is on the branch. |
| D13 | Izza | Ready | `docs/thresholds.md`: target FPR ≤ 0.15, set 13 Nov; binary thresholds per encoder and seed; 10 multilabel thresholds per encoder and seed; val and test specificity, FPR and sensitivity (mean ± std). `results/results_thresholds/specificity.csv`. Notes: test FPR is above target for all encoders (stated). Per-type multilabel specificity is not reported (stated). |
| D12 | Izza | Ready | `docs/model_selection.md`: rule dated 14 Nov, 3-encoder val table, winner MentalRoBERTa (val multilabel macro-F1 0.315 ± 0.011) with its reason, and checkpoint names `checkpoints/{binary,multilabel}_mental-roberta-base_42`. |
| D15 | Izza | Ready | `docs/truncation_test.md`: head 0.215 ± 0.025 vs head_tail 0.223 ± 0.012 val macro-F1; share cut at 256 (27.5%) and 512 (3.2%); conclusion. `results/results_truncation/` (`summary.csv`, `token_lengths.json`, per-seed meta). |
| D16 | Izza | Ready | `results/results_relabelled/train_{orig,relab}__test_{orig,relab}/`: 24 eval JSONs (the 12 required + the 12 original-label comparison runs). `docs/relabelled_results.md` has the 4-cell table with specificity, FPR, sensitivity and macro-F1, and the not-comparable-to-paper note. |
| D21 | Izza | Ready | `results_cbt_transfer/transfer_evaluation_final.md` (+ `d21/` per-class and prediction CSVs, 3 seeds): per-class P/R/F1/support, macro and micro mean ± std, the 4 Oct column, and the format-difference line. Known caveat stated: CBT-CD texts overlap our splits (110 train / 18 val / 18 test), with a `not_in_train` subset (36 items). The commit was made by Lumia. |

## Span selection and architecture

| Task | Owner | Overall | Gaps / evidence |
|---|---|---|---|
| D10 | Lumia | Ready | `results/results_span_selector/baselines.{md,csv}`: selector + first/last/random; P, R, F1 and row hit rate; val and test blocks with rows and sentences scored; the 13-excluded line; the beats-by-more-than-std sentence. |
| D17 | Lumia | Ready | `results/results_cascade_inputs/summary.md`: 5-row table (val/test macro-F1, test specificity, mean ± std), cost of automatic selection, and 15 eval JSONs in `full/`, `crop_sel/`, `crop_gold/`, `marked_sel/`, `marked_gold/`. |
| D20 | Lumia | **Gaps** | OK: `results/results_multitask/summary.md` has the multi-task vs flat vs cascade table (binary specificity, multilabel macro-F1, selector F1; 3 seeds) and `eval_multitask_{42,1337,2024}.json`. **Defect:** the same file still ends with the unconditional "Fallback" paragraph ("The multi-task model was not trained by the D3 freeze … Chapter 9 therefore reports flat vs cascade…"), which contradicts the results table above it. Test case B-06 (Task D20) is empty in the sheet, although the test exists (`tests/test_multitask.py`, B-06 block) and has no `docs/test_evidence/B-06.txt`. |
| D22 | Lumia | Ready | `results/results_explain/captum_check.md`: agreement 57 of 157 (36.3%), 20-row table (`Id_Number`, top 5 words, verdict) and summary. **Disclosed:** the verdicts were assigned by Claude at Lumia's request, not by an independent human reviewer (stated in the summary and in commit `07c4012`). |

## Entrepreneurial data and synthetic generation

| Task | Owner | Overall | Gaps / evidence |
|---|---|---|---|
| S01 | Izza | Ready | `data/supplementary/raw/{hn_raw,failory_raw}.csv` (columns text, url, platform, thread_id, source_item_id, posted_at, collected_at). `SOURCES.md` covers platforms, terms and counts. 500 rows after exact dedup (target 489). Notes: `SOURCES.md` says both scrapers were run by Lumia, and it still lists an "Indie Hackers … `raw/indiehackers_raw.csv` *(to add)*" row for a file that does not exist. |
| S02 | Izza | Ready | `data/supplementary/supplementary.csv` (500 rows; `item_id, text, source, thread_id, collected_at, role`). Roles assigned by `src/scrape/clean_supplementary.py` (seed 42). `cleaning_log.md` gives drops (0) and the real discard rate (0.0%), and pastes the thread check output ("threads in both roles: 0"). Test case C-07 (Task S02) is empty in the sheet. |
| S03a / S03b | Lumia / Izza | Ready | `data/supplementary/labels_lumia.csv` (266) and `labels_izza.csv` (264), with the required columns. All 500 rows have ≥ 1 label, and 30 have both. `agreement.md` gives kappa and raw % for type and binary (n = 30). `supplementary_labelled.csv` has 500 rows with one final label each; the 10 disagreements are settled in `adjudication.csv`. |
| S04 | Lumia | **Gaps** | OK: `data/synthetic_gen/seeds/entrepreneurial_seeds.jsonl` (44 seeds, `seed_source` on each, `item_id` on the scraped ones). OK: `seed_counts.md` gives the per-type table by source, the one-line role check (`'evaluation': 0`) and the gender decision (removed). **Open:** `seed_counts.md` says the 12 Claude-drafted team seeds "must be reviewed by Lumia before S06", and flags seed_040 (sup_384) for a check. Nothing committed records that either was done, and S06 has already run. Test case C-08 (Task S04: Jaccard of seeds against survey, evaluation pool, val and test) is empty and not run. |
| S05 | Izza | **Gaps** | OK: `docs/synth_calibration.md` gives 17 accepted / 5 revised / 3 rejected, cost ($0, with 47 GPU-min; per accepted row), bugs (commit `30b16b7`; 2 left unfixed), and the decision **GO**. **Partial:** the "do the accepted rows show their label" count (14 yes / 3 no) is "Claude's reading; Lumia/Izza to confirm", and no confirmation is committed. Deviation (disclosed): open models on Kaggle instead of the paid Claude/Gemini APIs. The commits are on lumia-space. |
| S06 | Izza | Ready | `data/synthetic/synthetic_train_en.jsonl` (258 rows; the field check pasted in `docs/synth_run.md` shows 0 rows missing text, label, `critic_votes` or `seed_ids`). `docs/synth_run.md` gives requested 300, accepted 258, rejected 41, cost $0 (125 GPU-min) and accepted per type. Its spot-read is also marked "Claude's reading; Lumia/Izza to confirm" (not a Done-when item). |
| S07 | Lumia | In other session | Committed (`8aae121`): `docs/synth_checks.md` (keyword list; macro-F1 0.254 synthetic vs 0.108 val), `results/results_synth_checks/`, `data/synthetic/overlap_flags.csv` and `synthetic_train_en_clean.jsonl` (258 rows, 0 removed). Overlap: 0 rows > 0.6 against seeds, evaluation rows and test. **Not yet:** the survey overlap ("updated when the export is available"). Sheet row C-09 is empty. `synth_checks.md` documents reading `data/splits/test.csv` texts as an exception to the evaluate-only rule. |
| S09 | Lumia | In other session | Committed (`54a366e`): `data/synthetic/flags.csv`, `review_sheet.xlsx`, `review_key.csv`, `docs/synth_label_flags.md`, and `--input` in `src/cluster_label_errors.py` (line 556). Sheet status is still "Not started". |
| S10 | Izza | In other session | Nothing committed: no `docs/synth_label_check.md`, no `data/synthetic/synthetic_train_en_final.jsonl`. |
| S08 | Izza | In other session | Nothing committed: no `docs/synth_effect.md`. Sheet row C-10 is empty. |
| D23 | Izza | Moved to FYP-2 | `docs/survey_results.md`: 0 labelled survey responses (330 usable answers + 14 interviews, none labelled), which moves D23 to FYP-2. Sheet status: Moved to FYP-2. |

## Challenge set

| Task | Owner | Overall | Gaps / evidence |
|---|---|---|---|
| D18a | Lumia | Ready | `data/challenge_set/pairs_lumia.csv`: 25 pairs, 5 per type × 5 types, required columns; 0 pairs with a word-length difference > 20% (max 15.6%). |
| D18b | Izza | Ready | `data/challenge_set/pairs_izza.csv`: 25 pairs, 5 per type × 5 types, required columns; 0 pairs > 20% (max 10.7%). |
| D19 | Izza | Ready | `data/challenge_set/results.md`: annotator 100 of 100, no disagreeing pairs; leak-check F1 0.184; per-type spec and sens counts per seed. Evidence: `annotator_sheet_returned.xlsx`, `leak_check.json`, `predictions_<seed>.csv`. Note: `results.md` does not name the annotator, so "a psychology annotator who didn't write the pairs" cannot be checked from the repo. The commit was made by Lumia. |

## App and testing

| Task | Owner | Overall | Gaps / evidence |
|---|---|---|---|
| D25 | Izza | Ready | `docs/app_model_swap.md`: demo app `570b82a` loads `binary_/multilabel_mental-roberta-base_42` (D12) with the threshold re-fitted on val at FPR ≤ 0.15 (0.9767, as `model_selection.md` requires for retrained weights). The researcher view shows the name and version (`/api/health` output pasted). `app/README.md` names the model. Parity: val row 6 matches between the app and `src.app_parity`. |
| D26 | Izza | **Gaps** (minor) | OK: all 24 D26 rows in the sheet have Actual, Pass/Fail, Evidence and Tested by. Every cited evidence file exists in `docs/test_evidence/`. A-19 was run. `docs/app_tests.md` is committed. **Open:** (1) the 5 Fails (A-04, A-06, A-08, A-16, D-02) are fixed on the empowerlens-demo branch `d26-fixes` (`3a23957`…`f1b849e`), which is **local only**: no `origin/d26-fixes`, not merged into the demo `main` (still at `570b82a`). (2) The re-test (8 Pass, `docs/app_tests.md` "Re-test after fixes") is not in the sheet, whose Actual cells still say "Not fixed". (3) It was committed to lumia-space, not izza-space as the brief says. Disclosed: Tested by "Claude agent, 2026-11-20". |
| D27 | Lumia | Ready | `docs/pipeline_tests.md`: pytest 83 passed; the C-01 line; the C-04 audit table; E-01 69.0 s on a Tesla T4; E-02 71.4 s on an i5-1245U with 15.6 GB RAM; Windows/Linux 83/83. All 16 D27 rows are filled in the sheet. D-04 is a recorded Fail (102 older multilabel runs). C-06 is the expected Fail (owned by C02). |

## Freeze and merge

| Task | Owner | Overall | Gaps / evidence |
|---|---|---|---|
| D24 | Team | Not started | See the checklist below. |
| D28 | Lumia | Not started | `origin/main` is 11 commits behind `origin/lumia-space` (S05 → S09). `origin/izza-space` is fully in main. `origin/nayab-space` has 1 commit not in main or lumia (`8a8f5db`, 12 Nov, "Synth: retry once from the specification…"). It edits `src/synth/generate.py`, `prompts.py`, `README.md` and `PIPELINE.md`, which lumia-space has since changed heavily (generate.py: +438/−275 lines vs its parent), so expect conflicts. pytest on a fresh `main` checkout has not been run. |

---

## Cross-checks

**1. No weights committed.** `git log origin/lumia-space --name-only --format=` (and `--all`), filtered for
`*.safetensors|*.bin|*.pt|*.pth|*.ckpt|checkpoints/`. Result: **no model weights** in any commit
(`model.safetensors|pytorch_model|optimizer.pt`: 0 hits). The only matches are 25 files under
`results/results_reannotation/fold{0-4}_ckpt/checkpoint-*/`. Each folder holds `config.json`, the tokenizer files,
`trainer_state.json` and `rng_state.pth`, which is a **14,645-byte RNG state, not weights**. They were added in `2fa7bea`
(26 Sep, "cleaned results folders"). They don't break the rule, but they are checkpoint folders that `.gitignore`'s
`*.pt`/`checkpoints/` patterns don't catch (`*.pth` is not listed). Weights for the app live outside git
(`empowerlens-demo/models/d12/`, gitignored, per `docs/app_model_swap.md`).

**2. `data/splits/` unchanged. Pass.** `git log origin/lumia-space -- data/splits` returns only `51cafef`
(2026-07-22, the creation commit), and the same holds on main. The blob IDs of `train.csv`, `val.csv`, `test.csv` and
`split_manifest.json` are identical in `51cafef`, `origin/main` and `origin/lumia-space`. The manifest's `n` (2024 / 253 / 253) matches the
committed train and val files (test was not parsed). The manifest's `source_sha256` (`22da63fd…`) equals the sha256 of
the committed `Annotated_data.csv` with CRLF line endings (as checked out on Windows). The D14 manifest's
recorded hashes of `data/splits/{train,val,test}.csv` (`2410e1d6…`, `82510ce7…`, `1013562d…`) also equal the
CRLF sha256 of the committed blobs.

**3. Frozen files.** All 7 (`cd_pipeline.py`, `README.md`, `requirements.txt`, `binary_f1_results.csv`,
`multiclass_f1_results.csv`, `labeled_data_sample.csv`, `Annotated_data.csv`) are **byte-identical between
`origin/lumia-space` and `origin/main`**, and none was touched since 1 Nov (0 commits). "Unchanged since first commit"
holds only for `requirements.txt` and `Annotated_data.csv` (1 commit each):

| File | Commits | After first commit |
|---|---:|---|
| `cd_pipeline.py`, `binary_f1_results.csv`, `multiclass_f1_results.csv`, `labeled_data_sample.csv` | 4 | `7011cfc` (21 Jul), `8acb63b` (27 Jul), `67dccf6` (15 Aug, the merge that also added the FROZEN FILES section to CLAUDE.md) |
| `README.md` | 2 | `1bb8b7a` (9 Oct, Izza, "added results for t14 hpo"): changes "only the" to "only thes". This is a one-word edit to a frozen file **after** the freeze rule, and it introduces a typo. It is on main too. |

**4. `planning/` not tracked. Pass.** `git ls-tree -r origin/lumia-space` has 0 paths under `planning/` (same on main).
It was untracked in `397dea7` (4 Nov) and `.gitignore` line 22 lists `planning/`. Note: `D3/` (the brief and the task sheet) is
also untracked, although it is not gitignored.

**5. Report files cite committed paths.** Every backticked or linked path in 31 D3 report files was resolved against the
`origin/lumia-space` tree (root-relative, then file-relative, with `<seed>`/`{a|b}` patterns expanded). Files checked: `data/challenge_set/results.md`,
`results_cbt_transfer/transfer_evaluation_final.md`, `results/results_explain/captum_check.md`,
`results/results_multitask/summary.md`, `docs/thresholds.md`, `docs/model_selection.md`,
`docs/relabelled_results.md`, `docs/truncation_test.md`, `results/results_cascade_inputs/summary.md`,
`results/results_span_selector/{baselines,d03_summary}.md`, `results/results_cluster_label_errors/{duplicates_report,route_comparison}.md`,
`docs/pipeline_tests.md`, `docs/app_tests.md`, `docs/app_model_swap.md`, `reannotation_2026-09-27/outputs/relabel/{agreement_report,agreement_tables,label_changes_summary}.md`,
`data/supplementary/{agreement,SOURCES,cleaning_log}.md`, `data/synthetic_gen/seeds/seed_counts.md`,
`docs/{collected_data_counts,survey_results,synth_calibration,synth_run,synth_checks,synth_label_flags,t14_hpo_results}.md`,
`app/README.md`. Unresolved hits that were bare filenames of files in the same folder or in `src/` were checked by
hand and exist.
**Real dangling references:**
- `D3/tasks_D3_2026-11-15.xlsx`, cited in `docs/pipeline_tests.md` and `docs/app_tests.md`: `D3/` is not committed.
- `raw/indiehackers_raw.csv`, cited in `data/supplementary/SOURCES.md` ("to add"): does not exist.

Minor: `results/results_multitask/summary.md` cites `results_RUN2/results_multilabel_flat`. The real path is
`results/results_RUN2/results_multilabel_flat/`, and a different top-level `results_RUN2/` also exists.
External, by design: `backend/…`, `models/d12/`, `MODEL_CARD.json` (empowerlens-demo repo), and `/kaggle/temp/…`.
All evidence files named in the sheet's *Evidence file* column exist in `docs/test_evidence/` (78 files).

**6. Test cases with empty Pass / Fail (Lumia/Izza rows).** The sheet copy may lag the repo.

| ID | Task | Owner | Note |
|---|---|---|---|
| B-06 | D20 | Lumia | Test exists (`tests/test_multitask.py`, B-06 block); sheet row and `docs/test_evidence/B-06.txt` are missing |
| C-07 | S02 | Izza | Output already pasted in `data/supplementary/cleaning_log.md` ("threads in both roles: 0"); needs the sheet row + `C-07.txt` |
| C-08 | S04 | Lumia | Not run (seed vs survey / evaluation / val / test Jaccard) |
| C-09 | S07 | Lumia | In other session (S07 overlap result: 0 rows > 0.6 vs seeds / evaluation / test; survey pending) |
| C-10 | S08 | Izza | In other session (S08 not committed) |

The D26 rows are filled, but they show the original run (5 Fail). The re-test after the fixes (8 Pass) is only in
`docs/app_tests.md`. C-06 is Fail as expected (C02). D-04 is a recorded Fail (D27).

---

## Gaps to close before the freeze

| Owner | Task | What is missing | Smallest fix |
|---|---|---|---|
| Lumia | D02 | `unsure_for_adjudication.csv`: 45 rows with empty `final_label` / `dropped_reason` | Fill `dropped_reason` ("not adjudicated; vote dropped, see label_changes_summary.md") in all 45 rows, or adjudicate them. Either way, say in Ch9 which was done. |
| Izza | D05 | `app/README.md` "Tested by" row empty | Someone other than the builder starts the app from the README and fills in name, date and Yes/No |
| Lumia | D20 | Stale "Fallback" paragraph contradicts the results in `results/results_multitask/summary.md` | Delete the paragraph (or make the report script emit it only when the eval files are missing) |
| Lumia | D20 / B-06 | Sheet row empty, no evidence file | Run `pytest tests/test_multitask.py -v`, save the output as `docs/test_evidence/B-06.txt`, fill the row |
| Izza | D26 | Fixes on the local-only `d26-fixes`; sheet still shows the pre-fix results | Push and merge `d26-fixes` into empowerlens-demo `main`; add the re-test result (or a "fixed in f1b849e, re-test Pass" note) to the 5 Fail rows' Actual |
| Lumia | S04 | Review of the 12 Claude-drafted team seeds and of seed_040 not recorded | Add one line to `seed_counts.md`: reviewed by Lumia (date) and the outcome for seed_040 |
| Lumia | S04 / C-08 | Seed-overlap test case not run | Run the Jaccard check of seeds against the evaluation pool, val and test (survey when available), save `C-08.txt`, fill the row |
| Izza | S02 / C-07 | Sheet row empty | Copy the cleaning_log check output to `docs/test_evidence/C-07.txt`, mark Pass |
| Izza (or Lumia) | S05 | The yes/no judgement on the 17 accepted rows is Claude's, still "to confirm" | A human reads the 17 rows and confirms or changes the column in `docs/synth_calibration.md` |
| Lumia | S07 / S09 | In other session | Finish and commit; fill C-09 |
| Izza | S08 / S10 | In other session; nothing committed | `docs/synth_label_check.md` + `synthetic_train_en_final.jsonl` (S10), then `docs/synth_effect.md` (S08), fill C-10. Otherwise mark them Moved to FYP-2 |
| Lumia | (citations) | `D3/tasks_D3_2026-11-15.xlsx` is cited but untracked; `raw/indiehackers_raw.csv` cited in SOURCES.md | Either commit `D3/` or reword the two docs to "the D3 task sheet (kept off-repo)"; delete the Indie Hackers row from `SOURCES.md` |
| Izza | (frozen file) | `README.md` typo "thes" from `1bb8b7a` | Optional and team decision: revert the one word (needs an exception to the freeze rule) or leave it and mention it |

Notes (not gaps):
- D04's counts are dated 17 Nov (scraped labelled 0, synthetic 0). Update them if Ch9 quotes D04 for these.
- `results/results_experiments/` holds 0-byte files from `2fa7bea` (26 Sep), including `month1_summary_meanstd.csv`. Ch9 should cite
  `results/results_mentalroberta_frozensplits/month1_summary_meanstd.csv`.
- `docs/~$powerLens_Project_Proposal.docx` is a committed Office lock file (`437db3f`, 13 Sep).

---

## D24 checklist: what remains

**1. Every result Ch7 and Ch9 use is committed and pushed.** Done for every task above except S08 and S10 (nothing
committed), the S07 survey overlap, and the open items in the gaps table.

**2. Status values in the sheet** (Tasks tab of the copy read today). These Lumia/Izza rows don't match the repo:

| Task | Sheet says | Repo says | Set to |
|---|---|---|---|
| S04 | Not started | committed (`c2f8529`) | Done (after the seed-review line) |
| S05 | Not started | committed (`6e748d5`) | Done (after human confirmation) |
| S06 | Not started | committed (`cc8776d`) | Done |
| D19 | In progress | committed (`1012229`) | Done |
| D20 | Not started | committed (`406977e`) | Done (after removing the fallback paragraph) |
| D21 | In progress | committed (`f07022d`) | Done |
| D22 | In progress | committed (`07c4012`) | Done |
| S07 | Not started | partly committed (`8aae121`) | Done / In progress (other session) |
| S09 | Not started | committed (`54a366e`) | Done (other session) |
| S08, S10 | Not started | not committed | Done, or Moved to FYP-2 |
| D24, D28 | Not started | — | after the freeze |

Rows already correct: D02–D05, D07–D18b, D25–D27, C02, S01–S03b (Done), and D23 (Moved to FYP-2). D02, D05 and D26 are
"Done" in the sheet but have the gaps above. No Lumia/Izza row may be left "In progress". The Summary tab's
counts are formulas and need Excel to recalculate.

**3. Nayab's traceability check:** every Ch9 number traces to a committed file. Not done yet. Nayab's rows (D06,
W01–W11) are outside this audit. Cross-check 5 found two dangling citations to fix first.

**4. After D24: D28.** Merge `lumia-space` (11 commits ahead of main), `izza-space` (already in main) and `nayab-space`
(1 commit, `8a8f5db`, expect conflicts in `src/synth/`) into `main`. Then run `venv\Scripts\python.exe -m pytest -q` on a
fresh checkout of `main` and confirm every file the report cites exists on `main`.
