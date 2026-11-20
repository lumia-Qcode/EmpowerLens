# D28 trial merge

Date: 2026-11-20. This was a dry run of D28 in a throwaway detached worktree. No real branch was changed, nothing was pushed and no PR was opened. The worktree has since been removed.

## What was merged

| ref | SHA | vs `origin/main` (behind / ahead) |
|---|---|---|
| `origin/main` | `47018ded166eabd817e9901365d1c249fd542858` | — |
| `origin/lumia-space` | `54a366e5ae4cdf0cf3f495a02f4cbc4f19435692` (S09) | 1 behind / 11 ahead |
| `origin/izza-space` | `4c89cbedeb336fbdfaab26a68dce270e50659a54` | 19 behind / 0 ahead |
| `origin/nayab-space` | `8a8f5dba7a3e150b64506621cb427e67d5061aa7` | 49 behind / 1 ahead |

The one main commit that lumia-space doesn't have is the PR #12 merge commit. It contains no content that lumia-space lacks.

## Results by step

1. **lumia-space → main (`--no-ff`)**: merged cleanly with **no conflicts**. The 11 commits were S05, D19 fix, D20 setup, D26 re-test, S06, D19, D20, D21, D22, S07 and S09.
2. **izza-space**: "Already up to date". `izza-space` is an ancestor of both `origin/main` and `origin/lumia-space`, so its PR has nothing new to merge.
3. **nayab-space** (`8a8f5db` "Synth: retry once from the specification instead of rewriting"): this merge had **1 textual conflict and 1 hidden semantic break**:
   - `src/synth/generate.py` had a CONFLICT with 5 hunks.
   - `src/synth/prompts.py`, `src/synth/README.md` and `src/synth/PIPELINE.md` all **auto-merged without conflicts**. But the auto-merged `prompts.py` no longer has `build_reviser_user`, and lumia-space's `src/synth/batch_generate.py` (line 284) still calls it. I tested this: keeping the auto-merged `prompts.py` makes `tests/test_batch_generate.py::test_reject_revise_accept` fail with an `AttributeError`. The sequential `generate.py` on the lumia side would also break, because it calls `P.build_reviser_user` too.
   - **planning/** was already gone from main. Commit `397dea7` "Keep planning/ local" removed it before this trial. So merging nayab-space removes nothing, and the brief's warning no longer applies.

## Conflict table

| file | lumia-space side | nayab-space side | recommended resolution |
|---|---|---|---|
| `src/synth/generate.py` (CONFLICT) | S04–S06 (`c2f8529`, `30b16b7`, `229d1d8`) split `run()` into helpers that `batch_generate.py` and the tests import: `prefilter`, `rejected_row`, `accepted_row(..., revised=...)`, `RunStats`, `failure_reason`. It also added the NVIDIA NIM provider, critic slots with backups, and `seed_ids` on rows. It keeps **one critic-informed revision**, and rows carry `revised`. | Replaces the revision with **one fresh regeneration from the same spec**. The generator never sees the failed text or the critics' note, the retry goes through the length, clinical-term, English and duplicate filters, rows carry `attempt` (1 or 2) instead of `revised`, and `n_revised` becomes `n_retried`. It cites methodology §5.2.4. The commit was written against the old monolithic `run()`. | **Keep lumia-space** (`git checkout --theirs` when merging nayab into main, or `--ours` in the order used here). Nayab's hunks target code that no longer exists in that form. |
| `src/synth/prompts.py` (auto-merged, **semantic break**) | Gender-neutral wording ("the writer") and `LAST_SEED_IDS` for `seed_ids`. | Removes `build_reviser_user` and updates the docstring. | **Keep lumia-space.** `batch_generate.py` needs `build_reviser_user`. |
| `src/synth/README.md` (auto-merged) | 13 triggers, `gender_expectation` removed, NIM defaults, a Kaggle `batch_generate.py` section ("revise the failures once"). | Changes "revise/revision" to "regenerate/retry" and documents the `attempt` field. | **Keep lumia-space.** Otherwise the README describes a retry that the code and the S06 data don't use. |
| `src/synth/PIPELINE.md` (auto-merged) | unchanged | Row 5 "Reviser" becomes "Retry" and "revision rate" becomes "retry rate". | **Keep lumia-space**, for the same reason. |

**Why I didn't port nayab's change:** the committed S05/S06 data was produced with the **revision** path through `batch_generate.py`. `docs/synth_calibration.md` reports 5 revised rows (2 accepted), and `docs/synth_run.md` reports "Revised once: 80" and "Rejected… (after a revision): 41". S07 and S09 build on those rows. Porting regenerate-from-spec now would leave code, docs and data out of step. Making it true would mean changing both `generate.py` and `batch_generate.py`, renaming `revised` to `attempt` in the tests and S07/S09, and **re-running S06 on Kaggle**. Nayab's commit has one more fix: retries now go through the filters, while lumia's revision path skips the filters on the revised text. That fix belongs to the same retry design, so it wasn't ported either.

**Decision for the team (not for the merger):** Nayab's commit shows that the S06 run did not follow methodology §5.2.4, which says "regenerated, never edited to fit its label". The team has two options:

- **(a)** Keep the code as it is, and say in Ch7/Ch9 that S06 used one critic-informed revision (80 of 300 specs went through it). Rows with `revised: true` can be compared with first-pass rows. Nayab would then drop or revert her commit before pushing, or the D28 merger takes lumia's side as above.
- **(b)** Adopt the retry and re-run S06 (and then S07–S10).

Option (a) is the realistic one before the deadline.

## Tests on the merged tree

The tests ran with the worktree as the working directory and the main repo's venv. `src.__file__` resolved to the worktree. The command was `"venv\Scripts\python.exe" -m pytest -q`.

**162 passed, 0 failed** (1 unrelated warning: anyio `CallSpec2` deprecation, 17.8 s). The final merged tree is byte-identical to `origin/lumia-space` `54a366e`.

## Checks on the merged tree

| check | result |
|---|---|
| `planning/` absent | **Yes**: no directory and 0 tracked files |
| `data/splits/` identical to `origin/main` | **Yes**: `git diff --stat origin/main -- data/splits` is empty |
| Frozen files unchanged vs `origin/main` (`cd_pipeline.py`, `README.md`, `requirements.txt`, the 3 result/sample CSVs, `Annotated_data.csv`) | **Yes**: the diff is empty |
| Cited "Done when" files exist | **All 20 present**: `docs/{thresholds,model_selection,relabelled_results,truncation_test,pipeline_tests,app_tests,survey_results,collected_data_counts,t14_hpo_results,synth_calibration,synth_run}.md`, `data/challenge_set/results.md`, `results_cbt_transfer/transfer_evaluation_final.md`, `results/results_explain/captum_check.md`, `results/results_multitask/summary.md`, `results/results_cascade_inputs/summary.md`, `results/results_span_selector/baselines.md`, `results/results_cluster_label_errors/duplicates_report.md`, `data/splits_relabelled/manifest.json`, `data/supplementary/supplementary_labelled.csv` |

**Windows path-length gotcha:** some tracked paths are 176 characters long, for example `results/results_RUN1/results_cascade_codipas_cls/per_class_cascade_multilabel_binary_...csv`. Checking out into a deep folder fails with "Could not reset index file". This trial's first `worktree add` failed that way. For the fresh-clone test, clone into a short folder (e.g. `C:\el`) and/or use `git clone -c core.longpaths=true`.

## What will differ at real-D28 time

- **lumia-space is still moving.** Another session is still committing S07–S10 and S08 on it, and this trial used `54a366e` (S09). Redo the merge after the freeze. It should stay conflict-free against main unless main changes in the meantime.
- **nayab-space may change.** Nayab pushes only after she approves. If she drops or reverts `8a8f5db`, step 3 becomes a no-op or a trivial merge. If she rebases the retry onto lumia's refactored `generate.py`, the textual conflict goes away, but the semantic question above remains (data vs code). In that case, re-run `pytest -q` and check `batch_generate.py`.
- **izza-space** only needs a PR if new commits appear. As of this trial it is fully contained in main.

## Command sequence for the real D28

Run from the main repo folder (`venv\Scripts\python.exe` = the project venv):

```
git fetch origin
git rev-list --left-right --count origin/main...origin/lumia-space   # expect 1 / N
git rev-list --left-right --count origin/main...origin/izza-space    # expect X / 0 => no PR needed
git rev-list --left-right --count origin/main...origin/nayab-space

# 1. lumia-space (after the S-task freeze)
gh pr create --base main --head lumia-space --title "D28: merge lumia-space" --body "..."
gh pr merge <n> --merge            # or Merge in the GitHub UI; expect no conflicts

# 2. izza-space: only if ahead > 0
gh pr create --base main --head izza-space --title "D28: merge izza-space" --body "..."
gh pr merge <n> --merge

# 3. nayab-space (only after Nayab has pushed her approved version)
gh pr create --base main --head nayab-space --title "D28: merge nayab-space" --body "..."
#   If GitHub reports a conflict, resolve locally on a temp branch from main:
git fetch origin
git switch -c d28-nayab origin/main
git merge --no-ff origin/nayab-space
#   resolve: keep main's (= lumia's) src/synth (see conflict table above)
git checkout origin/main -- src/synth/generate.py src/synth/prompts.py src/synth/README.md src/synth/PIPELINE.md
venv\Scripts\python.exe -m pytest -q             # must be all green
git commit                                       # merge commit
git push origin d28-nayab
gh pr create --base main --head d28-nayab --title "D28: merge nayab-space (synth kept at S06 state)" --body "..."
gh pr merge <n> --merge
#   (if the team picks option (b) instead, do not use this resolution; see "Decision for the team")

# 4. Fresh-clone check on main (short path because of the 176-char filenames)
cd C:\
git clone -c core.longpaths=true <origin-url> C:\el
cd C:\el
"C:\Users\HF SYTORE\Desktop\EmpowerLens\venv\Scripts\python.exe" -m pytest -q
#   then re-run the check list above: planning/ absent, data/splits unchanged,
#   frozen files unchanged, every cited file present.
```

---

## Real D28 (21 Nov 2026)

| Branch | State on 21 Nov | What happened |
|---|---|---|
| `lumia-space` | all earlier work already in `main` via PRs #10–#13 | PR #15 merges the last commits: S04/S05 sign-offs, the README typo fix, and the `nayab-space` merge below |
| `izza-space` (`4c89cbe`) | 38 behind / 0 ahead of `main` | nothing to merge: GitHub refuses a PR with no new commits ("No commits between main and izza-space"); all its work is already in `main` |
| `nayab-space` (`8a8f5db`) | 1 ahead, conflicts in `src/synth/generate.py` | PR #14; conflict resolved on `lumia-space` in `3f03491` with **option (a)**: the four `src/synth` files keep the revision path; #14 is marked merged when #15 lands |

**Option (a), as recommended above.** The committed S05–S10 data and analyses were produced with the critic-informed
revision, so Ch7/Ch9 should say S06 used one critic-informed revision (80 of 300 specs), not a regeneration from the
spec. Nayab's retry design (which also sends retries through the filters) goes to the FYP-2 plan.

**README.md** (frozen) changed by one word ("thes" → "the", line 110), approved by Lumia as an exception to the
freeze rule.

**Tests:** 170 passed on `lumia-space` after the merge; the fresh-checkout run on `main` is recorded below.
