# RUN 1 — frozen history

Everything in this folder was produced **before** the August 2026 rerun. It is
kept for provenance and for the thesis's "what changed and why" section. Nothing
new is ever written here.

## Why it is quarantined rather than deleted

These numbers are real — they were measured, and several are cited in
`docs/RESULTS.md` and the supervisor reports. But three defects make them
**non-comparable with RUN 2**, and two of them make some of these folders
non-comparable with *each other*:

1. **Determinism was off.** Two runs of the same config at the same seed differ
   by up to **0.047 macro-F1** here — larger than the seed-to-seed SD that was
   being reported (0.007–0.030). Any gap smaller than ~0.05 between two RUN 1
   folders is unreadable noise, not a finding. This is exactly what sank the
   first flat-vs-cascade comparison (+0.003).
2. **Leaked splits.** `results_combined/` was trained on `data/splits_combined`,
   in which 396 train rows also appear in the Annotated val/test — 195 of 253
   test rows (77%). `results_stage2/` inherits the same leak through
   `data/splits_stage2`. `src/compile_results.py` flags these as `valid=False`.
3. **Uncontrolled budgets.** The cascade track ran at `--max-length 256`
   (24.9% of test rows truncated) while the experiment suite ran at 512 (2.4%).
   Part of any gap between those two folders is truncation, not architecture.

## Reading a RUN 1 number safely

Compare **within** one folder (e.g. seed-to-seed inside `results_stage1/`), and
state the determinism caveat. Do not put a RUN 1 number and a RUN 2 number in the
same table cell without saying which is which — `src/compile_results.py` emits a
`run` column precisely so that distinction survives into the compiled CSV.

## Folder map

The registry with one line per folder — which splits dir produced it, and whether
it is valid — lives in `SPLITS_BY_DIR` at the top of `src/compile_results.py`.
That is the single source of truth; this README does not duplicate it.
