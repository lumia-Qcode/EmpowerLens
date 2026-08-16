# TODO — everything flagged, ranked

Written 2026-08-15, updated 2026-08-16 after the first successful cascade run.

Priority key: **P0** blocks other work · **P1** correctness, will corrupt results
if ignored · **P2** improvement, safe to defer.

---

## Start here (in this order)

1. **P0** — Tell Lumia about the `splits_combined` leakage (§4.1). Her committed
   results are affected, and you now have a number for how much it inflated them.
2. **P1** — Train a flat multilabel baseline on `data/splits` (§2.2). Without it
   the flat-vs-cascade comparison is unanswerable. ~5 minutes.
3. **P1** — Decide the synth critic-panel question (§3.2) **before** spending
   anything on calibration, or the rates you measure won't transfer.

---

## 1. Results so far (2026-08-16 run — clean `data/splits`, 3 seeds)

All leakage-free. Stage 1 + Stage 2 trained on `data/splits` (train-in-val 0,
train-in-test 0); Stage 2 on `data/splits_stage2` (1,278 distorted rows, derived
by filtering, not re-splitting).

| Model | Metric | Test |
|---|---|---|
| **Cascade** (Stage 1 → Stage 2) | macro_f1 | **0.225 ± 0.027** |
| | weighted_f1 | 0.229 ± 0.023 |
| **Stage 1** (binary) | positive_class_f1 | 0.791 *(seed 2024; other seeds not recorded)* |
| | weighted_f1 | 0.746 *(seed 2024)* |
| **Flat multiclass** (`results_multiclass_v2`) | macro_f1_10 | 0.125 ± 0.008 |
| | weighted_f1 | 0.280 ± 0.022 |

**Not comparable — leaked, retained for reference only:**

| | macro_f1 | macro_f1_10 |
|---|---|---|
| `results_combined` multilabel (flat) | 0.279 ± 0.005 | — |
| `results_combined` multiclass | — | 0.188 ± 0.012 |

### The leakage, quantified — this is a thesis figure

Same model, same metric, leaked splits vs clean:

```
multiclass macro_f1_10:  0.188 (leaked)  ->  0.125 (clean)   = 50% relative inflation
```

**Caveat to state honestly:** the clean run also had a smaller training set
(2,024 vs 4,645 rows), so that 50% mixes the leakage effect with reduced training
data. Do not attribute all of it to leakage.

### Open question the run could not answer

The cascade (0.225) appears *below* the flat multilabel (0.279) — but that flat
number is leaked, so the comparison is meaningless in either direction. If flat
multilabel deflates by the same factor as multiclass, it would land near 0.186,
i.e. **below** the cascade. That is arithmetic on an assumption, **not evidence**.
§2.2 resolves it.

### To record

- [ ] Stage 1 `positive_class_f1` for seeds 42 and 1337 (only 2024 was captured)
- [ ] Stage 2 isolated test macro_f1 (val was 0.343 / 0.331 on seeds 42 / 1337)
- [ ] Whether the paper's binary F1 = 0.79 is weighted or positive-class — it
      decides whether your comparable number is 0.791 or 0.746

---

## 2. Cascade — what's next

### 2.1 Save the outputs — P0 if the session is still open
`/kaggle/working` dies with the session unless you Save Version.
```
!cd /kaggle/working && zip -r results_all.zip results_stage1 results_stage2 results_cascade results_multiclass_v2
```
Checkpoints are not worth saving — retraining is ~90s/seed.

### 2.2 Flat multilabel baseline on clean splits — P1, ~5 min
The single missing piece for a defensible flat-vs-cascade claim.
```python
for seed in SEEDS:
    run_and_report("multilabel", PARENT_SPLITS, "results_multilabel_flat", seed,
                   ckpt_dir="/kaggle/working/checkpoints_flat",
                   extra_flags="--max-length 256 --truncation head_tail --batch-size 32")
```
**`ckpt_dir` is mandatory** — checkpoints are named `{task}_{TAG}_{seed}`, so
without it this silently overwrites Stage 2.

Then swap `results_combined` → `results_multilabel_flat` in cell 15's `SOURCES`.

### 2.3 Re-run cell 15 — P2
Only meaningful after 2.2. Its current "flat" column is the leaked baseline.
Runs anywhere — it just reads CSVs, no GPU or Kaggle session needed.

### 2.4 Stage 2 threshold provenance — P2 (writeup)
Stage 2's per-class thresholds are tuned on distorted-only val, where *every* row
contains a distortion, so they are permissive by construction. At cascade time
they are applied to Stage 1's output, which includes false positives from a
distribution Stage 2 never saw. Softened by `max_labels` capping but never
*forcing* a label. **One honest sentence in the methodology, not a code fix.**

### 2.5 Exposure bias in Stage 2 — P2 (ablation)
`data/splits_stage2` has **zero all-zero-target rows** (verified). Stage 2 trains
in a world where every input is genuinely distorted, then at inference receives
Stage 1's false positives. Cheap ablation: mix some No-Distortion rows into
Stage 2's training with all-zero targets and see whether end-to-end F1 improves.

---

## 3. Synthetic data pipeline (`src/synth/`)

### 3.1 Set API keys — P0
Never run against a real API. Needs `ANTHROPIC_API_KEY` + `GEMINI_API_KEY`, set at
Windows **User** level so new shells inherit them.

### 3.2 Decide the critic panel — P1, decide before spending
All three critics are `gemini-2.5-flash` at different temperatures — that is
**self-consistency, not independent annotation**. Correlated errors never get
voted out and `y_soft` is overconfident. Fix: `--critic-model` accepting a
comma-separated list (e.g. Gemini + Kimi K2.5 + Gemini). Kimi is open-weight with
an OpenAI-compatible API; adds ~3.5% to run cost.

### 3.3 Calibration run — P0 after 3.1/3.2
```
venv\Scripts\python.exe -m src.synth.generate --n 20 --out data/synthetic/calibrate.jsonl
```
~₨56. Then **read all 20 rows** — counters cannot show artificiality. Cost per
spec ≈ `$0.0089 × (1 + revision_rate)`, so 300 specs is anywhere between ₨740 and
₨1,485 depending on a rate nobody has measured.

### 3.4 Sonnet 5 intro pricing expires 2026-08-31 — P1, 15 days
`PRICES` in `src/synth/llm.py` hardcodes the intro rate ($2/$10 per MTok).
Standard is $3/$15. After 31 Aug the script **silently under-reports cost by a
third**. Run the bulk generation before then, or update the dict.

### 3.5 Known bugs — P1
- **`_critique` is not wrapped in try/except** — a Gemini rate-limit ends the run.
- **A dropped critic verdict skews the vote** — silently omitted, but
  `aggregate()` still divides by `n_critics`, inflating the revision rate and cost.
- **`estimate_cost` reports $0 for unknown models** — `(0.0, 0.0)` fallback.

### 3.6 Diversity checks — P2
On calibration output: distinct openers at 1/2/4 words (already printed);
max-pairwise Jaccard *distribution* (a near-verbatim paraphrase measures **0.762
and passes** the 0.8 threshold); vocabulary + length vs `Annotated_data.csv`.

### 3.7 Keyword-only leakage baseline — P2
`src/synth/leakage_check.py`. If a hand-written keyword list beats chance on the
synthetic set, lexical leakage remains. Near chance while the fine-tuned model
scores well = proof the model learned structure.

### 3.8 Granularity band vs the form — P2
Form asks 60–120 words; `snippet` is 25–60, `reflection` 105–195. Length mismatch
between synthetic train and real test is a domain-shift confound.

### 3.9 Rebuild the seed set after collection — P1 (blocked on the form)
All 13 seeds are `source: authored_from_*` — written, not collected. They anchor
*format* but not *register*. After collection, **humans hand-write 25–30 grounded
seeds**; participant text never goes to an API. Cheap test: 20 rows at `k=4` vs
`k=0` (~₨110) — if seeds move nothing, they're decoration.

---

## 4. Repo-wide issues

### 4.1 `data/splits_combined` has ~75% train/test leakage — P0, tell Lumia
```
data/splits (original):  train-in-val 0    train-in-test 0     <- clean
data/splits_combined:    train-in-val 194  train-in-test 189
                         = 76.7% of val    = 74.7% of test
```
**Root cause:** `make_splits_combined.py` is written correctly — its docstring
says val/test stay frozen and the code does that. The bug is its *input
assumption*: it treats CODIPAS as disjoint. It isn't. **1,937 of CODIPAS's 2,621
rows (74%) are already in `Annotated_data.csv`** — 1,554 in train, 194 in val,
189 in test. Only 684 rows are genuinely new.

**Invalidates** everything in `results_combined/`. Now quantified: 50% relative
inflation on multiclass macro_f1_10 (§1).

**Fix:** dedup CODIPAS against the frozen Annotated val/test before merging
(mandatory) and against train too (recommended). Post-fix combined train ≈ 2,708.

**Do not regenerate unilaterally** — Izza and Lumia have committed results
depending on those files, and `CLAUDE.md` treats committed splits as immutable.

### 4.2 `CLAUDE.md` on main is corrupted — P1
Line 8: `...*"Detecting Cognitive# CLAUDE.md — EmpowerLens` — sentence cut off
mid-phrase, file header spliced in, opening block repeated (~7 duplicated lines
in 274). Someone pasted the file into itself during a merge. It's the instruction
file every AI session reads. **Worth its own small PR.**

### 4.3 `captum` missing from `requirements-transformer.txt` — P1
`src/explain.py` imports it; only installed ad hoc in the cascade notebook's
cell 1. A fresh clone plus the documented install fails to import.

### 4.4 `explain.py` is hardcoded to RoBERTa — P1
Line 26 is `model.roberta.embeddings`. `AttributeError` on
`mental/mental-bert-base-uncased`, which needs `model.bert.embeddings`.

### 4.5 `run_name` uses POSIX-only path splitting — P2
`args.model.split('/')[-1]` breaks on Windows paths. Only matters if you pass a
local checkpoint to `--model` (sequential fine-tuning) on Windows.

### 4.6 Cell 15's `SOURCES` still points at leaked data — P2
Fixed by §2.2 + the swap.

---

## 4b. Span-level training — the idea worth testing next

**The proposal:** stop collapsing CODIPAS's spans into one label per message. Train
on the spans directly, since each span is text with a *directly human-assigned*
label.

**Why it's attractive**
```
span-level rows : 5,055   (vs 3,277 messages — +54%)
  No distortion : 2,778   (55%)
  distorted     : 2,277
per class: Labeling 398 · Mind reading 296 · Should statements 281
           Overgeneralization 265 · Emotional reasoning 262 · Fortune-telling 247
           Magnification 154 · All-or-nothing 138 · Mental filter 121 · Personalization 115
```
- **Zero derived labels.** The aggregation rule in
  `make_splits_codipas_classification.py` disappears entirely — no most-frequent
  vote, no tie-breaks. Every label is a human judgement about that exact text.
- 54% more examples, and `Emotional reasoning` (262) — the class PatternReframe
  lacks — is present.

**The objection in the current docstring does NOT apply.** It says feeding the
same message repeatedly with different labels would train against contradictory
targets. True — but that is an argument against feeding the *message* n times, not
against feeding the *span*, which is different text each time.

**Three real objections**
1. **Context loss.** Span median is 21 words vs 129 for the message (span ≈ 36% of
   its message). "He has tried a number of medications and none of them have
   worked" — is that Overgeneralization? You need the surrounding facts to weigh
   the thought against, which is precisely the Diagnosis-of-Thought argument. A
   21-word fragment often carries the thought and none of the evidence.
2. **Deployment mismatch.** At inference EmpowerLens gets a whole reflection, not a
   pre-extracted span — so a span-trained model meets inputs it never saw, unless
   a span extractor is built first (a harder task).
3. **Comparability.** The paper and `Annotated_data.csv` are message-level. Span
   results compare to neither, losing the replication anchor.
4. 322 exact `(message, span)` duplicates to dedupe.

**The version to actually build: span WITH its message as context.** Standard
span-classification — input is the full message with the target span marked,
output is that span's label:
```
[CLS] ...full message... [SEP] ...target span... [SEP]  ->  Overgeneralization
```
Keeps all three properties: 5,055 examples, zero derived labels, full context
retained. And it produces span-level output directly, which is what the dashboard
needs for highlighting (§5).

`Annotated_data.csv` has 1,597 spans in `Distorted part` too, so the same design
works on both datasets and comparability is restored.

**You are not starting from a blank page.** `src/make_splits_codipas.py` already
exists and is unused — it produces **span-level** splits (`data/splits_codipas`,
5,055 rows) and, critically, already solves the hardest correctness problem:

> *Splitting rows independently would put near-duplicate or literally identical
> source text in both train and test — a leakage bug worse than the one already
> fixed in cd_pipeline.py, since here it would leak the exact same input text
> across splits, just with a different span highlighted.*

It splits at the **group level** (`Id_Patient_Question`), so every span from one
message lands in the same split. That is exactly the hazard that produced the
`splits_combined` disaster, anticipated and prevented.

What it does *not* do is pair each span with its parent message as context, which
is the version worth building (a bare 21-word span usually lacks the facts needed
to judge the thought). So the work is: extend that script to emit
`(message, span, label)` rather than span-only, and add a two-segment input path
to `encode_texts`.

Note `data/splits_codipas` has **never been generated** — only the message-level
`data/splits_codipas_cls` exists, which is what every CODIPAS result so far used.

**Cost:** extend `src/make_splits_codipas.py` (group-level splitting already done),
plus a two-segment input path in `encode_texts`. `train_transformer.py` otherwise
unchanged. **Sequence it after the re-run** — there are already three open
experiments (CODIPAS cascade, flat baseline, PatternReframe).

---

## 4c. PatternReframe — available, strong fit for the data-volume problem

**Download (the `parl.ai` URL redirects twice; use the final one):**
```
https://dl.fbaipublicfiles.com/parlai/reframe_thoughts/reframe_thoughts_v0.1.tar.gz
2,482,759 bytes (2.4 MB) · sha256 bfbfc61c26341dd64b59945c3d290caba67fa2db435fb01ac309cef295222c99
```
The GitHub page (`facebookresearch/ParlAI/projects/reframe_thoughts`) has only a
README — no data. ParlAI is **not** required; the tarball is public.

Maddela, Ung, Xu, Madotto, Foran & Boureau (2023), *Training Models to Generate,
Recognize, and Reframe Unhelpful Thoughts*, ACL 2023. arXiv:2307.02768.
~10k thoughts + ~27k positive reframes, persona-conditioned.

**Taxonomy overlap is 9 of 10:**

| PatternReframe | EmpowerLens |
|---|---|
| Catastrophizing | Magnification |
| Overgeneralizing | Overgeneralization |
| Personalization | Personalization |
| Black-and-white | All-or-nothing thinking |
| Mental filter | Mental filter |
| Mind Reading | Mind Reading |
| Fortune Telling | Fortune-telling |
| Should statements | Should statements |
| Labeling | Labeling |
| Discounting the positive | *(no direct match)* |
| *(none)* | **Emotional Reasoning** |

**Measured after downloading — the taxonomy table above is right, but the
"4-9x more data for the rarest classes" claim needs three big qualifications:**

| | |
|---|---|
| total thoughts | 9,688 (official split is test-heavy: 1,920 / 961 / 6,807 — ignored; all become train) |
| after mapping | **8,712 rows** usable — 970 dropped ("Discounting the positive" has no counterpart, and merging it into `mental_filter` would corrupt that class) |
| **No-Distortion rows** | **ZERO.** Cannot help Stage 1 (binary needs negatives). An exact structural fit for **Stage 2 only** |
| **emotional_reasoning** | **ZERO coverage.** The key exists in `marked_patterns` but its intensity is 0 in all 9,688 rows — so 9 of your 10 classes get augmented and one does not, making it *relatively* rarer |
| median thought length | **17 words** vs **129** in Annotated — the biggest risk by far |
| `marked_patterns` | graded intensity 0-5 across 11 patterns; **93%** of thoughts carry 2+ |

`src/make_splits_patternreframe.py` handles all of this. It builds the multi-label
columns from `marked_patterns` rather than the single primary label, with
`--min-intensity 3` as the default because that yields **1.48 labels/row** —
closest to Annotated's 1.3. (`>=1` gives 3.55, `>=2` 1.83, `>=4` only 0.61.)
Getting this wrong makes the *label structure itself* a distribution shift.

```
python -m src.make_splits_patternreframe --source <extracted-dir>     --merge-into data/splits_stage2 --out data/splits_stage2_pr --force
```

Takes Stage 2 from **1,278 -> 9,990 train rows (7.8x)**; val/test pass through
untouched, per "train augmented, test natural".

**The honest experiment:** train Stage 2 on the merged set, evaluate on the
unchanged Annotated test set. If the 17-vs-129-word gap dominates, this will not
help — and that is itself a result about what shape the synthetic pipeline's
output needs to be. The 27k reframes are a separate, currently unused asset for
the reframing half of the product.

---

## 5. Distorted-span work (not started)

### 5.1 The spans exist and are being thrown away — P2
`Annotated_data.csv` has `Distorted part`, **1,597 of 2,530 rows** populated. It
survives into `data/splits` but is **dropped** in `splits_combined` and
`splits_stage2` (`KEEP_COLS` in `make_splits_cascade.py` excludes it).

**No model in the repo is trained to predict spans.** All tasks are
sentence-level classification. `explain.py` approximates it post-hoc with Captum
attribution — "which words moved the classifier", not "which span a clinician
would mark" — and never sees the gold spans.

### 5.2 Free evaluation you're not doing — P2
1,597 gold spans + `explain.py` = measure token overlap between highlighted words
and the gold span. **No training, no GPU.** Turns "we highlight the distorted
part" into a number, and attribution methods are usually reported *without*
ground truth to check against.

### 5.3 CODIPAS is the real span asset — P2
**4,066 annotated spans** across 2,621 messages (vs Annotated's 1,597); raw
`CODIPAS.json` is in the repo. Currently flattened into message-level labels.

But for *classification*, CODIPAS's `y_mc` is **derived, not annotated** — your own
docstring says so, and 714 of 3,277 messages have 2+ distinct dominant types
across spans, so the mode-with-tiebreak rule decides ~22% of cases. Better span
annotation, weaker message-level labels.

---

## 6. Google Form — before distributing

**Content**
1. **Sector → multiple choice** (7 options + Other), so it maps to `SECTORS`
2. **Add Reflection 4 — family/community pushback.** Highest-value addition;
   nothing currently elicits `gender_expectation` or `family_pressure`
3. **Add an open recall question** — "a setback you actually faced". The existing
   vignettes are hypothetical, and hypotheticals elicit the composed register,
   i.e. the No-Distortion class you already have 933 examples of
4. **Delete R2's priming sentence** — "You feel a sense of profound disappointment"
5. **Add a language invitation** — "English, Urdu, or a mix"
6. **Add an optional gender question** with *Prefer not to say*
7. **Reorder stage options** into lifecycle order
8. **Fix the required-vs-skip contradiction** — consent promises skipping, but the
   reflections are required with "type skip" as the workaround

**Settings**
9. **"Limit to 1 response"** is the likely cause of the sign-in wall
10. **Confirm it is not domain-restricted to FAST-NU**, or external respondents
    silently cannot submit. Test from a personal account.

**Annotation after collection:** two annotators independently → Cohen's kappa on
the dominant label → resolve, keeping per-annotator labels. ~200 rows ≈ 4–5 hours
each. **Never let the LLM critics label the real responses.** Keep the result in
`data/real/`, never merged into `data/splits/`.

---

## 7. Later: dataset combination experiments

Three cells, not five. Replay and EWC solve a problem you don't have — they exist
for when you *can't* keep the old data; yours fits in memory.

| | Train | Test |
|---|---|---|
| baseline | Annotated | Annotated |
| joint | Annotated + CODIPAS (**deduped**, §4.1) | Annotated |
| sequential | CODIPAS → Annotated | Annotated |

**Order matters: CODIPAS first, Annotated last.** The model specialises on what it
saw most recently, and Annotated test is what you report.

Sequential needs **no code change** — `--model` goes straight to
`from_pretrained()`, so a local checkpoint path works with a lower `--lr`.

---

## Fixed this session (reference)

**The one that mattered: `sh()` deadlocked on a full pipe buffer.**
It passed no `stdout`/`stderr`, so the child inherited the kernel's descriptors
and wrote into a pipe with a ~64KB OS buffer that nothing drained while the parent
sat in `proc.wait()`. Once training emitted 64KB — around the "Loading weights"
bar — the child blocked on `write()` forever and the parent blocked on the child.
Reproduced and fixed with a reader thread. This explains the flakiness (same seed
working then hanging — it's a race), why `!python ...` always worked (Jupyter
drains as it goes), the GPU at 0%, and very likely run 2's original 11.4-hour hang.

**Three wrong diagnoses before that**, recorded so they aren't repeated: Hub
freshness check (no evidence), `HF_HUB_OFFLINE=1` (which *created* a new failure —
an `allow_patterns` allowlist skipped files, then offline mode made them
unfetchable), and slow network. All were inferred from where output stopped rather
than from a traceback. Running the command directly with `!` and interrupting is
what localised it.

**Other fixes**
- Cell 5 shadowed the bootstrap with the leaked splits dir → Stage 2 was rebuilt
  from `splits_combined`. All local reassignments removed.
- `evaluate.py` guard renamed `tag`, which broke resumability and made completed
  seeds report as failures. Now only `model_name` is tagged.
- `evaluate_cascade.py` embedded the seed in the model name → every seed was its
  own group → mean ± std came out `NaN`.
- Checkpoints moved to `/kaggle/working/checkpoints`, outside the repo clone that
  cell 1 wipes. `run_and_report` gained `ckpt_dir` so a flat multilabel run can't
  overwrite Stage 2.
- `meta.json` now records `splits` — the provenance question that couldn't be
  answered for the earlier checkpoints.
- Bootstrap (`notebooks/cascade_bootstrap.py`) puts config on disk so a kernel
  restart can't produce `NameError: name 'STAGE2_SPLITS' is not defined`.
- Speed: `--max-length 256 --truncation head_tail --batch-size 32` plus
  `LengthGroupedSampler` wired manually (transformers 5.x removed
  `group_by_length`). ~90s/seed for Stage 1, ~60s for Stage 2.
- `data/splits_stage2` regenerated from clean `data/splits` — 0 leakage.

**Synth pipeline** (earlier in the session): `max_tokens` 400→800, KoACD
Appendix G/H constraints, selective Tier-2 marker bans, two granularities,
scenario × distortion grid (154/154 cells), `Unknown` split from `No Distortion`,
mandatory evidence enforced in code, rejects log with abandonment / taxonomy /
calibrated-pessimism analysis, opener tracking at 1/2/4 words.
