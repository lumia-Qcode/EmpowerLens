# Task briefs — week of 28 Sep 2026

Companion to `tasks_week_2026-09-28.xlsx`. One brief per task: what it means, how to
do it, when it is done, and what goes wrong. Grouped by module; the spreadsheet is
ordered by deadline.

**Standing rules that apply everywhere**

1. Augmentation is **train only** — never val, never test.
2. An augmented row stays in its source row's split. A paraphrase of a train row
   appearing in val or test is leakage and silently inflates every number.
3. Form and interview responses are **evaluation only** — never prompt content,
   never training data.
4. The test set is never balanced. It stays at its natural class distribution.
5. Thresholds and epochs are chosen on **val**. Test is read once, at the end.
6. The challenge set is evaluation only, and frozen once verified.

---

## Module: Re-annotation

Fixing the label noise in the benchmark. The original corpus had 33.7% annotator
agreement on distortion type (61% on the binary decision), so label quality is the
ceiling on everything downstream.

### T01 — Label CS workbook
**Team · Sat 03 Oct · no dependencies**

Fill `annotator_4_CS_majors.xlsx`: 413 rows (60 overlap + 293 training + 60 test).
Read the codebook first and keep it open.

**Done when:** Primary filled for every row attempted, workbook returned unrenamed.

**Watch out:** the first 60 rows are the overlap set and are the *only* rows that
produce agreement statistics. They must be labelled independently, with no discussion
with the other annotators, before or during. One conversation about a shared row and
the kappa is meaningless.

### T02 — Label psych workbooks
**Team · Sat 03 Oct · no dependencies**

Laiba 421 rows, Reesha 418, Hurema 415. About 5h 10m each at 45 s/row.

**Done when:** three workbooks returned.

**Watch out:** unfinished rows are fine — the codebook says so. Rushing the last
hundred is worse than leaving them blank, because bad labels are harder to detect
than missing ones. Rows marked Unsure go to T04.

### T03 — Merge labels + agreement
**Lumia · Sun 04 Oct · needs T01, T02**

```
venv\Scripts\python.exe reannotation_2026-09-27/scripts/relabel_agreement.py
```

Put the returned workbooks back in `outputs/relabel/` first, with their original
filenames. Produces `merged_labels.csv`, `agreement_report.md` and
`unsure_for_adjudication.csv`.

**Done when:** the report exists and the invalid-row list is empty or explained.

**Watch out:** the report gives Fleiss' and Cohen's kappa, which are chance-corrected.
The original paper's 33.7% is a *joint probability of agreement*, which is not
chance-corrected and is therefore a higher number for the same quality of labelling.
Do not present ours as a regression against theirs; say which metric each is.

### T04 — Adjudicate Unsure rows
**Izza · Tue 06 Oct · needs T03**

Work through `unsure_for_adjudication.csv`. The codebook routes these to the
psychology annotators.

**Done when:** every Unsure row has a final label, or is explicitly dropped with a
reason recorded.

**Watch out:** a row nobody can label confidently may be genuinely ambiguous. Dropping
it is a legitimate outcome and better than forcing a label — but record the count,
because it is a finding about the dataset.

### T05 — Separate errors from redefinitions
**Lumia · Tue 06 Oct · needs T03**

Every changed label falls into one of two groups:

- **Genuine error** — the original label was wrong under the original definition.
- **Redefinition** — the label changed because our codebook added the
  calibrated-pessimism rule, which the original annotation did not have. Shreevastava
  and Foltz state that "each entry is perceived as a viable candidate for cognitive
  distortion" and that they had no way to verify the truth of any statement.

**Done when:** a count of each, and the IDs of the second group.

**Watch out:** this is the difference between "we found N label errors" (a claim about
the dataset) and "we changed N labels" (a claim about our own definition). Conflating
them overstates the finding, and it is the first thing a careful examiner will ask.
It also means results on relabelled data are **not** comparable to the published
0.79 binary / <0.30 multi-class figures. Say so wherever those numbers appear.

---

## Module: Data augmentation

Under-represented classes (All-or-nothing 100 rows, Should statements 107, Mental
filter 122) get inflated by back-translation so the classifier sees more of them.

### T07 — Decide span handling for augmented rows
**Lumia · Wed 07 Oct · needs T11**

*Moved before T06: this is a design decision that augmentation depends on, not a
follow-up to it.*

`Distorted part` is a verbatim quote into `Patient Question`. Translate the text and
that quote no longer matches, so `find_span` fails and the row cannot be used for span
training. Two options:

- **Re-translate the span separately**, then re-locate it in the back-translated text,
  accepting fuzzy matches. Keeps the row usable for span training; costs a match rate.
- **Mark augmented rows classification-only.** Simpler and honest; the span selector
  then trains on original rows only.

**Done when:** the choice is written down and reflected in the augmentation script's
output schema (a flag column saying whether the row has a usable span).

### T06 — Back-translation augmentation
**Lumia · Thu 08 Oct · needs T03, T04, T07**

Translate each row out to another language and back (en→de→en, en→fr→en), producing a
paraphrase that keeps the distortion label.

**Done when:** an augmented train split exists with a `source_id` column linking every
augmented row to its original, and per-class counts before and after.

**Watch out:**
- **Train only.** Augmenting val or test invalidates the evaluation.
- **Same split as the source row.** This is why `source_id` is mandatory.
- **It must run after T03/T04.** Augmenting before the labels are final means
  inflating labels that are about to change.
- Do not augment to perfectly equal class counts. Mild rebalancing helps; forcing
  uniformity teaches the model a prior the real world does not have.

### T08 — Hand-check augmented samples
**Izza · Fri 09 Oct · needs T06**

Read 30 augmented rows per class and ask: does the distortion still hold?

**Done when:** a pass rate per class, and any class below ~90% is re-generated or
dropped.

**Watch out:** translation hedges. "I'll never get funding" (fortune-telling) can come
back as "I might not get funding", which is not a distortion at all. Absolutes
(*always, never, everyone*) and identity labels are the most fragile, which means
exactly the classes you are trying to inflate are the ones most likely to be damaged.

---

## Module: Transfer evaluation

Does a model trained on TherapistQA work on anything else? This is the first external
evidence that the model learned distortions rather than one dataset's quirks.

### T09 — CBT-Bench: acquire, map labels, run the org-dataset model
**Izza · Wed 30 Sep · no dependencies**

`Psychotherapy-LLM/CBT-Bench` on HuggingFace; use the **CBT-CD** split — 146 examples,
10 distortion categories. Paper: CBT-Bench (NAACL 2025), arXiv 2410.13218.

Write an explicit mapping from their category names to ours, in the style of
`LABEL_CANON` in `src/data.py` — no fuzzy matching, unmapped values raise.

**Done when:** the mapping file exists, the number of dropped items is recorded, and
the current fine-tuned model has been scored on it.

**Watch out:** it was built to evaluate LLM prompting, not to fine-tune on. With 146
items over 10 classes (~15 each), per-class numbers are extremely noisy. Report
support counts beside every score.

### T10 — Evaluate the relabelled + augmented model on CBT-Bench
**Izza · Sat 10 Oct · needs T06, T08, T09**

*Date moved from Fri 02 Oct: it needs the augmented data, which only lands Thu 08 Oct.*

Same protocol as T09, on the model trained with relabelled + augmented data. The
comparison against T09's score is the result: did cleaning and augmenting the training
data improve transfer?

**Done when:** both scores sit in one table with the same metrics and support counts.

**Watch out:** on 146 items, a difference of a few points is noise. Report three seeds
as mean ± standard deviation, and if the intervals overlap, say the result is
inconclusive rather than claiming an improvement.

---

## Module: Span selection

Teaching the model to point at the sentence carrying the distorted reasoning. This is
the explainability claim in the report, and our strongest early result
(macro-F1 0.322 → 0.402 when the model reads only the distorted part).

### T11 — Span approach: literature + decision
**Lumia · Wed 30 Sep · no dependencies**

Compare three designs:

| Approach | How | Trade-off |
|---|---|---|
| Sentence classification | Split into sentences, classify each as carrying the distortion | Simplest; matches what the report promises |
| BIO token tagging | Tag every token B/I/O via `return_offsets_mapping` | Finer-grained; noisier; needs reassembly rules |
| Extractive QA | Predict start and end token, SQuAD-style | Clean when there is exactly one span |

Search terms: *token classification BIO tagging rationale extraction*, *extractive
rationale prediction*, *sentence-level evidence selection*.

**Recommendation: sentence classification.** Shreevastava and Foltz say annotators
were "asked to flag the sentences that led them to conclude that the reasoning was
distorted" — so the supervision is sentence-level, and a trainer reads sentences.

**Done when:** a one-paragraph decision with a reason, which becomes report text.

### T12 — Build sentence labels from spans
**Lumia · Fri 02 Oct · needs T11**

Split each `Patient Question` into sentences; the sentence containing the gold span
gets label 1, the rest 0. `find_span` (in `src/data_codipas.py`, imported by
`train_transformer.py`) already returns character offsets, so this is offset
arithmetic.

**Done when:** a labelled sentence dataset exists, with the share of rows where the
span could not be located reported (exact / fuzzy / none, as `build_inputs` already
counts).

**Watch out:** a span sometimes crosses a sentence boundary, and naive splitting on
"." breaks on abbreviations and decimals. Decide the rule — e.g. label every sentence
the span overlaps — and record it.

### T13 — Train + evaluate the span selector
**Lumia · Thu 08 Oct · needs T12**

Train the sentence classifier, then chain it: **selector picks the sentence →
classifier reads only that sentence → distortion type**.

**Done when:** selector metrics (did it pick the gold sentence? character overlap with
the gold span) are reported *separately* from classification metrics, plus the
end-to-end score.

**Watch out:** the current `span_crop` / `span_marked` modes are **oracle** setups —
the code says so — because they hand the model the human span at test time. The 0.402
is therefore a ceiling, not a score. If the selector is right 70% of the time, the
end-to-end result will land between 0.322 and 0.402. Report the oracle as an upper
bound and the gap as the selector's cost; that framing is honest and makes the
contribution measurable.

---

## Module: Model tuning

### T14 — Hyperparameter tuning
**Izza · Fri 02 Oct · no dependencies**

Sweep on the original dataset, in this order of impact: learning rate
(2e-5 / 3e-5 / 5e-5), epochs, `--max-length`, weight decay.

**Done when:** a results table (config → val macro-F1) and a chosen configuration.

**Watch out:** selection on **val only**. With 253 val rows, a 1–2 point difference is
noise, so confirm the winner across all three seeds (42, 1337, 2024) before adopting
it. A single-seed win is usually luck.

### T15 — Over/underfitting diagnosis
**Izza · Fri 02 Oct · needs T14**

Log train and val loss every epoch and plot them.

**Done when:** loss curves plus a one-line diagnosis, and early stopping on val
macro-F1 wired into the training run.

**Watch out:** with ~2,000 training rows, expect val loss to turn upward from epoch
3–4 while train loss keeps falling — the standard overfitting signature. If *both*
stay high, that is underfitting: raise the learning rate or train longer before
blaming the model.

---

## Module: Entrepreneurial data

Building the domain dataset. This is the week's headline deliverable and the part that
depends on other people, so it carries the most schedule risk.

### T16 — Decide target dataset sizes
**Lumia · Tue 29 Sep · no dependencies · do this first**

Everything else in this module is sized by this decision.

- For a specificity estimate with roughly ±10% confidence: **~100 non-distorted items**.
- For anything per-class: **~20 per class → 200–300 items**.
- Seed data for generation needs far less: tens, not hundreds.

**Done when:** target counts are written down for the test set, the seed set and the
scrape.

**Watch out:** without this, scraping has no stopping condition and expands to fill the
week.

### T17 — Scraping: sources + ToS check
**Izza · Wed 30 Sep · needs T16**

Pick sources and check what each one's terms allow.

**Done when:** a short list of sources with a note on what may be stored and what may
be redistributed.

**Watch out:** most platforms allow analysis but not republication of post text. That
matters because "an entrepreneurial dataset" is a stated deliverable — you may be able
to publish derived data (labels, features, models) but not the raw posts. Decide now,
not after collection.

### T18 — Scrape entrepreneurial text
**Izza · Sat 03 Oct · needs T17**

Collect to the T16 target, clean, de-duplicate.

**Done when:** a raw text file with one row per item, a source URL or ID per row, and a
collection date.

**Watch out:** de-duplicate properly — reposts and cross-posts are common, and a
duplicate that lands on both sides of a split is leakage. Keep the source ID so any
item can be traced back or removed later.

### T19 — Circulate form + start thread
**Lumia · Wed 30 Sep · no dependencies · longest lead time**

Post to relevant entrepreneurial groups and sites; start collecting responses.

**Done when:** the form is live and posted in at least three places.

**Watch out:** consent and anonymisation text must be in place *before* it circulates.
Check whether the department requires ethics review — that has a multi-week lead time
and is the single likeliest thing to derail the project. Responses are evaluation
only: never prompt content, never training data.

### T20 — Triage form responses
**Lumia · Sat 03 Oct · needs T19**

Keep responses that are usable: actual reflections, not one-liners.

**Done when:** a kept/discarded count with the discard reasons.

**Watch out:** do not balance. If the responses come back mostly non-distorted, that
*is* the natural distribution and it is what makes the specificity claim meaningful.
Per-class coverage is bought from the supplementary set, not by filtering this one.

### T21 — Compile the entrepreneurial dataset (unlabelled)
**Team · Sat 03 Oct · needs T18, T20 · WEEK GOAL**

Merge scraped and form text into one schema, ready for labelling.

**Done when:** one file whose columns match what the labelling workbooks will need —
an ID, the text, the source, the collection date, and the intended split role
(evaluation / supplementary / seed).

**Watch out:** record the split role **now**, at compile time. Form responses are the
natural-distribution evaluation set; scraped text is supplementary. Deciding later,
after seeing the labels, is how evaluation sets quietly become biased.

---

## Module: Specificity set

The justified-concern challenge set: the evidence behind the project's central safety
claim — that we do not flag realistic business worries as distorted thinking.

### T22 — Seed the justified-concern set
**Nayab · Mon 05 Oct · needs T03**

Pull every row the annotators marked `No Distortion (calibrated pessimism)`. The
agreement report already counts and lists them.

**Done when:** a seed file of real, human-verified justified concerns.

**Watch out:** these are clinical in topic ("I failed the last two driving tests"), so
they are a model of the *reasoning pattern*, not items for the final set. They need
rewriting into entrepreneurial situations.

### T23 — Write challenge-set pairs
**Team · Fri 09 Oct · needs T22**

For each distortion family, write ~20 matched pairs: the same scenario once as a
justified concern and once as a genuine distortion.

> *Justified:* "We lost two of eight clients this quarter, so retention is the thing to fix."
> *Distorted:* "We lost two clients. I clearly can't keep anyone happy."

**Done when:** ~200 pairs, each verified blind by two annotators, with an
`evidence_note` recording what fact makes the justified version reasonable. Then
frozen and versioned (`data/challenge_justified/v1/`).

**Watch out:** matched pairs give you controls for free — without genuinely distorted
items in the set, a model that flags nothing scores 100% specificity and looks perfect.
If items are generated rather than written, use a different model and prompt from the
training generator, or you are testing the model on its own generator's style.

### T24 — Style-leak check on the challenge set
**Nayab · Sat 10 Oct · needs T23**

Train a plain TF-IDF + logistic regression on the challenge set alone, justified vs
distorted. You want it to score **near chance**.

**Done when:** the score is recorded, and items are rewritten if it separates them too
easily (say above 0.75 F1).

**Watch out:** this is the step that makes the set defensible. If bag-of-words can tell
the classes apart, the items differ in surface cues — length, numeric tokens,
vocabulary — and a transformer will pass your test without understanding any reasoning.
`src/baseline_classical.py` already has the pieces. Also compare mean word count and
numeric-token rate across the two halves and rewrite the outliers.
