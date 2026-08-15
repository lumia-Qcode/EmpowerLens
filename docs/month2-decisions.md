# Month 2 — decisions, open questions, and the data-collection plan

Working notes for the synthetic-data + real-collection track — the *why* and the
*what's still undecided*.

- Mechanics, knobs, run commands → [`src/synth/README.md`](../src/synth/README.md)
- Research provenance, borrowed vs. original, honest limitations → [`src/synth/PIPELINE.md`](../src/synth/PIPELINE.md)

Last updated: 2026-08-09.

---

## 1. Status

| | State |
|---|---|
| `src/synth/` pipeline | Built, dry-run clean, **never run against a real API** |
| SDKs (`anthropic`, `google-genai`) | Installed in `venv` |
| API keys | **Not set** — blocks everything |
| Calibration run (`--n 20`) | Not started |
| Google Form | Live, needs the changes in §5 before distribution |
| Seed set | 13 authored rows — **scaffolding, not grounded data** (§3.2) |

---

## 2. Cost model

Measured from prompt sizes, not guessed.

| Call | Input | Output | Rate ($/MTok) | Cost |
|---|---|---|---|---|
| Generator (Sonnet 5) | ~1,050 | ~250 text + ~200 thinking | 2 / 10 | ~$0.0067 |
| Critics ×3 (Gemini 2.5 Flash) | ~770 each | ~200 each | 0.30 / 2.50 | ~$0.0022 |

**~$0.009 per spec**; a revision roughly doubles that row.

| `--n` | USD | PKR @ 278 |
|---|---|---|
| 20 (calibration) | ~$0.20 | **~₨56** |
| 200 | $2.20–2.60 | ₨610–720 |
| 300 | $3.30–4.00 | **₨920–1,110** |
| 1000 | $11–13 | ₨3,060–3,615 |

**The revision rate is the dominant uncertainty** — cost per spec ≈
`$0.0089 × (1 + revision_rate)`, so 300 specs is anywhere between ₨740 (no
revisions) and ₨1,485 (every row revised). The `--n 20` run resolves it.

**Three things that move the number:**

1. **Sonnet 5's introductory price expires 2026-08-31.** Standard rate is then
   $3/$15, +50% on the generator half (~+37% overall). `PRICES` in `llm.py`
   hardcodes the intro rate and must be updated or the printed cost lies.
2. **`max_tokens` was raised 400 → 800.** On Sonnet 5 adaptive thinking is on by
   default and its tokens count against `max_tokens`; at 400 a 150-word
   reflection left ~150 tokens for thinking and got truncated mid-sentence,
   failing the word-count filter. Billing is on tokens produced, not the cap, so
   the headroom is free.
3. **Prompt caching now actually works.** The generator system prompt grew past
   Sonnet 5's 1,024-token minimum when the KoACD constraints were added, so the
   `cache_control` block that was previously a silent no-op now caches.

Practical note: Anthropic console top-ups have a minimum (typically $5 ≈ ₨1,390),
and a Pakistani card will add ~1–3% foreign-transaction fee.

---

## 3. Decisions made

### 3.1 Train balanced, test natural

**Synthetic data is class-balanced; the test set stays at the natural
distribution.** Write this into the methodology explicitly — the field's biggest
paper blurs the distinction, so stating it earns credit.

Already enforced structurally: splits are frozen from real data, synthetic rows
only ever join `train`, and `test.csv` is read solely by `src/evaluate.py`.

**Consequence to disclose:** with 30–60 respondents on a natural distribution,
rare classes may land 2–3 test rows and their per-class F1 is noise. Report
support counts per class, and report macro-F1 over adequately-supported classes
alongside the full macro. Do not engineer the form to balance classes — that
would break the thing that makes the evaluation honest.

### 3.2 The current seed set is scaffolding

All 13 seeds carry `source: authored_from_*`. They were written, not collected.

- **What they genuinely anchor:** format — length, first person, no
  meta-commentary, distortion in the reasoning rather than announced. Structural,
  so it transfers regardless of author.
- **What they cannot anchor:** register. If a model wrote them and a model
  generates from them, the "voice anchor" anchors to that model's impression of
  the voice. Closed loop.

**Plan:** after the form collects responses, *humans* read them and hand-write
25–30 new seeds capturing the observed register. Same method the seeds README
already documents for public founder posts — read the pattern, re-author, no
verbatim reproduction. Register gets grounded in real language, no participant
text is transmitted to any API, no consent change needed, and the circularity
breaks.

**What limits the damage meanwhile:** the critic panel is a different model
family from the generator, and the test set is real human text. Circular seeds
degrade generation *quality*; they cannot corrupt *evaluation*. A bad textbook
lowers your exam score — it doesn't fake it.

**Testable, ~₨110:** generate 20 rows at `k=4` and 20 at `k=0` (`k` is the
few-shot count in `build_generator_user`), then compare both against real
responses on length and vocabulary overlap. If the seeds move nothing, they're
decoration.

### 3.3 Participant data stays local

Form responses are the **gold test set only**. They are not sent to Anthropic or
Google as few-shot exemplars.

Rationale: the current consent says responses will "help train and improve the
tool" — it does not disclose third-party AI services. Keeping participant text
out of API calls avoids both the consent gap and the risk of "synthetic" rows
being lightly-reworded participant data.

### 3.4 Critic sees full text and emits the span

KoACD sends only the distorted part to the critic. Rejected, because DoT stage 1
(separate facts from thoughts) has nothing to work on given a bare span, and the
whole grounding argument is that real distortions sit inside a paragraph about a
specific situation.

Instead the critic keeps full context and **outputs** `distorted_span`. This
yields the `Distorted part` column that `Annotated_data.csv` has and synthetic
rows previously lacked, plus `span_agreement` — critics agreeing on the label
while pointing at different sentences marks a diffuse row.

### 3.5 `Unknown` is separate from `No Distortion`

KoACD's Analyzer prompt carries an enumerated list of conditions forcing an
immediate "Unknown" verdict. We copied the structure but **split the verdict in
two**, because "I cannot judge this" and "I judged this and it is sound" are
different claims.

Without the split: the generator writes *"I feel awful today. Everything feels
heavy."* — pure emotion, no reasoning. The critics have no honest option except
No Distortion, `decide()` accepts it, and the row enters training as an example
of healthy thinking. It isn't healthy thinking; it's **no** thinking. Enough of
those and the model learns *"short emotional text = No Distortion"* rather than
*"sound reasoning = No Distortion"* — and then misfires on exactly the reasoned,
gloomy reflections D1 is about.

Two consequences:
- `UNKNOWN_CONDITIONS` and `NO_DISTORTION_CONDITIONS` are enumerated separately
  in `attributes.py`. The **calibrated-pessimism guardrail** lives on the second
  list as a formal exit condition, not a passing note.
- Abandonment is counted three ways — read as distorted / judged Unknown /
  dropped before the panel. **Only the first is taxonomy evidence.** Unknown
  rejections are excluded from the confusion tally, since they indicate a
  generation failure rather than a label that fails to fit founder reasoning.

### 3.6 Evidence quoting is enforced in code

Critics must quote ≥2 verbatim sentences (or the whole text if shorter).
`_critique()` checks this rather than trusting the prompt: a confident label
returned with fewer than two quoted sentences is **coerced to `Unknown`** and
flagged `coerced_unknown`. A label asserted without the evidence it was told to
produce is the speculative verdict the rule exists to prevent.

Side benefit: `evidence` and `critic_reasoning` land on every accepted row, which
is free span-level explainability material to sit alongside SHAP/Captum later.

### 3.7 Exclusion memory deliberately NOT implemented

KoACD's agents negotiate across turns and can loop by re-proposing a rejected
label. This pipeline has no multi-turn loop: the critics are one-shot stateless
calls with no memory of their own prior verdict, and there is exactly one
revision. There is nothing to loop, so the mechanism would be dead code.

Revisit only if the revision count is ever raised above one.

### 3.8 No `[sector / stage]` prefix in the text

KoACD's `[Gender/Age]` template was a generation control, not a training feature.
`sector` and `stage` are already JSONL columns, which gives the same slicing
without teaching the classifier that reflections begin with a bracketed tag.

---

## 4. Open decisions

| | Question | Why it matters now |
|---|---|---|
| **A** | Multi-model critic panel? | All three critics are currently `gemini-2.5-flash` at different temperatures — self-consistency, not independence. **Decide before calibrating**, or the measured rates won't transfer. `--critic-model` would need to accept a comma-separated list. |
| **B** | Shift `GRANULARITIES["reflection"]` to ~90–130 words? | The form asks for 60–120. Training on 30–50 and testing on 60–120 means part of the measured domain-shift drop is *length*, not content. |
| **C** | Write `src/synth/leakage_check.py`? | Keyword-only baseline over the synthetic set. If a hand-written keyword list scores well above chance, leakage remains and the data should be regenerated. If it scores near chance while the fine-tuned model scores well, that's proof the model learned structure — one table that pre-empts "your synthetic data is trivially separable". |
| **D** | Record §3.1 in `CLAUDE.md`? | So the convention survives future sessions. |

---

## 5. Google Form — required changes

Form: *EmpowerLens — Entrepreneur Reflection Study*. Currently collects consent,
venture stage, sector (free text), and three vignette reflections (funding
rejection / missed survey timeline / negative reviews).

### Content

1. **Sector → multiple choice**, replacing free text:

   ```
   Food & catering (home kitchen, bakery, packaged food, restaurant)
   Fashion & clothing (stitching, boutique, textiles, accessories)
   Handicrafts & home décor (embroidery, artisan goods, furnishings)
   Beauty & wellness (salon, skincare, cosmetics)
   Technology (software, app, online platform, IT services)
   Services (consulting, events, logistics, agency, professional)
   Education & training (tutoring, courses, skills training, edtech)
   Other (please specify)
   ```

   Maps 1:1 to `SECTORS`. The parentheticals matter — a woman running an
   embroidery business may not recognise herself in the word "handicrafts".

2. **Add Reflection 4 — family / community pushback.** Highest-value addition;
   nothing currently elicits `gender_expectation` or `family_pressure`, which is
   the dimension that makes the dataset unique.

   > You mention at a family gathering that you're putting your savings into your
   > business instead of taking a stable job. A relative comments that this is not
   > a serious path for a woman your age, and others nod along.
   >
   > *How do you interpret this reaction? How does it sit with you as you think
   > about your business?*

   Expected coverage — strong on **Mind Reading** ("others nod along" is a direct
   probe) and **Should statements** (the scenario *is* a rule); moderate on
   Magnification, Emotional Reasoning, Labeling, Fortune-telling.

3. **Add an open recall question:**

   > Think of a setback you actually faced in your business in the last few
   > months. What happened, and what was going through your mind at the time?

   The three existing vignettes are hypothetical, and hypotheticals elicit the
   composed register ("I'd take it as feedback and iterate") — i.e. the
   No Distortion class, of which there are already 933 examples. Distorted
   thinking shows up in *recall*, not projection. This question also surfaces
   triggers not on the list.

4. **Delete the priming sentence in R2** — *"You feel a sense of profound
   disappointment."* R1 and R3 don't tell the respondent how to feel.

5. **Add a language invitation** to the reflection section header — *"Please
   write however you naturally would — English, Urdu, or a mix of both."* An
   academic-looking form primes formal English, which is not the target register.

6. **Add an optional gender question** with *Prefer not to say*. The generator
   prompt commits to women's voice; a mixed incubator makes responses
   unattributable to the target population.

7. **Reorder stage options** into lifecycle order — Just an idea / Building the
   product/service / Launched, early customers / Growing–established / Prefer not
   to say. "Building" currently sits last, after "Growing".

8. **Fix the required-vs-skip contradiction.** The consent promises "you may skip
   any question", but sector and all three reflections are **required**, with the
   offered workaround being to type "skip" — which also pollutes the dataset with
   literal `"skip"` strings. Make the reflection questions optional.

### Settings

9. **"Limit to 1 response"** is the likely cause of the sign-in wall. Anonymity
   survives (email isn't collected), but every respondent needs a Google account
   and an extra mobile step. Decide deliberately.
10. **Confirm the form is not domain-restricted to FAST-NU** — if it is, external
    respondents cannot submit at all, silently. Test from a personal account.

### Coverage after the changes

| | R1 funding | R2 timeline | R3 reviews | R4 family |
|---|---|---|---|---|
| Mind Reading | ●● | | | ●●● |
| Should statements | | ●● | | ●●● |
| Mental filter | | | ●●● | ● |
| Fortune-telling | ●● | | ●● | ●● |
| Magnification | ●● | | ●● | ●● |
| Labeling | ●● | ●● | | ●● |
| Personalization | ● | ●●● | | ● |
| Overgeneralization | | ●● | | ● |
| All-or-nothing | | ●● | | ● |
| Emotional Reasoning | ● | ● | ● | ●● |

Emotional Reasoning stays thin. **Leave it thin** — see §3.1.

---

## 6. Failure modes to watch

Two kinds, very different economics:

**Mode A — caught and rejected.** Wrong length, non-English, literal duplicate,
or critics don't agree. Visible in the counters, bounded, ~₨1,500 worst case at
n=300.

**Mode B — bad text that gets accepted.** Artificial, formulaic, repeated
openers — but it *does* show the target distortion, so 3/3 critics vote yes.
Nothing in the pipeline checks whether text sounds human, and there's a perverse
incentive: **blatant writing is easier for critics to agree on**, so acceptance
rate partly measures how obvious the prose is. A 95% acceptance rate is more
worrying than 70%.

Measured evidence that the dedup filter won't catch Mode B:

| Case | Jaccard | Caught at 0.8? |
|---|---|---|
| Shared opener, different body | 0.159 | No |
| Near-verbatim paraphrase | 0.762 | **No** |

Mode B costs more than the API bill — Kaggle GPU hours on data that won't
transfer, plus a validation score that looks fine because synthetic-validated-on-
synthetic hides it. The real test set is the circuit breaker, but it fires *after*
the GPU time.

**Mitigation:** iterate at `--n 20` (₨56/round), not `--n 300` (₨1,000/round).
Read the 20 reflections. Then three checks, none needing an API call:

1. Distinct openers — count unique first-5-word sequences
2. Max-pairwise Jaccard *distribution*, not just the threshold
3. Vocabulary and length distribution against `Annotated_data.csv`

Rejected rows are cheap; **accepted-but-bad rows are recoverable by filtering the
JSONL, which costs nothing.** Only a prompt-level fix requires re-generating.

---

## 7. Next actions

1. Set `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`
2. Resolve open decision **A** (critic panel) — before calibrating
3. Run `--n 20` calibration (₨56), read all 20 rows
4. Apply the §5 form changes, test submission from a personal Google account
5. Distribute the form
6. After collection: hand-author 25–30 grounded seeds (§3.2), then the pilot run
