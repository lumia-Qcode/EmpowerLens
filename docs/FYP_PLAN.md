# EmpowerLens — FYP plan (final trajectory)

**Written 2026-08-29.** Supersedes the roadmap fragments in `docs/TODO.md` and the
"Moving ahead — ranked" section of `docs/EXPERIMENTS.md` for anything beyond the
next GPU run. Scope decisions here are fixed unless a tripwire in §9 fires.

**Dropped from the original FYP scope:** the Urdu prototype, the reframing module.
This plan is detection only.

---

## 1. The claim

> In cognitive distortion detection, the **input representation matters more than
> the encoder**. On the standard benchmark, restricting the classifier to the
> annotator-marked distorted sentence is worth **+0.080** macro-F1 — more than
> twice the **+0.035** gained by swapping TF-IDF for a domain-pretrained
> transformer. We build a joint sentence-selection and multi-label classifier that
> learns to find that region on its own, add an explicit **warranted-concern**
> class absent from prior work, and evaluate on purpose-collected reflections from
> Pakistani women entrepreneurs — reporting **specificity** on a challenge set of
> negative-but-justified statements alongside F1.

The first sentence is already measured (§3). The rest is the work.

---

## 2. Constraints this plan is built around

| constraint | value |
|---|---|
| Time to submission | 10–16 weeks (planned against 13) |
| Second annotator | psychology student, confirmed |
| Annotator budget | ~1,000 rows × 2 people ≈ 33 hrs each over 8 weeks (~4 hrs/wk) |
| Critical path | **Google Form distribution** — calendar-bound, cannot be compressed |
| Scarce resource | the annotator, not GPU time |
| GPU | free Kaggle P100; individual runs are ~15 min |

---

## 3. What is already measured

All on `data/splits_stage2` (1,278 train / 158 val / 161 test, distorted-only),
multilabel macro-F1 over 10 labels, per-class thresholds tuned on val.
**Val only — test was not read.** Scripts: `scratchpad/matched.py`, `why.py`.

### 3.1 Input representation beats architecture

| model | input | val macro-F1 |
|---|---|---|
| TF-IDF + LogReg | gold span, cropped | **0.402** |
| mental-roberta-base (3 seeds) | full document | 0.357 ± 0.019 |
| TF-IDF + LogReg | full document | 0.322 |
| TF-IDF + LogReg | document **minus** the span | 0.281 |

- Same input, the model is worth **+0.035**.
- Same model, the span input is worth **+0.080**.
- Span alone beats span + context: adding the surrounding 83% of the text **costs** 0.080.
- Context alone still scores 0.281, so it is weakly informative but a net drag.

**This is an oracle result.** It uses the gold `Distorted part`, which is
unavailable at inference. 0.402 is a ceiling, never a system score. Same shape as
ERD's ground-truth-span result (15.28 → 27.08).

### 3.2 Truncation is ruled out

Stage 2 ran `max_length=512`, head truncation, val truncation rate **4.4%**.
The transformer does see the span. The problem is dilution, not truncation.

### 3.3 Spans are sentences, not phrases

On the 1,355 spans that align exactly:

| measure | value |
|---|---|
| median IoU between gold span and its sentence-snapped version | **1.000** |
| spans with IoU ≥ 0.8 after snapping | 94% |
| spans that are exactly one sentence | 71% |
| median sentences per span / per document | 1 / 8 |
| median span as fraction of document | 17% |
| spans starting in the last third of the document | 48% |

**Consequence: build a sentence selector, not a per-token BIO tagger.** Eight
binary decisions per document instead of ~210 three-way ones, at a median IoU
cost of zero.

### 3.4 Clustering does not recover distortion type

K-Means (k=10) on TF-IDF, distorted train rows:

| clustered on | ARI | NMI | purity | majority-class floor |
|---|---|---|---|---|
| spans | 0.009 | 0.030 | 0.203 | 0.149 |
| full documents | 0.009 | 0.028 | 0.201 | 0.149 |

Clusters track **topic** (relationships, parents, family), not reasoning pattern.
Ordering the annotation queue by cluster would inflate Cohen's κ through anchoring
bias. **Randomise and interleave the queue instead.**

### 3.5 The task is lexically cued

Top learned features per class on spans map almost word-for-word onto the CBT
definitions: `should / supposed to` → should statements; `guilty / my fault` →
personalization; `they / think / everyone` → mind reading; `always / never /
everything` → overgeneralization; `stupid / failure / worthless` → labeling.
This is why a bag of words is competitive, and why the transformer's headroom over
it is structurally small (+0.035).

### 3.6 Warranted-concern pool (rough upper bound)

Of the 933 No Distortion rows, **340 (36%)** contain both negative-affect language
and evidence markers; ~272 fall in train. A keyword proxy, not a count — but the
pool is not empty.

---

## 4. The gate

**Run first, before anything else is built:** `mental-roberta-base` on gold spans
cropped, `data/splits_stage2`, 3 seeds, same config as the existing Stage 2 run
(max_length 512, 12 epochs, BCE, tuned thresholds). ~15 min GPU.

| gold-span val macro-F1 | reading | verdict |
|---|---|---|
| **≥ 0.40** | gain well beyond seed noise (±0.019) | build the span head |
| 0.37 – 0.40 | marginal; predicted spans recover perhaps half the oracle gain | keep for interpretability, drop the F1 claim |
| ≈ 0.357 | the perfect answer changes nothing, so a predicted one will not either | cut the span head |
| < 0.357 | the transformer needs context TF-IDF could not use | cut it — and report it, it is a finding |

**This gates the span head only.** The warrant head is independent. A span no-go
leaves a two-head model, not no model. Even on a cut, the selector still produces
the highlighting the product needs — an interpretability contribution, not an
accuracy one.

Second variant while the GPU is warm: full document with the span **marked in
place** rather than cropped. Distinguishes "context is harmful" from "context is
merely unmarked", which decides whether the span head crops or annotates. Note it
is equally an oracle setup — tags must be present at training *and* inference, so
the deployable form uses predicted spans.

---

## 5. Four tracks

| track | bound by | weeks |
|---|---|---|
| **A — Collection** | calendar | 1–8 |
| **B — Span** | GPU + code | 1–5 |
| **C — Annotation** | the psychology student | 2–11 |
| **D — Joint model + evaluation** | B and C landing | 6–13 |

**Track B depends on nobody.** Whatever happens with the form, the annotator, or
the synthetic API keys, B yields a complete defensible contribution by week 4.

---

## 6. Schedule

### Weeks 1–2 — start everything with a long lead time

- **A.** Apply the §5 changes in `docs/month2-decisions.md`. Test submission from a
  personal Google account — **verify the form is not FAST-NU restricted**, it fails
  silently if it is. **Distribute.** Highest-priority action in the plan.
- **B.** Repair the 215 misaligned spans (`rapidfuzz.partial_ratio` over sliding
  windows; log anything under a similarity floor rather than silently keeping it).
  Build the sentence-selection dataset. Run the trivial baselines: random (~0.125),
  first sentence, last sentence, longest sentence, most-trigger-words.
- **B.** Run the §4 gate.
- **C.** Write the codebook. Distortion definitions plus the warranted-concern rule,
  applied in order: **evidence** (references something observed?), **scope** (bounded
  to this event, or generalised to always / never / identity?), **revisability**
  (could new evidence change it?). All three pass → warranted. Any fail → distorted.

### Weeks 2–4 — lock the guaranteed contribution

- **B.** Train the sentence selector. Score with sentence F1 and IoU — **never exact
  match**. It must beat the keyword baseline to earn its place.
- **B.** The headline: **predicted-span pipeline end to end.** Three rungs — no span
  (0.357), predicted span (unknown), gold-span ceiling.
- **C.** 100-item pilot, both annotators, independent. Report Cohen's κ separately
  for distortion type **and** for warranted concern. The warrant κ is itself a
  finding: if two people cannot agree what "justified" means, say so. Revise the
  codebook on the disagreements.

**By week 4 the span chapter is written and needs no new annotation.**

### Weeks 4–7 — targeted re-annotation

Full 2,530-row re-annotation does not fit and is not necessary.

| pass | rows | why |
|---|---|---|
| clinical **test** set, complete | 253 | a clean answer key before anything else |
| model-disagreement triage | ~400 | rows where the model confidently disputes the gold label |
| random control | ~200 | proves the triage was not cherry-picked |

Randomise and interleave the queue (§3.4). Warranted concern is labelled in the
same pass — that is the only reason it is affordable.

**Reporting rule.** Re-annotation changes the exam, so a score on new labels is not
comparable to 0.357. Report both models on the **new** test set:

| trained on | tested on | isolates |
|---|---|---|
| original labels | re-annotated test | baseline |
| re-annotated labels | re-annotated test | **training-label quality — the result** |

- **D.** Build the joint model in parallel (§7).

### Weeks 6–9 — the domain data

- **A.** Close the form. Both annotators label every response — distortion type and
  warrant. This is the gold entrepreneurial test set. **Natural distribution, never
  balanced.**
- **Synthetic.** Run the 20-row / ~₨56 calibration first to measure the real revision
  rate, then scale only if acceptable. Synthetic rows enter **train only**.
- **Decision gate, week 7:** fewer than ~20 responses → switch to the §9 contingency.

### Weeks 8–11 — ablations and challenge set

| run | question it answers |
|---|---|
| binary head alone | the existing Stage 1, 0.794 — the safety net |
| distortion head alone | the existing Stage 2 baseline, 0.357 |
| span head alone | can it find the region without knowing the type? |
| warrant head alone | is "justified" learnable from text at all? |
| **joint, all four, span-weighted pooling** | do the heads help each other, or just share a checkpoint? |

Run the four single-head rows first. Without them the joint number has nothing to be
compared against, and "we built a joint model and got 0.44" is not a result.

**Challenge set:** 80–100 founder statements that are negative but justified. Half
written against the codebook rule, half drawn from real collected data and verified
by both annotators. **Test only, never trained on.**

### Weeks 10–13 — final evaluation

Touch the test sets **once**. Report in this order:

1. **Gate evaluation** (§6.1) — lead with this
2. Binary detection F1 (comparable to the paper's 0.79)
3. Per-class multi-label F1, **with per-class support counts printed alongside**
4. Span selection — sentence F1 and IoU
5. The three deltas: span vs no-span, joint vs single-task, clinical vs entrepreneurial

Three test sets reported **separately** — clinical, entrepreneurial, challenge. Never
pooled. Spoken interview transcripts, if collected, are a fourth, also separate.

### 6.1 Gate evaluation — both directions

The gate can fail two ways, and they are measured on different sets. Reporting only
the first would be the easiest thing in this thesis to attack.

| failure | what the user experiences | measured on | metric |
|---|---|---|---|
| gate too **quiet** — does not fire when it should | founder gets flagged for being realistic | **challenge set** | false-positive rate / specificity |
| gate too **eager** — fires when it should not | a real distortion is silently swallowed | **clinical + entrepreneurial test sets** | recall on distorted rows, gate **on** vs **off** |

The challenge set is purpose-built so that every one of its 80–100 items is
warranted concern. The correct answer for every row is "do not flag", so exactly one
number comes out: *of N justified statements, how many did the system wrongly flag?*
This cannot be measured on the normal test set — warranted-concern rows are rare and
unlabelled there, so the rate would rest on a handful of examples and mean nothing.

The second row is a cheap run: evaluate the same model twice, once with suppression
active and once disabled. **Report the recall lost against the specificity gained.**
If enabling the gate costs 20 points of recall to buy 8 points of specificity, that
is a trade to be argued explicitly, not assumed. Include the trade-off curve across
gate thresholds so the choice of operating point is visible rather than tuned
silently.

### Weeks 13–16 — writing, and stretch only if it exists

Interview transcription. Domain-adaptive pretraining (§8 — expect a null). Structural
marker feature fusion as a single ablation.

---

## 7. Architecture

Shared encoder, **four** heads, one forward pass.

### 7.1 The heads

| head | reads | outputs | trained on | loss |
|---|---|---|---|---|
| **binary** | plain pooled vector (whole document) | 1 sigmoid | all rows (2,530) | BCE |
| **warrant** | plain pooled vector (whole document) | 1 sigmoid | all rows (2,530) | BCE |
| **span** (sentence selector) | token vectors, mean-pooled within each sentence | 1 logit per sentence | distorted only (1,597) | BCE, `pos_weight ≈ 4` |
| **distortion** | **span-weighted** pooled vector | 10 sigmoids | distorted only (1,597) | BCE, class-weighted |

**Why binary is a head and not a leftover.** The obvious alternative is to call it
"no distortion" whenever all ten sigmoids fall below threshold. That fails on
compounding false positives: ten independent sigmoids at ~5% FPR each give
`1 - 0.95^10 ≈ 40%` chance that at least one fires spuriously on clean text. For a
tool whose safety claim is "we do not flag people who are fine", that is fatal.
Binary detection is also this project's *strongest* task — 0.794 F1, matching the
paper's 0.79 — so it should not be re-derived as a residual.

**Why binary and warrant read the plain pooled vector, not the span-weighted one.**
Deciding *nothing is wrong anywhere* requires seeing the whole document; focusing on
a selected sentence would beg the question. The span-weighted vector is for the ten
type heads, which have already been told a distortion is present and only need to
say which. This also matches the masking structure — the two whole-document heads
have labels on every row, the two focused heads only on distorted rows.

This layout is the **cascade you already validated, absorbed into one model**:
Stage 1 becomes the binary head, Stage 2 becomes the ten sigmoids; span and warrant
are the new parts.

### 7.2 Span-weighted pooling

What makes this one model rather than four sharing a checkpoint: the selector's
per-sentence probabilities weight the token vectors feeding the distortion heads, so
the model learns where to look and what to call it in a single differentiable pass.
This is the rationale-extraction family (Lei et al. 2016; ERASER, DeYoung et al.
2020) — **claim the application to cognitive distortions, cite the mechanism.**

### 7.3 Inference order

1. **Encode once.** The whole reflection through the encoder → one vector per token.
2. **Span head** scores each sentence: is this part of the distorted region?
3. **Two pooled vectors** are formed — the plain one (mean over all tokens / `[CLS]`)
   and the span-weighted one (tokens weighted by their sentence's probability).
4. **Binary and warrant heads** read the plain vector. **Ten distortion heads** read
   the span-weighted vector.
5. **Gate logic** (below) decides what the user actually sees.

| binary | warrant | output to the user |
|---|---|---|
| low | low | nothing flagged — reads as neutral |
| low | **high** | **warranted concern** — "this reads as a grounded worry" |
| **high** | low | distortion flagged: type from the ten sigmoids, location from the span head |
| **high** | **high** | **conflict — suppress by default** (see below) |

**The conflict row is a policy choice, not an edge case.** A statement can be partly
distorted and largely justified. Suppressing by default is the safer setting for a
wellbeing tool and is consistent with leading on specificity — but state it
explicitly in the thesis and report how often it fires, rather than letting the code
decide silently.

The gate threshold is tuned **separately** from the ten distortion thresholds, on
val, against a specificity-weighted objective rather than raw F1.

### 7.4 Dry run

**(a) Warranted concern — the gate fires.**
*"Twelve investors passed, so I don't expect this round to close."*

| head | output |
|---|---|
| binary | 0.44 — below threshold |
| span | s1 0.62 (only one sentence) |
| distortion | fortune_telling **0.71**, magnification 0.34, rest low |
| warrant | **0.83** |

Result: **warranted concern.** fortune_telling is suppressed. Note the distortion
head was not being unreasonable — that sentence *is* a prediction about the future.
The gate is what stops the tool telling a founder her arithmetic is a distortion.

**(b) Genuine distortion — the full pipeline runs.**
*"I pitched at demo day last week. Two of the three investors asked follow-up
questions and one asked for my deck. The third checked his phone the whole time.
I've never been able to hold a room and I never will."*

| head | output |
|---|---|
| binary | **0.91** |
| span | s1 0.05, s2 0.08, s3 0.31, **s4 0.94** |
| distortion | overgeneralization **0.78**, fortune_telling **0.66**, mental_filter 0.41 |
| warrant | 0.12 — fails scope ("never") and revisability ("never will") |

Result: **flagged.** Sentence 4 highlighted, labelled overgeneralization +
fortune-telling. Sentences 1–3 report evidence and are correctly left alone — which
is exactly the +0.080 dilution effect from §3.1 doing its job.

**(c) Neutral — nothing fires.**
*"I met twelve investors this month and I'm writing up my notes before the next
round."*

| head | output |
|---|---|
| binary | 0.08 |
| warrant | 0.11 |

Result: **nothing shown.** The span and distortion heads still produce numbers, but
the binary head gates them out before they reach the user. This is the case that the
"all ten sigmoids below threshold" design would have failed ~40% of the time.

### 7.5 Two implementation details that will bite

**Encode once, pool per sentence.** Do not feed 8 separate sentence inputs — each
sentence must keep the whole document as attention context.

**The heads see different rows.** Span and distortion labels exist only on distorted
rows; binary and warrant labels exist on all 2,530, and warrant's positives live
mostly in the No Distortion pile (§3.6), which currently conflates "fine" with
"worried for good reason". Each head's loss must be computed only where its label
exists. `MaskedBCEWithLogitsLoss` in `src/losses.py` already does this.

### Decoding the selector

Spans are contiguous by construction and 71% are a single sentence. Score every
contiguous run of 1–3 sentences and take the best, rather than thresholding each
sentence independently — independent decisions produce fragmented output that cannot
be right. Bias toward the observed median length (19% of the document).

---

## 8. Cut, and why

| cut | reason |
|---|---|
| Full 2,530-row re-annotation | ~100 annotator hours; triage + random control supports the same claim |
| Per-token BIO tagging | 71% of spans are one sentence, median snap IoU 1.000 (§3.3) |
| Clustering for annotation ordering | ARI 0.009 (§3.4); would inflate κ via anchoring |
| Domain-adaptive pretraining | `EXPERIMENTS.md` finding 2 already records that a domain-adapted encoder bought nothing beyond seed noise, and it is only scoreable after the entrepreneurial test set exists |
| Structural feature fusion | markers-only macro-F1 0.171, and the strongest markers fire on 1–2% of spans; one ablation at most |
| Urdu prototype, reframing module | out of scope |

---

## 9. Risk register

| risk | tripwire | fallback |
|---|---|---|
| Form yields too few responses | week 7, under ~20 | synthetic carries the domain claim; responses become a case study, not the headline test set |
| Warranted-concern positives too scarce | under ~100 after annotation | implement warrant as a **rule gate** on the evidence / scope / revisability markers, evaluated on the challenge set. Still a contribution, different claim |
| Annotator becomes unavailable | any missed pilot deadline | single-annotator relabelling with supervisor adjudication on a slice; weaker, survivable, must be disclosed |
| Predicted spans fall far short of the ceiling | week 4 | report the gap — it is a result about annotation granularity |
| Span gate fails (§4) | week 2 | two-head model; selector kept for highlighting only |
| Warrant gate costs more recall than the specificity is worth (§6.1) | recall drop exceeds specificity gain at every threshold | ship the gate off by default, report the curve, and frame warrant as a surfaced confidence signal rather than a suppressor |
| Kaggle GPU quota | ongoing | span runs are ~15 min; front-load them |

---

## 10. Standing conventions

Inherited from `CLAUDE.md`, restated because this plan depends on them:

1. Splits are immutable; only `src/make_splits*.py` may create them.
2. No leakage — every vectorizer, scaler and threshold is fit on train only.
3. Test sets are read once, at the end, by `src/evaluate.py`.
4. Three seeds (42, 1337, 2024), mean ± std. Anything inside ±0.02 is noise.
5. Never report accuracy as a headline.
6. Synthetic rows enter train only. The test set stays real and at its natural class
   distribution.
7. Every headline number carries its splits directory and a leakage check — three of
   the four "surprising" results this project produced turned out to be
   infrastructure defects.

---

## 11. Open questions

- Does the sentence selector beat the most-trigger-words baseline? If not, the head
  is not earning its place.
- What is Cohen's κ on warranted concern? Unknown, and it may be low enough to change
  the framing.
- Does re-annotation move macro-F1 at all, or is the ceiling elsewhere?
- Do the span gain (+0.080) and the architecture gain (+0.035) actually add, or do
  they overlap?
- What does the gate cost? There is no prior number for recall lost to
  warrant-suppression, because no prior work in this literature models warranted
  concern at all. Whatever the curve shows is a new result either way.
