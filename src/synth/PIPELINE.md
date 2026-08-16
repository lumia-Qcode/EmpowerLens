# Synthetic Data Generation — Methods & Provenance

Reference for the EmpowerLens Month-2 (English track) synthetic-data pipeline in
`src/synth/`. Summarises the research it draws on, the caveats it is designed to
address, and exactly what is borrowed vs. original.

> **Thesis-facing document.** For mechanics — modules, knobs, output schema, run
> commands, operational gotchas — see [`README.md`](README.md). For decisions and
> open questions, see [`docs/month2-decisions.md`](../../docs/month2-decisions.md).

> **Scope.** Synthetic rows are **training data only**. The evaluation (test) set
> is separate: real, human-written reflections labelled via
> [`data/synthetic_gen/labeling/labeling_guide.md`](../../data/synthetic_gen/labeling/labeling_guide.md).
> Synthetic data is never used for testing.

---

## 1. Research papers and their relevance

| Paper | Relevance to us | What we borrow | Where it lives |
|---|---|---|---|
| **KoACD** — Korean Adolescent Cognitive Distortion dataset via role-switching multi-LLM negotiation (Kim & Kim, 2025) | Closest domain match: multi-LLM consensus for **cognitive-distortion** labelling, with human-expert validation of a subset | (a) Multi-agent **consensus verification** — agreement across LLMs yields a trustworthy label. (b) The **Appendix G generation constraints**: hard length cap repeated several times, mandated first person, explicit situation grounding with a cause-and-effect chain, per-call banned-opener list, and the distortion vocabulary forbidden in the output. (c) The **Appendix H Analyzer structure**: an enumerated escape hatch with explicit trigger conditions rather than a soft "don't invent one" note, whole-context judgement, and mandatory sentence-level evidence quoting | Critic panel (Stage 3) + Generator prompt (Stage 1) |
| **AttrPrompt** — LLM as Attributed Training Data Generator (Yu et al., 2023) | Directly addresses diversity collapse and reports gains on **multi-label** tasks (we are multi-label) | **Attribute-conditioned generation**: condition on trigger/stage/sector/register/marker instead of a bare class label | Planner (Stage 0) + Generator (Stage 1) |
| **Self-Instruct** (Wang et al., 2023, ACL) | The canonical seed→generate→filter bootstrapping loop | **Seed few-shot bootstrapping** + heuristic filtering of invalid/near-duplicate rows | Generator (Stage 1) + Filters (Stage 2) |
| **Diagnosis of Thought (DoT)** — Cognitive Distortion Detection through DoT Prompting (Chen, Lu & Wang, 2023, EMNLP Findings) | Best-in-class **reasoning method** for distortion detection; produces expert-approved rationales | The critic's three-stage reasoning: **fact/thought split → contrastive reasoning → schema analysis** | Critic reasoning (Stage 4) |
| **Few-shot LLM Synthetic Data with Distribution Matching** (2025) | Keeps synthetic data's distribution close to real data rather than the model's clean priors | Realism control — match the synthetic distribution to the real corpus (applied once the real set exists) | Planner defaults + future refinement |
| **Shreevastava & Foltz (2021)** — Detecting Cognitive Distortions | Our detection **benchmark** and taxonomy source (10 distortions + No Distortion); its ~33.7% inter-annotator agreement sets the label-noise ceiling | Label taxonomy; the motivation for **soft labels** | Definitions + labels throughout |
| **A Survey of Cognitive Distortion Detection in NLP** (2025) | Positions the work and documents synthetic-data trade-offs | Related-work framing | Thesis write-up |

**One-line framing for the thesis:** *we adapt AttrPrompt-style attributed
generation (Yu et al., 2023) with a KoACD-style multi-agent consensus critic
(Kim & Kim, 2025) whose reasoning follows Diagnosis of Thought (Chen et al.,
2023), seeded and bootstrapped in the manner of Self-Instruct (Wang et al.,
2023).*

---

## 2. Caveats and how the pipeline addresses them

| # | Caveat (risk of naive LLM data generation) | How this pipeline addresses it |
|---|---|---|
| 1 | **Label leakage / circularity** — classifier learns the generator's lexical tells | Critics are **blind** to the intended label and from a **different provider**; acceptance requires them to *rediscover* the label unprompted; the distortion vocabulary is **banned from the output** and enforced as a post-hoc filter; **Tier-2 marker bans** applied selectively (structural classes lose their stock markers; lexical classes keep theirs but rotate the surface form) |
| 2 | **Distribution gap** — synthetic text is too clean/templated vs. real | Seed anchoring (Self-Instruct); situation grounding + cause-and-effect chain required in the prompt (KoACD); distribution matching once the real set exists; **real-only test set** keeps evaluation honest. ⚠️ *Partial:* the current seeds are authored, not collected — see §5 |
| 3 | **Clean-vs-noisy label mismatch** — real labels are ambiguous (~33.7% IAA) | **Soft labels** from inter-critic agreement mirror human disagreement; trainable with soft-target BCE (ablatable vs. hard labels) |
| 4 | **Diversity collapse** — repetitive, templated output | **Attribute conditioning** (AttrPrompt) over a scenario × distortion grid + **temperature-decorrelated** critics + rotating seed exemplars + a **banned-opener list refreshed per call** (KoACD) |
| 5 | **Critic rubber-stamping** — same-family judge shares blind spots | **Cross-provider**: Claude authors, **Gemini** judges — the author never grades itself. ⚠️ *Partial:* all three critics are the same Gemini model at different temperatures, so within the panel this is self-consistency, not independent annotation — see §5 |
| 6 | **Definitional drift** — generator/critic disagree on what a label means | A single **shared definitions block** is injected into *both* the generator and critic prompts (and matches the human labelling guide) |
| 7 | **Cultural inauthenticity** — generic Western-startup voice | Pakistani-context seeds + a `cultural_marker` attribute (light on the English track; expanded on the Urdu track) |
| 8 | **Dedup / contamination** — near-duplicate rows; leakage into test | Token-Jaccard dedup vs. seeds + accepted rows; synthetic is **structurally separated** from the test set |
| 9 | **No human verification** | The same definitions block guides human and machine labels. ⚠️ *Planned, not built:* a human **audit gate** on a sample after the calibration run, plus a **keyword-only baseline** over the synthetic set — if a hand-written keyword list scores well above chance, leakage remains and the data should be regenerated |
| 10 | **Dishonest reporting** | Headline metrics come from the **real** test set only; the synthetic origin of training data is stated explicitly |

---

## 3. Pipeline stages

Six stages — three are deterministic code (cheap, controllable), three are LLMs.
Cross-provider by design. **Mechanics, knobs, output schema and run commands live
in [`README.md`](README.md)**; this section exists so the stage numbers referenced
in §1 and §2 resolve.

| Stage | Component | Engine | Why |
|---|---|---|---|
| 0 | Planner | code | Deterministic class balance (rare classes oversampled), ~30% No Distortion, ~26% secondary rate, scenario × distortion grid |
| 1 | Generator | `anthropic:claude-sonnet-5` | Attribute- and seed-conditioned author; two granularities (snippet ≈ 30–50 w, reflection ≈ 150 w) |
| 2 | Filters | code | Cheap rejects before spending critic tokens — length, banned clinical vocabulary, English-dominance, Jaccard dedup |
| 3 | Critic panel | `gemini:gemini-2.5-flash` ×3 | Blind, cross-provider judges; DoT reasoning; temperature-decorrelated (0.3/0.6/0.9); each emits the span it judged from |
| 4 | Aggregate + decide | code | Soft labels from agreement; span agreement; anti-leakage acceptance gate |
| 5 | Reviser | `anthropic:claude-sonnet-5` | One feedback-guided retry carrying the critics' actual objection |

*(Role rotation, à la KoACD, is intentionally **not** used: our critics are
symmetric blind peers, so there are no asymmetric roles to rotate — temperature
variation provides the decorrelation instead.)*

---

## 4. What is borrowed vs. what is ours

**Borrowed (with citation):** attributed generation (AttrPrompt), seed
bootstrapping + heuristic filtering (Self-Instruct), multi-agent consensus
labelling **and the generation constraints** (KoACD), DoT critic reasoning
(Chen et al.), distribution matching, and the distortion taxonomy
(Shreevastava & Foltz).

**Adapted, not copied:**
- **Tier-2 lexical control is applied *selectively*.** KoACD bans stock markers to prevent the classifier learning label words. Applied naively that deletes classes whose definition *is* lexical — "should statements" without "should" is not a should statement. So structural classes (Overgeneralization, All-or-nothing, Magnification, Mind Reading, Fortune-telling, Mental filter) lose their markers outright, while lexical classes (Should statements, Labeling, Personalization, Emotional Reasoning) keep theirs and instead **rotate the surface form** per call, so the model learns the construction rather than one token.
- **KoACD's single "Unknown" verdict is split in two.** KoACD collapses "cannot judge" and "nothing wrong here" into one Unknown bin. We separate them: `Unknown` (unjudgeable — only emotion, a plain situation report, needs absent context) and `No Distortion` (judged, and the reasoning is sound), each with its own enumerated trigger conditions. Collapsing them would file unjudgeable text as calibrated reasoning, inflating the No-Distortion class with rows containing no reasoning at all — and No Distortion is precisely the class novelty direction D1 depends on. The split also keeps the abandonment statistic honest: a majority-Unknown rejection is a *generator* failure and is excluded from the taxonomy confusion tally, while a genuine label disagreement is kept.
- **The calibrated-pessimism guardrail is a formal exit condition, not a note.** "A negative prediction that is well-calibrated to a rejection-heavy domain is realism, not Fortune-telling" sits in the critic's enumerated No-Distortion list, in the same structural position KoACD gives its Unknown triggers.
- **The critic receives full context and *emits* the span**, rather than receiving only the distorted part as in KoACD. DoT's first stage separates facts from thoughts and has nothing to work on given a bare span; and the grounding argument requires that distortions sit inside a situation. Emitting the span instead yields the `Distorted part` column plus a **span-agreement** signal — critics agreeing on the label while pointing at different sentences marks a diffuse row.

**Original to EmpowerLens:**
- **Cross-provider author/judge separation** (Claude writes, Gemini grades) as an explicit anti-leakage mechanism, rather than same-family self-critique.
- **Soft labels derived from critic agreement, to be calibrated against human IAA** — turning disagreement into a training signal.
- **Deterministic code planner** with class-balance + rare-class oversampling + real-distribution-matched No-Distortion and secondary rates, over a **scenario × distortion grid** giving complete, auditable coverage (154/154 cells at n=300).
- **Two generation granularities** — short single-distortion snippets for clean classifier signal, longer multi-distortion reflections for the multi-label head and the dashboard.
- **Abandonment as evidence, not waste.** Every rejected spec is logged with the critics' verdicts and what they read *instead*. A spec that reached the panel and failed twice is a case where a clinical label could not be made to fit entrepreneurial reasoning — so the per-label abandonment rate, and the `asked for X → panel read Y` confusion structure, become a **quantitative argument for an entrepreneurial-specific taxonomy**. This is the analogue of KoACD's large "Unknown" pile, which in that paper became evidence of taxonomy inadequacy rather than a discard bin. The **No-Distortion abandonment rate** is broken out separately: it measures how hard the calibrated-pessimism distinction actually is, which is novelty direction D1 stated as a number.
- **Domain and application:** cognitive-distortion detection in **entrepreneurial self-reflection**, Pakistani incubator context (English track now; Urdu/code-mix track later).
- **Clean two-track separation:** synthetic = training, real = test — never mixed.

---

## 5. Honest limitations — state these before an examiner finds them

Referenced by the ⚠️ markers in §2. These are claims the pipeline does **not**
currently support, listed so the writeup does not overstate them.

1. **The critic panel is one model, three temperatures.** Author-vs-judge is
   genuinely cross-provider, but the three critics are all `gemini-2.5-flash`.
   Correlated errors are not voted out, and `y_soft` is correspondingly
   overconfident. Describe this as **self-consistency across temperature
   samples**, not inter-annotator agreement. Fixable by letting
   `--critic-model` take a list.
2. **The seed set is authored, not collected.** All 13 rows carry
   `source: authored_from_*`. They anchor *format* (length, first person, no
   meta-commentary), which transfers regardless of author — but not *register*,
   which is why they exist. Plan: after the Google Form collects responses,
   humans read them and hand-write 25–30 grounded seeds. Participant text itself
   is never sent to an API.
3. **Dedup is lexical only.** Threshold 0.8 on a word *set*. Measured: a
   near-verbatim paraphrase scores 0.762 and **passes**; two texts sharing an
   opener but diverging score 0.159. It catches literal repeats, not homogeneity.
4. **A dropped critic verdict skews the vote.** If a critic's JSON fails to
   parse it is silently omitted, but `aggregate()` still divides by `n_critics`,
   making the majority harder to reach and inflating the revision rate.
5. **No human verification has happened yet**, and the keyword-only leakage
   baseline is not written.
6. **The ~33.7% inter-annotator agreement figure** quoted for Shreevastava &
   Foltz is unverified against the source, and it matters which statistic it is:
   33.7% *raw agreement* on an 11-class task is poor, whereas 33.7% *kappa* is
   "fair" and respectable for a subjective clinical task. Confirm before citing.

---

## References

- Kim & Kim (2025). *KoACD: … via Role-Switching Multi-LLM Negotiation.* arXiv:2505.00367.
- Yu et al. (2023). *Large Language Model as Attributed Training Data Generator: A Tale of Diversity and Bias (AttrPrompt).* arXiv:2306.15895.
- Wang et al. (2023). *Self-Instruct: Aligning Language Models with Self-Generated Instructions.* ACL 2023.
- Chen, Lu & Wang (2023). *Cognitive Distortion Detection through Diagnosis of Thought Prompting.* Findings of EMNLP 2023.
- *Few-shot LLM Synthetic Data with Distribution Matching* (2025). arXiv:2502.08661.
- Shreevastava & Foltz (2021). *Detecting Cognitive Distortions from Patient-Therapist Interactions.*
- *A Survey of Cognitive Distortion Detection and Classification in NLP* (2025). arXiv:2508.09878.
