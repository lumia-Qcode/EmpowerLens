# `src/synth` — synthetic data generation pipeline

> **How it works and how to run it.** For the research provenance — which papers
> contribute what, borrowed vs. original, and the honest limitations — see
> [`PIPELINE.md`](PIPELINE.md). For decisions and open questions, see
> [`docs/month2-decisions.md`](../../docs/month2-decisions.md).

Generates labelled entrepreneurial self-reflections to augment the frozen
`Annotated_data.csv` benchmark, which is clinical patient–therapist text and
therefore the wrong domain for EmpowerLens.

**Training data only.** Synthetic rows may join `train`; they never enter
`data/splits/test.csv`, which stays 100% real and is read only by
`src/evaluate.py`.

---

## Why it exists

Month 1 replicated Shreevastava & Foltz on 2,530 rows of therapy transcript.
The FYP target is short written reflections from Pakistani women entrepreneurs
— a domain shift on three axes at once (clinical → entrepreneurial vocabulary,
transcript → written reflection, Western → South Asian framing), plus a class
imbalance where the rarest distortion has 100 rows and the commonest has 239.

Two design problems follow, and they shape the whole architecture:

| Problem | Solution |
|---|---|
| Labels must not be circular — "whatever we asked for" is not a label | **Blind critic panel** from a different provider |
| Text must be diverse — 21 requests for Mind Reading gives 21 paraphrases | **Attribute planner** over a scenario × distortion grid |

---

## Flow

```
planner.py      →  N specs (pure Python, no LLM, free)
      ↓
prompts.py      →  build_generator_system(granularity) + build_generator_user(spec)
      ↓
llm.py          →  Claude Sonnet 5 writes a reflection
      ↓
generate.py     →  cheap filters: length / banned terms / English / Jaccard dedup
      ↓
prompts.py      →  build_critic_user(text)   ← BLIND, label never shown
      ↓
llm.py          →  3 × Gemini 2.5 Flash diagnose independently (temps 0.3/0.6/0.9)
      ↓
generate.py     →  aggregate() → votes, y_soft, span, span_agreement
      ↓
generate.py     →  decide() → accept / revise-once / reject
      ↓
                   JSONL row
```

Cheap filters run **before** the paid critic calls, so a malformed generation
never burns three critic requests.

### What happens after generation

The generated text first passes a short sequence of free checks in order — word
count against the spec's granularity band, a banned-terms scan that drops any row
naming its own distortion, then English-dominance and Jaccard dedup against
everything accepted so far — and a failure at this stage costs nothing further,
because it happens before any critic call. Surviving text goes to three blind
Gemini critics at temperatures 0.3/0.6/0.9, each independently running
Diagnosis-of-Thought and returning JSON with the distortions present, a primary
and secondary, at least two verbatim sentences of supporting evidence, the
cause-and-effect chain in prose, the verbatim span it judged from, and a
confidence — or `Unknown` if the text cannot be judged at all. `aggregate()`
turns those verdicts into vote counts, the `y_soft` vector (each label's vote
fraction), a representative span (the longest returned), and a span-agreement
score. `decide()` then applies a 2-of-3 majority: if the planner asked for
No Distortion it needs a No-Distortion majority *and* no distortion reaching
majority itself, otherwise it simply needs the intended primary to clear the bar
— and the secondary written to the row is whatever the critics ranked second, not
what the planner requested. If the vote fails, the text goes back to Sonnet
**once** with the critics' actual objection attached, and the whole
critique-and-decide cycle repeats; if it fails again the spec is abandoned.
Accepted rows are written to the JSONL with labels, span, soft vector, raw votes
and all the attribute fields, and the text is added to the dedup pool while its
opening four words join the banned-opener list for subsequent prompts.

---

## Modules

| Module | Role |
|---|---|
| `attributes.py` | Distortion definitions, attribute axes, generation constraints, banned lists |
| `planner.py` | Deterministic spec generation — class balance + scenario grid. **No API calls** |
| `prompts.py` | Generator / critic / reviser prompt builders |
| `llm.py` | Provider abstraction (Anthropic + Gemini), cost estimation |
| `dedup.py` | Token-set Jaccard near-duplicate filter, dependency-free |
| `generate.py` | Orchestrator, accept/revise/reject loop, JSONL writer |

### `attributes.py`

Single source of truth for the taxonomy — `DISTORTIONS` is injected verbatim
into **both** the generator and critic system prompts so their notion of each
label matches. Definitional drift between the two would show up as a high
rejection rate with no visible cause.

Attribute axes sampled by the planner:

| Axis | Values |
|---|---|
| `TRIGGERS` | 14 scenarios (investor rejection, family disapproval, only-woman-in-the-room, …) |
| `REGISTERS` | `english_formal`, `english_casual` (Urdu track adds code-mix later) |
| `STAGES` | `idea`, `prototype`, `early_launch`, `growing` |
| `SECTORS` | food, fashion, handicrafts, beauty, tech, services, education |
| `CULTURAL_MARKERS` | `None` ×3, `gender_expectation`, `family_pressure` (weighted — 60% none) |

14 × 2 × 4 × 7 × 5 = **3,920 situational combinations**.

Distribution knobs, tied to the measured real corpus:

```python
NO_DISTORTION_SHARE = 0.30   # real corpus is ~0.37; slightly less for coverage
SECONDARY_RATE      = 0.26   # real corpus: 416/1597 distorted rows
SNIPPET_SHARE       = 0.70   # rest are full reflections
```

### `planner.py`

Emits one spec per target row, deterministic given `--seed`. Same seed →
byte-identical specs, forever.

Class balance is guaranteed by `divmod`, not by asking a model nicely:

```python
def _round_robin_counts(total: int, k: int) -> list[int]:
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]
```

The **scenario × distortion grid** walks `TRIGGERS` in order with a per-class
offset, so every (distortion, trigger) cell fills evenly rather than being
sampled at random. At `--n 300` all **154 cells** (11 classes × 14 triggers) are
covered, with per-trigger counts between 19 and 23.

A spec:

```json
{"spec_id": "spec_00000", "primary": "No Distortion", "secondary": null,
 "granularity": "reflection", "trigger": "imposter_feelings",
 "register": "english_casual", "stage": "idea", "sector": "beauty",
 "cultural_marker": "gender_expectation"}
```

### `prompts.py`

Three roles. Generation constraints follow KoACD's Appendix G/H.

**Generator** — `build_generator_system(granularity)` + `build_generator_user(spec, …)`

- Length cap stated **three times**: top of system prompt, restated at its end,
  again in the user prompt
- First person mandated
- Situation grounding — *"State when/where it happened and what concretely
  occurred… No floating aphorisms"*
- Cause-and-effect chain — *"something happened → she thought X → therefore she
  concluded Y. The distortion must live in the link"*
- Distortion vocabulary banned outright (also enforced post-hoc as a filter)
- Banned openers, refreshed per call from `STOCK_OPENERS` + the last 12 the run
  actually produced

**Tier-2 lexical control, applied selectively.** Banning `"should"` would delete
the Should-statements class rather than the leakage, so:

| Kind | Classes | Treatment |
|---|---|---|
| Structural | Overgeneralization, All-or-nothing, Magnification, Mind Reading, Fortune-telling, Mental filter | Stock markers **banned** — build the pattern from reasoning |
| Lexical | Should statements, Labeling, Personalization, Emotional Reasoning | Marker kept, but a **rotating required form** each call (`"ought to"` / `"supposed to"` / `"a real founder would"` …) so the model learns the construction, not one token |

**Critic** — blind, full context, Diagnosis-of-Thought, **emits** the span.

Three defensive constraints from KoACD's Appendix H Analyzer prompt:

- **Judge the whole text, not one sentence.** A single dramatic line inside
  otherwise sound reasoning is not a distortion.
- **Two escape hatches, deliberately kept apart.** `Unknown` (enumerated
  conditions: only emotion with no reasoning, a plain situation report, needs
  absent context, the thought belongs to someone else, …) means *"I cannot judge
  this"*. `No Distortion` (its own enumerated conditions, including **a negative
  prediction that is well-calibrated to a rejection-heavy domain is realism, not
  Fortune-telling**) means *"I judged this and the reasoning is sound"*. Merging
  them would file unjudgeable text as calibrated reasoning and poison the very
  class the pipeline works hardest to protect.
- **Evidence is mandatory** — at least two verbatim sentences, or the whole text
  if shorter. `_critique()` enforces this in code rather than trusting the
  prompt: a verdict returning a confident label with fewer than two quoted
  sentences is **coerced to `Unknown`** and flagged `coerced_unknown`, because a
  label asserted without the evidence it was told to produce is exactly the
  speculation the rule exists to stop.

**Exclusion memory is deliberately not implemented.** KoACD's agents negotiate
across turns and can loop by re-proposing a rejected label. This pipeline has no
multi-turn loop — the critics are one-shot stateless calls with no memory of
their own prior verdict, and there is exactly one revision. There is nothing to
loop, so the mechanism would be dead code. It becomes relevant only if the
revision count is ever raised above one.

The critic never sees the intended label. It sees the whole text (not just the
distorted span — DoT stage 1 needs a situation to separate facts from thoughts)
and returns the shortest verbatim span carrying the primary. That gives the
`Distorted part` column plus `span_agreement`: critics can agree on the label
while pointing at different sentences, which marks a diffuse row the vote count
alone cannot show.

**Cache discipline:** anything varying per call (banned openers, seed exemplars,
the spec) lives in the **user** prompt. The system prompt varies only by
granularity → exactly two cacheable prefixes, both ~1,100 tokens, above Sonnet
5's 1,024-token cache minimum.

### `generate.py`

```python
majority = math.ceil(n_critics / 2)      # 2 of 3
```

**No Distortion needs a stricter double condition** — a majority must say No
Distortion *and* no distortion may itself reach a majority. A false No-Distortion
label is more damaging than a false distortion label, because distinguishing
distorted thinking from calibrated realistic pessimism is novelty direction D1.

**The secondary label is discovered, not dictated.** The planner requests one 26%
of the time, but the label written is whatever the critics ranked second.

---

## Output schema

One JSON object per line:

| Field | Notes |
|---|---|
| `id` | `syn_00042` |
| `text` | the reflection |
| `dominant_distortion` | raw label string → `LABEL_CANON` in `src/data.py` |
| `secondary_distortion` | or `null` |
| `distorted_part` | critic-emitted span; maps to the `Distorted part` column |
| `span_agreement` | mean pairwise Jaccard of the critics' spans, 0–1 |
| `evidence` | ≥2 verbatim sentences the matching critic quoted in support |
| `critic_reasoning` | that critic's cause-and-effect chain, in prose |
| `y_soft` | 10-dim vector, fraction of critics seeing each distortion |
| `critic_votes` | raw counts, for auditing |
| `no_distortion_votes` | the calibrated-pessimism margin — 3/3 is clean, 2/3 contested |
| `revised` | `true` if this row came from the retry path (see below) |
| `n_critics`, `granularity` | |
| `trigger`, `register`, `stage`, `sector`, `cultural_marker` | slicing variables |
| `intended_primary`, `intended_secondary` | what the planner asked for — the gap vs. actual is a quality signal |
| `source` | `"synthetic"` |

### The rejects log — `<out>.rejected.jsonl`

Written alongside the accepted rows. **Abandoned specs are not noise.** A spec
that reached the panel and failed is a case where we asked for distortion X in an
entrepreneurial setting and independent critics *twice* declined to see X — which
is evidence that the clinical taxonomy does not map cleanly onto founder
thinking, i.e. the entrepreneurial-taxonomy novelty argument. Discarding it
throws away the strongest figure in the chapter.

Each reject row carries the reason (`length`, `named_own_pattern`, `not_english`,
`duplicate`, `critics_unknown`, `critics_disagreed`, `generator_error`), the full
spec attributes, the text, and — for `critics_disagreed` only — `critics_leaned`
(what the panel read instead), both rounds' votes, `y_soft`, and the `decide()`
note.

**`critics_unknown` and `critics_disagreed` are counted separately, and only the
second is taxonomy evidence.** A majority-Unknown verdict means the generator
produced something unjudgeable — a bare emotional vent, a plain situation report
— which is a *generation* failure. A `critics_disagreed` verdict means the panel
read the text fine and saw a different pattern, which is the taxonomy signal.
Merging them would put a wrong number in the thesis, so Unknown rejections are
excluded from the `[taxonomy]` confusion tally entirely.

The run then prints three summaries:

- **`[abandonment]`** — abandoned/attempted per intended label, sorted by rate. A
  label far above the mean is one the critics could not find in entrepreneurial
  text.
- **`[taxonomy]`** — the `asked for X → panel read Y` tally. Direction, not just
  rate.
- **`[calibrated-pessimism]`** — No-Distortion specs broken out separately, split
  three ways (read as distorted / judged Unknown / dropped before the panel),
  plus what the panel read instead
  and the unanimity margin on the ones that passed. This is the central domain
  risk: *"I probably won't close this round in this market"* is realistic, not
  Fortune-telling, and if the panel can't reliably pass intended calibrated
  pessimism then the No-Distortion class comes out under-populated and skewed —
  which is itself a result.

**Critics do not anchor across rounds.** `_critique()` issues three fresh
stateless API calls each time; `build_critic_user` sends the text and nothing
else, with no history or prior verdict. So a spec that fails twice has failed two
independent trials. The *generator*, by contrast, is deliberately anchored — the
reviser prompt carries the critics' objection — so accepted `revised: true` rows
risk overcorrecting into caricature and are worth comparing against first-pass
rows.

---

## Running

```
venv\Scripts\python.exe -m pip install -r requirements-synth.txt
```

Keys: `ANTHROPIC_API_KEY` (generator) + `GEMINI_API_KEY` or `GOOGLE_API_KEY` (critics).

```
# free — no API calls, nothing written
venv\Scripts\python.exe -m src.synth.generate --n 300 --dry-run

# calibration: measure real acceptance / revision rates before scaling
venv\Scripts\python.exe -m src.synth.generate --n 20 --out data/synthetic/calibrate.jsonl

# pilot
venv\Scripts\python.exe -m src.synth.generate --n 300 --out data/synthetic/synthetic_train_en.jsonl
```

| Flag | Default |
|---|---|
| `--n` | 300 |
| `--seed` | 42 |
| `--gen-provider` / `--gen-model` | `anthropic` / `claude-sonnet-5` |
| `--critic-provider` / `--critic-model` | `gemini` / `gemini-2.5-flash` |
| `--n-critics` | 3 |
| `--critic-temps` | `0.3,0.6,0.9` |

---

## Reference dry run (`--n 300 --dry-run`)

```
[planner] 300 specs | primary distribution:
      90  No Distortion
      21  Mind Reading
      21  All-or-nothing thinking
      21  Overgeneralization
      21  Magnification
      21  Should statements
      21  Emotional Reasoning
      21  Fortune-telling
      21  Labeling
      21  Personalization
      21  Mental filter
[planner] granularity: {'reflection': 87, 'snippet': 213}
[planner] distinct (distortion x trigger) cells covered: 154
[planner] English seeds loaded: 10
[panel] generator = anthropic:claude-sonnet-5
[panel] critics   = 3 x gemini:gemini-2.5-flash @ temps [0.3, 0.6, 0.9]

[dry-run] sample generator user prompt for spec[0]:

Target pattern: No Distortion — A balanced, evidence-based reading of the situation. Realistic concern, disappointment, or a rational plan is NOT a distortion.

Situation to ground it in: imposter_feelings
Business stage: idea | sector: beauty | register: english_casual | undercurrent: gender_expectation
Name a concrete detail of this situation (a number, a person, a place, a time) and let the thinking follow from it.

Do NOT open with any of these, or a close variant:
- So today...
- I can't stop thinking...
- I keep replaying...
- I don't know why...
- Today was...
- Another day...
- I've been sitting here...
- It's been a long day...
- I should be happy...
- I don't even know where to start...

Style examples (do not copy — match the voice, not the words):
[Mind Reading] The mentor smiled and nodded during my update but I could tell she thinks my idea is a waste of the incubator's time.
[Mental filter] We hit our sales target and got great feedback at the expo, but all I can think about is the one stall visitor who frowned at my prices.
[Labeling] I pitched to three investors this week and all of them passed. I guess I'm just not cut out to run a company.
[Should statements] A proper entrepreneur should never take a day off, so taking Sunday to rest means I don't deserve this incubator seat.

Write the reflection now — about 150 words, 6-9 sentences, first person, no clinical words.

[dry-run] no API calls made, nothing written.
```

> The `--dry-run` output is rendered by the terminal in cp1252, so em-dashes may
> appear as `?`. The files themselves are UTF-8.

---

## Cost

Roughly **$0.009 per spec** that passes straight through; a revision doubles that
row. At 300 specs: **$3.30–4.00 (≈ ₨920–1,110)** with paid Gemini, less on the
free tier. The generator is ~75% of spend.

`PRICES` in `llm.py` carries Sonnet 5's **introductory** rate ($2/$10 per MTok),
valid through **2026-08-31**. After that the standard rate is $3/$15 — update the
dict or the printed cost silently under-reports by a third.

Unknown models fall back to `(0.0, 0.0)` in `estimate_cost`, so passing a
`--gen-model` not in `PRICES` makes the cost readout report $0.

---

## Operational gotchas

Things that bite you while *running* it. For the methodological caveats — what
the pipeline does not support and must not be claimed in the writeup — see
[`PIPELINE.md` §5](PIPELINE.md).

1. **A dropped critic verdict skews the vote.** If a critic's JSON fails to parse
   it is silently omitted, but `aggregate()` still divides by `n_critics`, making
   the majority harder to reach and inflating the revision rate — and therefore
   the cost.
2. **`_critique` is not wrapped in try/except**, so a Gemini rate-limit error
   ends the run. Rows already written survive (the `with` block closes cleanly);
   the remainder do not.
3. **Length bands vs. real data.** The Google Form asks for 60–120 words;
   `snippet` is 25–60 and `reflection` is 105–195. A length mismatch between
   synthetic train and real test is a domain-shift confound — part of any
   measured drop would be length, not content.
4. **Dedup will not catch homogeneity.** Threshold 0.8 on a word *set*; a
   near-verbatim paraphrase measures 0.762 and passes. Do not rely on the
   `rejected` counter to tell you the corpus is diverse — read the rows.
5. **Unknown models cost $0 in the readout.** `estimate_cost` falls back to
   `(0.0, 0.0)` for any `(provider, model)` not in `PRICES`, so a custom
   `--gen-model` silently reports zero spend.

---

## Papers

Full provenance — which paper contributes what, what is borrowed vs. adapted vs.
original, and the reference list — lives in **[`PIPELINE.md`](PIPELINE.md)**.
That file is the thesis-facing document; this one is the operational reference.

One-line framing: *AttrPrompt-style attributed generation (Yu et al., 2023) with
a KoACD-style multi-agent consensus critic (Kim & Kim, 2025) whose reasoning
follows Diagnosis of Thought (Chen et al., 2023), seeded and bootstrapped in the
manner of Self-Instruct (Wang et al., 2023).*
