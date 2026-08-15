"""Prompt builders for the generator, the blind critic, and the reviser.

Design notes
------------
* The **critic is blind** to the intended label. It diagnoses the text
  independently; the orchestrator then compares its verdict to the spec. This
  is what makes acceptance meaningful (a critic told "confirm Mind Reading"
  just rubber-stamps) and is the core anti-leakage guard.
* The critic sees the **full text**, not just the distorted span. Diagnosis of
  Thought needs a situation to separate facts from thoughts — hand it a bare
  span and stage 1 has nothing to work on. Instead the critic *emits* the span
  it judged from ("distorted_span"), which gives us KoACD's misclassification
  check, the ``Distorted part`` column that ``Annotated_data.csv`` has, and a
  free quality signal: critics who agree on the label but point at different
  spans mark a diffuse row.
* The shared ``definitions_block()`` goes into the *system* prompt of both the
  generator and the critic so their taxonomy matches — and so it can be cached.
* **Cache discipline:** anything that varies per call (banned openers, seed
  exemplars, the spec) lives in the USER prompt. The system prompt varies only
  by granularity, so there are exactly two cacheable prefixes.

Generation constraints below follow KoACD's Appendix G/H: a hard length cap
repeated several times, mandated first person, explicit situation grounding
with a cause-and-effect chain, a refreshed banned-opener list, and a ban on the
distortion vocabulary itself.
"""

from __future__ import annotations

from . import attributes as A

_DEFS = A.definitions_block()

# --------------------------------------------------------------------------- #
# Generator (Sonnet)                                                          #
# --------------------------------------------------------------------------- #

_BANNED_TERMS_LINE = ", ".join(f'"{t}"' for t in A.BANNED_TERMS[:18])


def build_generator_system(granularity: str = "snippet") -> str:
    """System prompt for one granularity. Stable across calls => cacheable."""
    g = A.GRANULARITIES[granularity]
    multi = (
        "Two patterns may appear: one dominant, one clearly secondary."
        if g["multi"] else
        "Exactly ONE thinking pattern. Do not blend in a second."
    )
    return f"""You write short, realistic first-person self-reflections from women \
entrepreneurs in Pakistani startup incubators, for an academic research dataset \
on cognitive distortions.

LENGTH: {g["target"]}, {g["sentences"]} sentences. This is a hard cap.

You will be told ONE target thinking pattern to portray, plus situational \
attributes. Write a single reflection in which that pattern is genuinely present \
in the *reasoning*, not just announced. {multi}

Cognitive distortion definitions:
{_DEFS}

Hard rules:
- FIRST PERSON, past or present tense, natural spoken voice — a real founder \
venting, not an essay and not a case study.
- GROUND IT IN A SPECIFIC SITUATION. State when/where it happened and what \
concretely occurred — a named meeting, a number, a message, a room. No floating \
aphorisms.
- BUILD A CAUSE-AND-EFFECT CHAIN: something happened -> she thought X about it \
-> therefore she concluded / felt / decided Y. The distortion must live in the \
link between the event and the conclusion.
- The target pattern must be the dominant way of thinking in the text.
- NEVER name the pattern or use clinical vocabulary. Forbidden anywhere in the \
output: {_BANNED_TERMS_LINE}. She is venting, not diagnosing herself.
- Vary your phrasing. Do not reuse sentence shapes or openers across outputs.
- English only. No emojis, no hashtags, no meta-commentary, no quotation marks \
around the whole text, no title.
- If the target is "No Distortion", write a balanced, evidence-based reaction: \
realistic concern or a calm plan, still fully grounded in a specific situation, \
with NO distorted reasoning.
- Output ONLY the reflection text.

Re-read the LENGTH rule before you answer: {g["target"]}, {g["sentences"]} \
sentences. Going over is a failure even if the writing is good."""


# Back-compat: modules that imported the old constant get the snippet form.
SYSTEM_GENERATOR = build_generator_system("snippet")


def _seed_examples(seeds: list[dict], k: int, rng) -> str:
    """Format up to k rotating seed exemplars as few-shot demonstrations."""
    if not seeds:
        return ""
    chosen = rng.sample(seeds, min(k, len(seeds)))
    lines = []
    for s in chosen:
        lines.append(f'[{s.get("dominant_distortion", "?")}] {s["text"]}')
    return "Style examples (do not copy — match the voice, not the words):\n" + "\n".join(lines)


def _tier2_constraint(primary: str, rng) -> str:
    """Tier-2 lexical control, applied selectively.

    Structural classes: ban the stock marker outright — the pattern has a
    non-lexical definition, so removing the token removes leakage, not the class.

    Lexical classes: banning the marker would delete the class ("should
    statements" without "should" is not a should statement). Force variation in
    the surface form instead, so the classifier learns the construction rather
    than one token.
    """
    if primary in A.STRUCTURAL_MARKER_BANS:
        banned = A.STRUCTURAL_MARKER_BANS[primary]
        return ("Do NOT use any of these as a shortcut: "
                + ", ".join(f'"{w}"' for w in banned)
                + ". Build the pattern out of the reasoning instead.")
    if primary in A.LEXICAL_VARIANT_REQUIREMENTS:
        opts = A.LEXICAL_VARIANT_REQUIREMENTS[primary]
        pick = rng.choice(opts)
        return (f'This pattern is partly carried by its wording, so keep it — but '
                f'phrase it as something close to "{pick}" rather than the most '
                f'obvious form. Vary the construction, not just the noun.')
    return ""


def _banned_openers_block(recent: list[str] | None, rng, k: int = 10,
                          overused: list[str] | None = None) -> str:
    """Stock openers + whatever this run has already used. Refreshed per call.

    Two granularities on purpose. Banning only the opening *four* words is
    looser than it looks — "I have always felt" and "I have never felt" are
    different 4-grams but the same tic, so both would slip through and the
    corpus still collapses onto one opening shape. ``overused`` carries the
    short (1-2 word) starts that have already become habitual, which is the
    constraint that actually bites.
    """
    parts = []
    pool = list(A.STOCK_OPENERS) + list(recent or [])
    if pool:
        chosen = rng.sample(pool, min(k, len(pool)))
        parts.append("Do NOT open with any of these, or a close variant:\n"
                     + "\n".join(f"- {o}..." for o in chosen))
    if overused:
        parts.append("These sentence-starts are already overused in this batch — "
                     "begin somewhere else entirely: "
                     + ", ".join(f'"{o}"' for o in overused[:8]) + ".")
    return "\n\n".join(parts)


def build_generator_user(
    spec: dict,
    seeds: list[dict],
    rng,
    k: int = 4,
    recent_openers: list[str] | None = None,
    overused_starts: list[str] | None = None,
) -> str:
    primary = spec["primary"]
    granularity = spec.get("granularity", "snippet")
    g = A.GRANULARITIES[granularity]

    if primary == A.NO_DISTORTION:
        target = f'Target pattern: {A.NO_DISTORTION} — {A.NO_DISTORTION_DEF}'
    else:
        target = f'Target pattern: {primary} — {A.DISTORTIONS[primary]}'
        if spec.get("secondary") and g["multi"]:
            sec = spec["secondary"]
            target += (f'\nAlso weave in a secondary pattern, more subtly: '
                       f'{sec} — {A.DISTORTIONS[sec]}')

    ctx = (f'Situation to ground it in: {spec["trigger"]}\n'
           f'Business stage: {spec["stage"]} | sector: {spec["sector"]} | '
           f'register: {spec["register"]}')
    if spec.get("cultural_marker"):
        ctx += f' | undercurrent: {spec["cultural_marker"]}'
    ctx += ("\nName a concrete detail of this situation (a number, a person, a "
            "place, a time) and let the thinking follow from it.")

    parts = [target, ctx]

    tier2 = _tier2_constraint(primary, rng)
    if tier2:
        parts.append(tier2)

    openers = _banned_openers_block(recent_openers, rng, overused=overused_starts)
    if openers:
        parts.append(openers)

    ex = _seed_examples(seeds, k, rng)
    if ex:
        parts.append(ex)

    parts.append(f'Write the reflection now — {g["target"]}, {g["sentences"]} '
                 f'sentences, first person, no clinical words.')
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- #
# Critic — BLIND, full context, Diagnosis-of-Thought, emits the span           #
# --------------------------------------------------------------------------- #

_UNKNOWN_BLOCK = "\n".join(f"- {c}" for c in A.UNKNOWN_CONDITIONS)
_ND_BLOCK = "\n".join(f"- {c}" for c in A.NO_DISTORTION_CONDITIONS)

SYSTEM_CRITIC = f"""You are a careful clinical annotator labelling cognitive \
distortions in a self-reflection, for a research dataset. You do NOT know what \
the text was written to portray — diagnose it independently and honestly.

Judge the WHOLE text and its overall flow. Do not label from one isolated \
sentence: a single dramatic line inside otherwise sound reasoning is not a \
distortion.

Cognitive distortion definitions:
{_DEFS}

Method (Diagnosis of Thought):
1. Separate the objective FACTS of the situation from the writer's THOUGHTS / \
interpretations about them.
2. For the main thoughts, weigh the evidence FOR and AGAINST them.
3. Trace the cause-and-effect chain: what happened -> what she concluded from \
it -> why that step does or does not follow.
4. Decide which distortion schema(s), if any, are actually present.

STOP AND RETURN "{A.UNKNOWN}" if ANY of these hold. This verdict is final — do \
not reconsider or talk yourself into a label afterwards:
{_UNKNOWN_BLOCK}

Return "No Distortion" — a different verdict, meaning you CAN judge it and the \
reasoning is sound — if any of these hold:
{_ND_BLOCK}

EVIDENCE IS MANDATORY. Quote at least TWO complete sentences from the original \
text, verbatim and exactly as written (or the entire text if it is shorter than \
two sentences). If you cannot find two sentences that support your verdict, \
your verdict is "{A.UNKNOWN}".

Then return your verdict as a JSON object with exactly these fields:
- "facts": string (one line)
- "thoughts": string (one line)
- "reasoning": string — the cause-and-effect chain from step 3, in complete \
sentences: what happened, what she concluded, why that step does or does not follow
- "evidence": array of at least two verbatim sentences from the text
- "present": array of distortion names actually present (use the exact names \
above; ["No Distortion"] if the reasoning is sound; ["{A.UNKNOWN}"] if unjudgeable)
- "primary": the single most prominent name, or "No Distortion", or "{A.UNKNOWN}"
- "secondary": the next most prominent name, or null
- "distorted_span": the SHORTEST verbatim span carrying the primary (this is a \
tighter quote than "evidence" — a phrase, not sentences), or null
- "confidence": number 0.0-1.0 for how clear the primary is

Only report a distortion you can justify from the quoted evidence. Do not invent \
one to be safe, and do not stretch an interpretation to make one fit. Return \
ONLY the JSON object."""


def build_critic_user(text: str) -> str:
    return f"Reflection to diagnose:\n\n{text}"


CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "string"},
        "thoughts": {"type": "string"},
        "reasoning": {"type": "string"},
        # minItems is advisory here — provider support for it varies, so the
        # count is also checked in generate.py rather than trusted.
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "present": {"type": "array", "items": {"type": "string"}},
        "primary": {"type": "string"},
        "secondary": {"type": ["string", "null"]},
        "distorted_span": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
    },
    "required": ["facts", "thoughts", "reasoning", "evidence", "present",
                 "primary", "secondary", "distorted_span", "confidence"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# Reviser (Sonnet) — one fixable retry                                        #
# --------------------------------------------------------------------------- #

def build_reviser_user(spec: dict, text: str, critic_note: str, seeds: list[dict],
                       rng, recent_openers: list[str] | None = None,
                       overused_starts: list[str] | None = None) -> str:
    base = build_generator_user(spec, seeds, rng, recent_openers=recent_openers,
                                overused_starts=overused_starts)
    return (f"A previous attempt did not clearly show the target pattern.\n\n"
            f'Previous attempt: "{text}"\n\n'
            f"Independent reviewers instead read it as: {critic_note}\n\n"
            f"Rewrite it so the TARGET pattern below is clearly the dominant "
            f"reasoning, while staying natural, specific and grounded in the "
            f"situation.\n\n{base}")
