"""Distortion definitions + the attribute schema the planner samples from.

The distortion strings are the **raw** label strings expected by
``src/data.py``'s ``LABEL_CANON`` — so synthetic rows flow through the same
``_canon`` / ``make_targets`` path as the frozen ``Annotated_data.csv``.

Month 2 is the **English track only**: ``REGISTERS`` excludes Urdu/code-mix,
and cultural markers are kept light. The Urdu track reuses this module later
with an expanded register/marker set.
"""

# Raw label string -> short definition + entrepreneurial cue.
# Used verbatim in BOTH the generator and critic prompts so their notion of
# each distortion matches the annotation guide (guards against definitional drift).
DISTORTIONS = {
    "Mind Reading": "Assuming you know what others think — usually that they judge you negatively — without real evidence.",
    "Fortune-telling": "Predicting the future negatively and treating that prediction as a settled fact.",
    "Magnification": "Blowing the importance or consequences of something out of proportion (catastrophising), or shrinking the positives.",
    "Labeling": "Attaching a fixed, global label to yourself or others from one event ('I'm a failure', 'they're incompetent').",
    "Personalization": "Taking excessive responsibility for an outcome that had other causes ('it's all my fault').",
    "Overgeneralization": "Treating a single event as a never-ending pattern ('this always happens', 'no one ever...').",
    "Emotional Reasoning": "Taking a feeling as proof of fact ('I feel like a failure, so I must be one').",
    "Mental filter": "Fixating on one negative detail while ignoring the larger positive picture.",
    "Should statements": "Rigid rules about how you or others must behave ('a real founder should never rest').",
    "All-or-nothing thinking": "Seeing things in absolute, black-and-white terms with no middle ground.",
}

# Balanced, calibrated, NON-distorted interpretations. Not in the 10-dim
# multi-label vector (those rows are all-zeros), but a first-class training target.
NO_DISTORTION = "No Distortion"
NO_DISTORTION_DEF = (
    "A balanced, evidence-based reading of the situation. Realistic concern, "
    "disappointment, or a rational plan is NOT a distortion."
)

DISTORTION_LABELS = list(DISTORTIONS.keys())  # the ten, order fixed

# --- The two escape hatches, kept deliberately separate ----------------------
# KoACD's Analyzer prompt carries an enumerated list of conditions that force an
# immediate "Unknown", rather than a single soft "don't invent one" note. We copy
# that structure but split it in two, because "I cannot judge this" and "I judged
# this and it is healthy" are different verdicts. Collapsing them would label
# unjudgeable text as calibrated reasoning and pollute the No-Distortion class.

UNKNOWN = "Unknown"

# Return Unknown: the text cannot be judged at all.
UNKNOWN_CONDITIONS = [
    "assigning a label would require speculation beyond what the text says",
    "the text is only emotional expression with no reasoning or interpretation in it",
    "the text merely describes a situation, states facts, or asks a question",
    "the writer's intent or meaning is genuinely unclear",
    "the thought is attributed to someone else rather than held by the writer",
    "judging it would need conversational context that is not present",
    "there is negative emotion but no identifiable pattern of distorted reasoning",
    "there is no value judgement or personal interpretation to evaluate",
]

# Return No Distortion: the text CAN be judged, and the reasoning is sound.
NO_DISTORTION_CONDITIONS = [
    "the conclusion is proportionate to the evidence actually given in the text",
    "a negative prediction is well-calibrated to the domain — in a rejection-heavy "
    "field, expecting rejection is realism, not Fortune-telling",
    "the writer states a difficulty and then reasons about it soundly, or makes a plan",
    "disappointment, frustration or fear is expressed but the reasoning around it holds",
    "the writer acknowledges a real limitation she actually has evidence for",
]

# ---- Attribute dimensions the planner samples (AttrPrompt-style) -------------

TRIGGERS = [
    "investor_rejection", "loan_rejection", "cofounder_conflict", "low_sales",
    "customer_criticism", "competitor_comparison", "imposter_feelings",
    "missed_deadline", "team_setback", "family_disapproval", "supplier_problem",
    # Domain-specific scenarios — the situations are where EmpowerLens differs
    # from the clinical benchmark, so they are worth naming explicitly.
    "only_woman_in_the_room", "first_customer_or_churn", "mentor_feedback_stung",
]

# English track only for Month 2.
REGISTERS = ["english_formal", "english_casual"]

STAGES = ["idea", "prototype", "early_launch", "growing"]

SECTORS = ["food", "fashion", "handicrafts", "beauty", "tech", "services", "education"]

# Kept minimal on the English track; expands on the Urdu track.
CULTURAL_MARKERS = [None, None, None, "gender_expectation", "family_pressure"]

# ---- Distribution knobs (defaults mirror the real corpus, roughly) ----------

# Real corpus: 933/2530 == ~0.37 No Distortion. We use a bit less so the ten
# distortion classes get more per-class coverage in a small run.
NO_DISTORTION_SHARE = 0.30

# Real corpus: 416/1597 == ~0.26 of distorted rows carry a secondary.
SECONDARY_RATE = 0.26

# ---- Generation constraints (KoACD-style) -----------------------------------
# Source: KoACD's Appendix G/H generation constraints. Rationale for each is in
# the prompt builders; the lists live here so generator and critic share them.

# Two granularities, generated in one run:
#   snippet    - clean single-distortion signal for the classifier
#   reflection - longer, multi-distortion text, which is what the multi-label
#                head and the dashboard actually consume
GRANULARITIES = {
    "snippet": {
        "min_words": 25, "max_words": 60,
        "target": "30-50 words", "sentences": "2-3",
        "multi": False,
    },
    "reflection": {
        "min_words": 105, "max_words": 195,
        "target": "about 150 words", "sentences": "6-9",
        "multi": True,
    },
}
SNIPPET_SHARE = 0.70  # rest are full reflections

# The label vocabulary must NEVER appear in generated text. A classifier that
# sees "I'm probably catastrophising" learns the word, not the thinking pattern.
BANNED_TERMS = [
    "cognitive distortion", "distortion", "distorted", "cognitive bias",
    "mind reading", "fortune telling", "fortune-telling", "magnification",
    "magnifying", "catastrophise", "catastrophize", "catastrophising",
    "catastrophizing", "personalisation",
    "personalization", "overgeneralise", "overgeneralize", "overgeneralising",
    "overgeneralizing", "overgeneralisation", "overgeneralization",
    "emotional reasoning", "mental filter", "should statement",
    "all-or-nothing", "black and white thinking", "black-and-white thinking",
    "cbt", "therapy", "therapist", "irrational belief", "thought pattern",
    "negative self-talk", "inner critic",
]
# Deliberately NOT banned: "labeling"/"labelling". It is ordinary product
# vocabulary in the food / fashion / handicrafts sectors, so a substring ban
# would drop legitimate rows and skew the corpus by sector. The risk of a
# first-person vent naming the distortion "Labeling" is far smaller.

# --- Tier 2: stock lexical giveaways, applied SELECTIVELY --------------------
# For classes with a STRUCTURAL definition, ban the stock marker outright — the
# pattern has to be built out of reasoning rather than one token.
STRUCTURAL_MARKER_BANS = {
    "Overgeneralization": ["always", "never", "every single time", "everyone", "no one", "constantly"],
    "All-or-nothing thinking": ["completely", "totally", "utterly", "either way", "100%", "zero"],
    "Magnification": ["disaster", "catastrophe", "ruined", "the end of everything", "worst thing"],
    "Mind Reading": ["everyone thinks", "they all think", "I just know they", "obviously they"],
    "Fortune-telling": ["I know it will", "it's going to fail", "there's no way", "definitely going to"],
    "Mental filter": ["all I can see is", "the only thing that matters"],
}

# For classes whose definition IS partly lexical, banning the marker deletes the
# class rather than the leakage ("should statements" without "should" is not a
# should statement). Force variation in the surface form instead, so the model
# learns the construction rather than one token.
LEXICAL_VARIANT_REQUIREMENTS = {
    "Should statements": ["should", "ought to", "have to", "must", "supposed to",
                          "a real founder would", "any serious person would",
                          "I'm meant to"],
    "Labeling": ["I'm a failure", "I'm a fraud", "I'm not cut out for this",
                 "I'm just not a builder", "she's incompetent", "he's useless"],
    "Personalization": ["it's all my fault", "I caused this", "this is on me",
                        "if I had been better", "I'm the reason"],
    "Emotional Reasoning": ["I feel like", "it feels", "my gut says", "something in me knows"],
}

# Stock LLM story openers. Banned outright and topped up per call with whatever
# the current run has already produced (see generate.py).
STOCK_OPENERS = [
    "I keep replaying", "I can't stop thinking", "I've been sitting here",
    "Today was", "I don't know why", "It's been a long day", "I just got out of",
    "Sitting in my car", "I keep telling myself", "Another day", "So today",
    "I should be happy", "Honestly", "I don't even know where to start",
]


def definitions_block(include_no_distortion: bool = True) -> str:
    """Render the shared definition list injected into prompts (and cached)."""
    lines = [f"- {name}: {desc}" for name, desc in DISTORTIONS.items()]
    if include_no_distortion:
        lines.append(f"- {NO_DISTORTION}: {NO_DISTORTION_DEF}")
    return "\n".join(lines)
