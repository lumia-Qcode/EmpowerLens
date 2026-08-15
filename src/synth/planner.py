"""Deterministic, code-only planner (no LLM).

Emits one generation *spec* per target row. Guarantees class balance (rare
distortions are oversampled to an equal share rather than mirroring the natural
imbalance), the No-Distortion share, the ~26% secondary-distortion rate, and
random attribute combinations. Deterministic given ``seed`` so a run is
reproducible.

A spec is a plain dict so it serialises straight into the output JSONL:

    {
      "spec_id": "spec_0007",
      "primary": "Mind Reading",          # raw label string (or "No Distortion")
      "secondary": "Personalization" | None,
      "trigger": "investor_rejection",
      "register": "english_casual",
      "stage": "prototype",
      "sector": "handicrafts",
      "cultural_marker": None | "gender_expectation" | ...,
    }
"""

from __future__ import annotations

import random
from typing import Optional

from . import attributes as A


def _round_robin_counts(total: int, k: int) -> list[int]:
    """Split ``total`` into ``k`` near-equal integer buckets (oversamples rare)."""
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def build_specs(
    n: int,
    seed: int = 42,
    no_distortion_share: float = A.NO_DISTORTION_SHARE,
    secondary_rate: float = A.SECONDARY_RATE,
) -> list[dict]:
    """Return ``n`` generation specs, class-balanced and shuffled."""
    rng = random.Random(seed)

    n_no = round(n * no_distortion_share)
    n_dist = n - n_no

    # Equal allocation across the ten distortions (rare classes oversampled).
    per_class = _round_robin_counts(n_dist, len(A.DISTORTION_LABELS))

    specs: list[dict] = []

    def _attrs(trigger: str, granularity: str) -> dict:
        return {
            "trigger": trigger,
            "granularity": granularity,
            "register": rng.choice(A.REGISTERS),
            "stage": rng.choice(A.STAGES),
            "sector": rng.choice(A.SECTORS),
            "cultural_marker": rng.choice(A.CULTURAL_MARKERS),
        }

    def _granularities(count: int) -> list[str]:
        """Split a class's quota into snippets + full reflections, deterministically."""
        n_snip = round(count * A.SNIPPET_SHARE)
        return ["snippet"] * n_snip + ["reflection"] * (count - n_snip)

    # Scenario x distortion grid: walk TRIGGERS in order with a per-class offset
    # so every (distortion, trigger) cell fills evenly instead of being sampled
    # at random. This is the coverage figure for the thesis — and the honest
    # answer to "how did you ensure coverage?".
    n_trig = len(A.TRIGGERS)

    for cls_i, (primary, count) in enumerate(zip(A.DISTORTION_LABELS, per_class)):
        grans = _granularities(count)
        for j in range(count):
            secondary: Optional[str] = None
            # Only full reflections carry a second pattern — a 30-50 word
            # snippet cannot host two distortions without becoming a caricature.
            if grans[j] == "reflection" and rng.random() < secondary_rate:
                others = [d for d in A.DISTORTION_LABELS if d != primary]
                secondary = rng.choice(others)
            trigger = A.TRIGGERS[(cls_i + j) % n_trig]
            specs.append({"primary": primary, "secondary": secondary,
                          **_attrs(trigger, grans[j])})

    no_grans = _granularities(n_no)
    for j in range(n_no):
        trigger = A.TRIGGERS[(len(A.DISTORTION_LABELS) + j) % n_trig]
        specs.append({"primary": A.NO_DISTORTION, "secondary": None,
                      **_attrs(trigger, no_grans[j])})

    rng.shuffle(specs)
    for i, s in enumerate(specs):
        s["spec_id"] = f"spec_{i:05d}"
    return specs


def distribution(specs: list[dict]) -> dict[str, int]:
    """Count primary labels — for a quick sanity print / --dry-run."""
    out: dict[str, int] = {}
    for s in specs:
        out[s["primary"]] = out.get(s["primary"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))
