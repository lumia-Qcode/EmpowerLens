"""Cheap, dependency-free near-duplicate filter.

Token-set Jaccard over lowercased word bags — good enough to catch the
repetitive templates LLMs fall into, with no torch / embedding dependency
(keeps the pilot off the Windows OpenMP path). Swap in sentence-embedding
cosine here later if you want semantic dedup; the interface stays the same.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9']+")


def _bag(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    sa, sb = _bag(a), _bag(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def is_duplicate(text: str, corpus: list[str], threshold: float = 0.8) -> bool:
    """True if ``text`` is >= ``threshold`` similar to anything in ``corpus``."""
    return any(jaccard(text, c) >= threshold for c in corpus)
