"""Tests for src/synth/build_seeds.py (S04)."""

import pandas as pd
import pytest

from src.synth.build_seeds import excerpt, pick_supplementary


def _row(i, dom, text, part="", role="seed", cp=False):
    return {"item_id": f"sup_{i:03d}", "text": text, "role": role, "dominant": dom, "secondary": "",
            "distorted_part": part, "calibrated_pessimism": str(cp), "y_bin": "0" if dom == "No Distortion" else "1",
            "source": "hackernews", "thread_id": "t", "collected_at": "2026-11-06"}


def test_excerpt_keeps_short_text_whole():
    t = "We lost the deal. Nobody will ever buy from us."
    assert excerpt(t, "Nobody will ever buy from us.", max_words=120) == (t, False)


def test_excerpt_cuts_long_text_around_span_verbatim():
    filler = " ".join(f"Sentence {k} is filler text here." for k in range(40))
    t = f"{filler} We lost the big client. Nobody will ever buy from us. Then we moved on. {filler}"
    out, cut = excerpt(t, "Nobody will ever buy from us.", max_words=40)
    assert cut and out in t
    assert out == "We lost the big client. Nobody will ever buy from us. Then we moved on."


def test_pick_uses_seed_role_only_and_caps_per_type():
    rows = [_row(1, "Labeling", "I'm a failure.", "I'm a failure."),
            _row(2, "Labeling", "I'm useless at sales.", "I'm useless at sales.", role="evaluation")]
    rows += [_row(10 + k, "Mind Reading", f"They think I'm weak {k}.", f"They think I'm weak {k}.") for k in range(6)]
    rows += [_row(30, "No Distortion", "Sales fell, which worries me, and the market is down, so I expect a slow quarter.", cp=True)]
    seeds = pick_supplementary(pd.DataFrame(rows), per_type=4, n_cp=2)
    ids = [s["item_id"] for s in seeds]
    assert "sup_002" not in ids                                         # evaluation row never used
    assert sum(s["dominant_distortion"] == "Mind Reading" for s in seeds) == 4
    assert any(s["item_id"] == "sup_030" and s["dominant_distortion"] == "No Distortion" for s in seeds)
    assert all(s["seed_source"] == "supplementary" for s in seeds)


def test_pick_fails_when_seed_pool_is_empty():
    rows = [_row(1, "Labeling", "I'm a failure.", "I'm a failure.", role="evaluation")]
    with pytest.raises(ValueError):
        pick_supplementary(pd.DataFrame(rows), per_type=4, n_cp=0)
