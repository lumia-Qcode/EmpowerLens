"""S04 - build the generator's seed file from the S03 seed pool plus team-authored seeds.

    venv\\Scripts\\python.exe -m src.synth.build_seeds

Inputs
  data/supplementary/supplementary_labelled.csv   S03 labels; only rows with role == "seed" are used
  data/synthetic_gen/seeds/team_seeds.jsonl       team-authored seeds (English only)
Output
  data/synthetic_gen/seeds/entrepreneurial_seeds.jsonl
  data/synthetic_gen/seeds/seed_counts.csv        seeds per type x seed_source

Rules (D3 brief S04): up to 4 seed-pool rows per distortion type plus a few calibrated-pessimism rows; a seed is
never a supplementary-evaluation row (checked, fails loudly). Shortest rows first, because the generator mostly
writes 30-50-word snippets. A row over 120 words is cut to the sentence(s) holding its distorted part plus one
sentence either side, verbatim (`excerpt: true`).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.span_check import split_sentences
from src.synth import attributes as A

SUPP = Path("data/supplementary/supplementary_labelled.csv")
SEEDS_DIR = Path("data/synthetic_gen/seeds")
TEAM = SEEDS_DIR / "team_seeds.jsonl"
OUT = SEEDS_DIR / "entrepreneurial_seeds.jsonl"
PER_TYPE, N_CP, MAX_WORDS = 4, 2, 120


def excerpt(text: str, part: str, max_words: int = MAX_WORDS) -> tuple[str, bool]:
    """Whole text if short; else the span's sentences +-1, verbatim (falls back to the span's sentences)."""
    if len(text.split()) <= max_words:
        return text, False
    start = text.find(part)
    if start < 0:
        raise ValueError(f"distorted part not found in text: {part[:60]!r}")
    end = start + len(part)
    sents = split_sentences(text)
    hit = [i for i, (s, e) in enumerate(sents) if s < end and e > start]
    lo, hi = hit[0], hit[-1]
    for a, b in ((max(lo - 1, 0), min(hi + 1, len(sents) - 1)), (lo, hi)):
        out = text[sents[a][0]:sents[b][1]]
        if len(out.split()) <= max_words or (a, b) == (lo, hi):
            return out, True
    raise AssertionError("unreachable")


def _seed(r, text, cut, dominant):
    return {
        "id": None, "item_id": r["item_id"], "text": text,
        "distorted_part": r["distorted_part"] or None, "dominant_distortion": dominant,
        "secondary_distortion": r["secondary"] or None, "trigger": None, "register": "english_casual",
        "cultural_marker": None, "source": f"supplementary:{r['source']}", "seed_source": "supplementary",
        "excerpt": cut, "calibrated_pessimism": r["calibrated_pessimism"] == "True",
    }


def pick_supplementary(df: pd.DataFrame, per_type: int = PER_TYPE, n_cp: int = N_CP) -> list[dict]:
    pool = df[df["role"] == "seed"].copy()
    if pool.empty:
        raise ValueError("no seed-role rows in the input")
    pool["_w"] = pool["text"].str.split().str.len()
    pool = pool.sort_values(["_w", "item_id"])
    seeds = []
    for lab in A.DISTORTION_LABELS:
        for _, r in pool[pool["dominant"] == lab].head(per_type).iterrows():
            text, cut = excerpt(r["text"], r["distorted_part"])
            seeds.append(_seed(r, text, cut, lab))
    cp = pool[pool["calibrated_pessimism"] == "True"].head(n_cp)
    for _, r in cp.iterrows():
        seeds.append(_seed(r, r["text"], False, A.NO_DISTORTION))  # no span to anchor an excerpt: full text
    return seeds


def main() -> int:
    df = pd.read_csv(SUPP, dtype=str, keep_default_na=False)
    supp = pick_supplementary(df)
    role = dict(zip(df["item_id"], df["role"]))
    bad = [s["item_id"] for s in supp if role[s["item_id"]] != "seed"]
    if bad:
        raise SystemExit(f"seed taken from a non-seed row: {bad}")
    team = [json.loads(l) for l in TEAM.read_text(encoding="utf-8").splitlines() if l.strip()]
    if any(t.get("register") == "urdu_english_codemix" for t in team):
        raise SystemExit("team_seeds.jsonl must be English only")
    allowed = set(A.DISTORTION_LABELS) | {A.NO_DISTORTION}
    seeds = team + supp
    for i, s in enumerate(seeds, start=1):
        if s["dominant_distortion"] not in allowed:
            raise SystemExit(f"unknown label {s['dominant_distortion']!r}")
        s["team_id"] = s.pop("id", None) if s["seed_source"] == "team-authored" else None
        s["id"] = f"seed_{i:03d}"
    seeds = [{"id": s["id"], **{k: v for k, v in s.items() if k != "id"}} for s in seeds]
    OUT.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in seeds), encoding="utf-8")

    counts = (pd.DataFrame(seeds).assign(n=1)
              .pivot_table(index="dominant_distortion", columns="seed_source", values="n", aggfunc="sum", fill_value=0))
    counts["total"] = counts.sum(axis=1)
    counts.to_csv(SEEDS_DIR / "seed_counts.csv")
    print(counts.to_string())
    print(f"\nwrote {OUT}: {len(seeds)} seeds ({len(team)} team-authored, {len(supp)} supplementary, "
          f"{sum(s['excerpt'] for s in supp)} excerpts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
