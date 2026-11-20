# S04 — Seed file

`entrepreneurial_seeds.jsonl` holds 44 seeds, built by `python -m src.synth.build_seeds` from:

- **the S03 seed pool:** rows of `data/supplementary/supplementary_labelled.csv` with role `seed`;
- **team-authored seeds:** `team_seeds.jsonl`.

Every seed carries `seed_source` (`supplementary` or `team-authored`). Scraped seeds also carry their `item_id`.

## Seeds per type

| Type | supplementary | team-authored | total |
|---|---|---|---|
| All-or-nothing thinking | 2 | 2 | 4 |
| Emotional Reasoning | 0 | 4 | 4 |
| Fortune-telling | 3 | 1 | 4 |
| Labeling | 2 | 2 | 4 |
| Magnification | 2 | 2 | 4 |
| Mental filter | 0 | 4 | 4 |
| Mind Reading | 4 | 1 | 5 |
| Overgeneralization | 3 | 1 | 4 |
| Personalization | 1 | 3 | 4 |
| Should statements | 3 | 1 | 4 |
| No Distortion (calibrated pessimism) | 2 | 1 | 3 |
| **Total** | **22** | **22** | **44** |

### How the scraped seeds were chosen

- **Up to 4 per type from the seed pool,** shortest first, because the generator mostly writes 30–50-word snippets.
  Mind Reading is the only type with 4 in the pool; its one older team seed is kept as well.
- **Long rows become excerpts.** 11 rows over 120 words are cut to the sentences holding the distorted part,
  plus one sentence either side. The excerpt is verbatim and marked `"excerpt": true`.
- **Calibrated pessimism: the 2 shortest seed-pool rows,** used whole. They have no distorted part to anchor an
  excerpt.
- Same tables in `seed_counts.csv`.

### Team-authored seeds

- **10 are the original English seeds.** The 3 Urdu-English code-mix seeds (old `seed_007`, `seed_011`,
  `seed_013`) were removed, since the project is English only. The generator's English mode already skipped them.
- **12 are new, written to bring every type up to 4:**
  - Emotional Reasoning +4
  - Mental filter +3
  - Personalization +2
  - All-or-nothing thinking, Labeling, Magnification +1 each

  They are short, first-person, about business setbacks, and gender-neutral. Their `source` field says they were
  **drafted with Claude Code and must be reviewed by Lumia before S06** (ids `team_new_01` … `team_new_12` in
  `team_seeds.jsonl`).
- **Why so many are team-written.** The seed pool has no Emotional Reasoning or Mental filter rows and only one
  Personalization row, so those types depend almost entirely on team seeds.

## Role check

No seed comes from a supplementary-evaluation row:

```
$ python -c "import json,pandas as pd; r=dict(pd.read_csv('data/supplementary/supplementary_labelled.csv',dtype=str).set_index('item_id').role); s=[json.loads(l) for l in open('data/synthetic_gen/seeds/entrepreneurial_seeds.jsonl',encoding='utf-8')]; ids=[x['item_id'] for x in s if x.get('item_id')]; print(len(ids),'scraped seeds; roles:',{v:sum(r[i]==v for i in ids) for v in set(r.values())})"
22 scraped seeds; roles: {'seed': 22, 'evaluation': 0}
```

`build_seeds.py` also refuses to run if any chosen row is not in the seed pool. No seed is a survey response.

## Decision: gender conditioning removed (20 Nov 2026)

Chapter 2 says the training material is not conditioned on gender. The generator conditioned on it in four places,
and all four are now changed:

| Where | Before | Now |
|---|---|---|
| `src/synth/attributes.py` `CULTURAL_MARKERS` | `gender_expectation` as an undercurrent for about 1 in 5 specs | removed; `family_pressure` is the only marker left (1 in 4 specs) |
| `src/synth/attributes.py` `TRIGGERS` | `only_woman_in_the_room` scenario | removed (13 scenarios left) |
| `src/synth/prompts.py` generator system prompt | "self-reflections from **women** entrepreneurs in Pakistani startup incubators" | "self-reflections from entrepreneurs in Pakistani startup incubators" |
| `src/synth/prompts.py` and `attributes.py` instructions | "she thought / she concluded / she actually has evidence" | "the writer …" |
| team seed old `seed_004` | "they'll reject a **woman** running a workshop like mine", marker `gender_expectation` | "they'll reject someone running a small workshop like mine", marker none |

**Why remove it rather than explain it.**
- Removing it keeps Chapter 2 true as written.
- It matches the rest of the data: S01 scraped without filtering by gender, and the scraped seeds are
  gender-unknown.
- The women-only analysis is about who uses the app, not about what the model is trained on. It doesn't need
  gendered synthetic text.

**Chapter 7 can say:** "The synthetic generator is not conditioned on gender: no prompt, attribute or seed names
the writer's gender."

## Rows to check before S06

- **All 12 new team seeds:** read for clinical sense. They were drafted with Claude Code.
- **seed_040 (sup_384, Should statements).** The quoted "You should take your technology…" may be advice someone
  gave the writer rather than the writer's own rule. If so, the S03 label should be No Distortion and this seed
  dropped.
