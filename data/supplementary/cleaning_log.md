# S02 cleaning log

Script: `src/scrape/clean_supplementary.py` (role seed 42, 30% of threads -> seed).

## Input

| Raw file | Rows |
|---|---|
| `raw/failory_raw.csv` | 170 |
| `raw/hn_raw.csv` | 330 |
| **Total** | **500** |

## Rows dropped

| Step | Rule | Dropped |
|---|---|---|
| 1 | exact duplicate (same text after whitespace normalisation) | 0 |
| 3 | under 10 words (after scrubbing) | 0 |
| 4 | non-English (English stop-word share < 0.25) | 0 |
| 5 | near-duplicate (character 5-gram Jaccard >= 0.8) | 0 |
| | **Total dropped** | **0** |

**Kept: 500 of 500. Discard rate: 0.0%.**

The S01 scrapers already keep only 10-250-word, first-person posts from English-language sites, so
these S02 filters have little left to remove. Margins on the kept rows: shortest 13 words; lowest English stop-word share 0.36; most similar pair Jaccard < 0.8 by construction.

## Scrubbed (step 2, rows kept)

- Person names -> `[NAME]` (spaCy `en_core_web_sm`): 20 names in 17 rows
- Links -> `[LINK]`: 42
- Emails -> `[EMAIL]`: 6
- Handles -> `[HANDLE]`: 2

## Output by source and role

| Source | evaluation | seed |
|---|---|---|
| failory | 133 | 37 |
| hackernews | 243 | 87 |

## Check: no thread in both roles

```
threads: 154
threads in both roles: 0
            rows  threads
role                     
evaluation   376      108
seed         124       46
OK: every thread has exactly one role
```

## Dropped rows

| source_item_id | Reason | Detail |
|---|---|---|
