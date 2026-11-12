# S01 sources — entrepreneurial setback text

Collected 12 Nov 2026. None of the sources is filtered by gender; all are public, need no login
and cost nothing. Raw files are in `raw/`, one row per post / comment / paragraph, with columns
`text, url, platform, thread_id, source_item_id, posted_at, collected_at`.

| Platform | Scraped by | Raw rows | After exact dedup | Threads | File |
|---|---|---|---|---|---|
| Hacker News | Lumia | 330 | 330 | 121 | `raw/hn_raw.csv` |
| Failory (founder interviews) | Lumia | 170 | 170 | 33 | `raw/failory_raw.csv` |
| Indie Hackers | Izza | — | — | — | `raw/indiehackers_raw.csv` *(to add)* |
| **Total so far** | | **500** | **500** | 154 | target 489 after exact dedup: **met** |

Regenerate the counts with `venv\Scripts\python.exe -m src.scrape.dedup_check`.

## Hacker News

- **Access:** the free Algolia HN Search API (`hn.algolia.com/api/v1/search` and `/items/<id>`). No
  key and no login. This is the API the HN site itself uses for search; no HTML is scraped. The
  official Firebase API (`hacker-news.firebaseio.com/v0/item/<id>.json`) holds the same items by ID.
- **Terms:** content is copyrighted by its authors (Y Combinator's terms; see
  `D3/FYP scraping sources.pdf`). Training a model on it is treated as fair use. **Verbatim text
  must not be republished**; a public release ships only HN item IDs plus labels and re-fetches the
  text at runtime.
- **What was kept** (`src/scrape/hn.py`):
  - threads found with 21 setback queries ("my startup failed", "ran out of runway",
    "co-founder left", "founder burnout", …) on `ask_hn` and `story`, at least 3 comments;
  - the post text and top-level comments of 10–250 words that mention a business and are an own
    story: at least 3 first-person words and at least twice as many first-person as "you" words,
    which drops advice aimed at someone else;
  - at most 12 rows per thread.
- `thread_id = hn_<story id>`.

## Failory

- **Access:** plain HTML pages from `failory.com/sitemap.xml`, `/interview/` URLs only, fetched
  1.5 s apart. robots.txt (checked 12 Nov 2026) says `Allow: /` and disallows only `/author/`,
  `/get/` and `/startup/`, none of which are fetched. `/cemetery/` posts are skipped because they
  are third-person journalism, not founder voice.
- **Terms:** no formal ToS. Text is copyrighted by Failory LLC / the founders. Training is fair use;
  **no public redistribution of verbatim text**. A public release ships only URLs plus labels.
- **What was kept** (`src/scrape/failory.py`):
  - interviews visited in a seeded random order (seed 42);
  - first-person paragraphs of 10–150 words under setback questions (problem, mistakes, why it
    failed, challenges, what was learned, …);
  - at most 6 paragraphs per interview.
- `thread_id = failory_<slug>`, so a whole interview goes to one S02 role.

## Storage

The repository is private, so raw text is committed as S01 requires. Before any public release,
replace `raw/` with pointer files (`platform, source_item_id / url, thread_id` plus labels) and
ship the scrapers so users can re-fetch the text.
