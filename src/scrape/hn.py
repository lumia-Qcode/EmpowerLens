"""S01 — scrape founders' setback posts from Hacker News (free Algolia HN Search API, no key).

Finds threads with setback queries, then pulls each thread's post + comment tree in one call
(hn.algolia.com/api/v1/items/<id>). Keeps the story text and top-level comments that are
first-person, about a business, and 10-250 words. A per-thread cap stops one big thread
from dominating.

    venv\\Scripts\\python.exe -m src.scrape.hn --target 330
"""
from __future__ import annotations

import argparse
import re

from src.scrape.common import (FIRST_PERSON, RAW_DIR, PoliteSession, append_rows, clean_html,
                               load_seen, n_words, today)

OUT = RAW_DIR / "hn_raw.csv"
SEARCH = "https://hn.algolia.com/api/v1/search"
ITEM = "https://hn.algolia.com/api/v1/items/{}"

QUERIES = [
    "my startup failed", "shutting down my startup", "startup post mortem", "ran out of runway",
    "co-founder left", "cofounder quit", "investors passed", "failed to raise", "no customers",
    "no one is buying", "founder burnout", "should I quit my startup", "lost our biggest client",
    "lost our biggest customer", "failed business", "my side project failed", "bootstrapped struggling",
    "startup depression", "close my company", "pivot or quit", "launch got no users",
]
BUSINESS = re.compile(
    r"\b(startup|start-up|company|business|founder|co-?founder|customers?|clients?|revenue|"
    r"product|launch(ed)?|investors?|funding|raise|runway|saas|users|sales|employees|hire|"
    r"bootstrap\w*|side project|mrr|arr|shut (it )?down|failed|failure)\b", re.I)
MIN_W, MAX_W, PER_THREAD = 10, 250, 12


SECOND_PERSON = re.compile(r"\b(you|your|you're|you've|yours)\b", re.I)


def keep(text: str) -> bool:
    """Own-story test: about a business, and written mainly about the writer, not advice to 'you'."""
    fp = len(FIRST_PERSON.findall(text))
    sp = len(SECOND_PERSON.findall(text))
    return (MIN_W <= n_words(text) <= MAX_W and BUSINESS.search(text) is not None
            and fp >= 3 and fp >= 2 * sp)


def story_ids(sess: PoliteSession, pages: int) -> list[int]:
    ids: dict[int, None] = {}
    for q in QUERIES:
        for tag in ("ask_hn", "story"):
            for page in range(pages):
                hits = sess.get(SEARCH, params={"query": q, "tags": tag, "hitsPerPage": 50,
                                                "page": page,
                                                "numericFilters": "num_comments>=3"}).json()["hits"]
                for h in hits:
                    ids.setdefault(int(h["objectID"]))
                if len(hits) < 50:
                    break
    return list(ids)


def rows_for_thread(item: dict) -> list[dict]:
    sid = item["id"]
    out = []

    def row(node: dict, text: str) -> dict:
        return {"text": text, "url": f"https://news.ycombinator.com/item?id={node['id']}",
                "platform": "hackernews", "thread_id": f"hn_{sid}",
                "source_item_id": f"hn_{node['id']}", "posted_at": (node.get("created_at") or "")[:10],
                "collected_at": today()}

    body = clean_html(item.get("text"))
    if body and keep(body):
        out.append(row(item, body))
    for c in item.get("children") or []:
        if len(out) >= PER_THREAD:
            break
        t = clean_html(c.get("text"))
        if t and keep(t):
            out.append(row(c, t))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", type=int, default=330)
    ap.add_argument("--pages", type=int, default=2, help="search result pages per query/tag")
    args = ap.parse_args()

    sess = PoliteSession(min_delay=1.0)
    seen = load_seen(OUT)
    n = len(seen)
    print(f"resuming with {n} rows" if n else "starting fresh")
    ids = story_ids(sess, args.pages)
    print(f"{len(ids)} candidate threads")
    done_threads = load_seen(OUT, "thread_id")  # resume: skip threads already scraped
    for sid in ids:
        if n >= args.target:
            break
        if f"hn_{sid}" in done_threads:
            continue
        rows = [r for r in rows_for_thread(sess.get(ITEM.format(sid)).json())
                if r["source_item_id"] not in seen][: args.target - n]
        append_rows(OUT, rows)
        seen.update(r["source_item_id"] for r in rows)
        n += len(rows)
        if rows:
            print(f"  thread {sid}: +{len(rows)}  (total {n})")
    print(f"done: {n} rows in {OUT}")


if __name__ == "__main__":
    main()
