"""S01 — scrape founder interviews from Failory (/interview/ pages, server-rendered HTML).

robots.txt allows everything except /author/, /get/ and /startup/; only /interview/ URLs
from the sitemap are fetched. Each interview is a first-person Q&A with a failed-startup
founder. Paragraphs under setback questions (problem, mistakes, why it failed, what was
learned, ...) become rows; the interview slug is the thread, so one founder's paragraphs
stay in one S02 role.

    venv\\Scripts\\python.exe -m src.scrape.failory --target 170
"""
from __future__ import annotations

import argparse
import random
import re

from bs4 import BeautifulSoup

from src.scrape.common import (FIRST_PERSON, RAW_DIR, PoliteSession, append_rows, load_seen,
                               n_words, today)

OUT = RAW_DIR / "failory_raw.csv"
SITEMAP = "https://www.failory.com/sitemap.xml"
SETBACK_Q = re.compile(r"problem|mistake|fail|wrong|challeng|disadvantage|learn|differently|"
                       r"feel|hard|struggl|shut|regret|lesson|obstacle|difficult", re.I)
STOP_H = re.compile(r"more interviews|newsletter|learn more about|where can we go", re.I)
BOILER = re.compile(r"Join \d[\d,]*\+? founders|weekly guides and stories|90% of startups fail", re.I)
MIN_W, MAX_W, PER_THREAD = 10, 150, 6


def interview_urls(sess: PoliteSession) -> list[str]:
    locs = re.findall(r"<loc>([^<]+)</loc>", sess.get(SITEMAP).text)
    return [u for u in locs if "/interview/" in u]


def rows_for_page(url: str, html: str) -> list[dict]:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    soup = BeautifulSoup(html, "html.parser")
    out, take, idx = [], False, 0
    for el in soup.find_all(["h2", "h3", "p"]):
        if el.name in ("h2", "h3"):
            h = el.get_text(" ", strip=True)
            if STOP_H.search(h):
                break
            take = bool(SETBACK_Q.search(h))
            continue
        idx += 1
        t = " ".join(el.get_text(" ", strip=True).split())
        if (take and MIN_W <= n_words(t) <= MAX_W and not BOILER.search(t)
                and FIRST_PERSON.search(t)):
            out.append({"text": t, "url": url, "platform": "failory",
                        "thread_id": f"failory_{slug}", "source_item_id": f"failory_{slug}#{idx}",
                        "posted_at": "", "collected_at": today()})
        if len(out) >= PER_THREAD:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", type=int, default=170)
    ap.add_argument("--seed", type=int, default=42, help="order in which interviews are visited")
    args = ap.parse_args()

    sess = PoliteSession(min_delay=1.5)
    seen = load_seen(OUT)
    done = load_seen(OUT, "thread_id")
    n = len(seen)
    urls = interview_urls(sess)
    random.Random(args.seed).shuffle(urls)  # don't bias toward alphabetically-early startups
    print(f"{len(urls)} interview pages; resuming with {n} rows" if n else f"{len(urls)} interview pages")
    for url in urls:
        if n >= args.target:
            break
        slug = url.rstrip("/").rsplit("/", 1)[-1]
        if f"failory_{slug}" in done:
            continue
        rows = [r for r in rows_for_page(url, sess.get(url).text)
                if r["source_item_id"] not in seen][: args.target - n]
        append_rows(OUT, rows)
        seen.update(r["source_item_id"] for r in rows)
        n += len(rows)
        print(f"  {slug}: +{len(rows)}  (total {n})")
    print(f"done: {n} rows in {OUT}")


if __name__ == "__main__":
    main()
