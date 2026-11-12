"""Shared helpers for the S01 scrapers: polite HTTP session, HTML cleaning, raw CSV I/O.

Raw schema (one row per post / comment / paragraph):
    text, url, platform, thread_id, source_item_id, posted_at, collected_at

Text is stored as scraped (HTML stripped only). Name/link removal, language filtering,
near-duplicate removal and role assignment are S02's job.
"""
from __future__ import annotations

import csv
import html
import re
import time
from datetime import date
from pathlib import Path

import requests

RAW_DIR = Path("data/supplementary/raw")
COLUMNS = ["text", "url", "platform", "thread_id", "source_item_id", "posted_at", "collected_at"]
USER_AGENT = "EmpowerLens-academic-research/1.0 (university FYP; non-commercial)"

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
FIRST_PERSON = re.compile(r"\b(I|I'm|I've|I'd|we|we're|we've|my|our|me|us)\b", re.I)


class PoliteSession:
    """requests.Session with a fixed User-Agent, a minimum delay between calls and retries."""

    def __init__(self, min_delay: float = 1.0, retries: int = 4):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = USER_AGENT
        self.min_delay, self.retries, self._last = min_delay, retries, 0.0

    def get(self, url: str, **kw) -> requests.Response:
        for attempt in range(self.retries):
            wait = self.min_delay - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            try:
                r = self.s.get(url, timeout=30, **kw)
                if r.status_code == 200:
                    return r
                if r.status_code not in (429, 500, 502, 503, 504):
                    r.raise_for_status()
            except requests.ConnectionError:
                pass
            time.sleep(2 ** (attempt + 1))
        raise RuntimeError(f"giving up on {url}")


def clean_html(s: str | None) -> str:
    """Strip tags, unescape entities, keep paragraph breaks as single spaces."""
    if not s:
        return ""
    s = re.sub(r"<\s*(p|br)\b[^>]*>", " ", s, flags=re.I)
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", s))).strip()


def n_words(s: str) -> int:
    return len(s.split())


def today() -> str:
    return date.today().isoformat()


def load_seen(path: Path, column: str = "source_item_id") -> set[str]:
    """Values of `column` already written, so an interrupted run resumes without duplicates."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {r[column] for r in csv.DictReader(f)}


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
