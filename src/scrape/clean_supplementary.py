"""S02: clean the scraped supplementary text and assign roles.

Reads every data/supplementary/raw/*.csv (S01 schema), then in this order:
  1. drop exact duplicates (same text after whitespace normalisation)
  2. scrub links, emails, @handles, and person names (spaCy en_core_web_sm PERSON -> [NAME])
  3. drop rows under 10 words
  4. drop non-English rows (share of English stop words < MIN_STOP_SHARE)
  5. drop near-duplicates (character 5-gram Jaccard >= NEAR_DUP_JACCARD with an earlier row)
Then gives item_id sup_NNN and assigns a role per thread: a random ROLE_SEED_SHARE of threads
(seeded shuffle, ROLE_SEED) -> seed, the rest -> evaluation. A whole thread gets one role.

Writes supplementary.csv and cleaning_log.md, and prints the thread/role check.

    python -m src.scrape.clean_supplementary
"""
from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

ROLE_SEED = 42
ROLE_SEED_SHARE = 0.30
MIN_WORDS = 10
MIN_STOP_SHARE = 0.25
NEAR_DUP_JACCARD = 0.80

URL = re.compile(r"(?i:https?://\S+|www\.\S+)|\b[\w-]+\.(com|io|net|org|co|ai|app|dev)(/\S*)?\b")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
HANDLE = re.compile(r"(?<!\w)@\w{2,}")
WORD = re.compile(r"[a-z']+")

# spaCy PERSON hits on the S01 raw text that are not people (products, companies, places, plain words).
# Checked by hand on 16 Nov 2026; these are left as written. Every other PERSON hit becomes [NAME].
NOT_PERSON = {
    "290k", "Aldi", "Alpha", "Angel", "App Store", "C++", "COVID", "Covid", "CS", "Career Sidekick",
    "Checkmk", "Churn Buster", "Cloud", "Cynefin", "Devs", "devs", "Dude", "Facebook Live", "Failed",
    "Find", "Fogbeam Labs", "GIT", "Gen AI", "Graphite", "Grass", "Groupon", "Java", "Jeez", "Konfig",
    "Lesson", "Lesson learned= Know", "Linux", "Mac Apps", "Mokusio", "Netto", "Onista", "Periscope",
    "Pixar", "Prezi", "Problem", "Refolo", "Robotics", "Scraper API", "Siempo", "Study Mode",
    "Sudden Facebook", "Ted Talk", "Twitter", "Universe", "WeFunder", "X", "X.", "XYZ", "YOLO",
    "faang", "jiu-jistu", "sauna", "~60%", "Chuck", "Chin",  # "Chuck-e-Cheese", "Chin up"
}


def scrub(texts: list[str], nlp) -> tuple[list[str], list[int]]:
    """Links, emails and handles first (so spaCy never sees them), then person names."""
    pre = [HANDLE.sub("[HANDLE]", URL.sub("[LINK]", EMAIL.sub("[EMAIL]", t))) for t in texts]
    out, n_names = [], []
    for doc in nlp.pipe(pre):
        s, k = doc.text, 0
        for e in reversed(doc.ents):
            if e.label_ == "PERSON" and e.text not in NOT_PERSON and "[" not in e.text:
                s = s[: e.start_char] + "[NAME]" + s[e.end_char :]
                k += 1
        out.append(re.sub(r"\s+", " ", s).strip())
        n_names.append(k)
    return out, n_names


def stop_share(t: str) -> float:
    w = WORD.findall(t.lower())
    return sum(x in STOP_WORDS for x in w) / max(len(w), 1)


def shingles(t: str, k: int = 5) -> set[str]:
    t = re.sub(r"\W+", " ", t.lower()).strip()
    return {t[i : i + k] for i in range(max(len(t) - k + 1, 1))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/supplementary/raw")
    ap.add_argument("--out", default="data/supplementary")
    a = ap.parse_args()
    raw_dir, out_dir = Path(a.raw), Path(a.out)

    files = sorted(raw_dir.glob("*.csv"))
    df = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False) for f in files], ignore_index=True)
    df = df.sort_values(["platform", "thread_id", "source_item_id"], kind="stable").reset_index(drop=True)
    n0 = len(df)
    dropped: list[tuple[str, str, str]] = []  # (source_item_id, reason, detail)

    norm = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    dup = norm.duplicated()
    dropped += [(i, "exact duplicate", "") for i in df.loc[dup, "source_item_id"]]
    df = df[~dup].reset_index(drop=True)

    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    df["text"], df["n_names"] = scrub(df["text"].tolist(), nlp)

    wc = df["text"].str.split().str.len()
    short = wc < MIN_WORDS
    dropped += [(i, "under 10 words", f"{w} words") for i, w in zip(df.loc[short, "source_item_id"], wc[short])]
    df = df[~short].reset_index(drop=True)

    ss = df["text"].map(stop_share)
    non_en = ss < MIN_STOP_SHARE
    dropped += [(i, "non-English", f"stop-word share {s:.2f}") for i, s in zip(df.loc[non_en, "source_item_id"], ss[non_en])]
    df = df[~non_en].reset_index(drop=True)

    sh = [shingles(t) for t in df["text"]]
    keep = []
    for i, s in enumerate(sh):
        hit = next(((j, len(s & sh[j]) / len(s | sh[j])) for j in keep if len(s & sh[j]) / len(s | sh[j]) >= NEAR_DUP_JACCARD), None)
        if hit:
            dropped.append((df.at[i, "source_item_id"], "near-duplicate", f"Jaccard {hit[1]:.2f} with {df.at[hit[0], 'source_item_id']}"))
        else:
            keep.append(i)
    df = df.loc[keep].reset_index(drop=True)

    threads = sorted(df["thread_id"].unique())
    random.Random(ROLE_SEED).shuffle(threads)
    seed_threads = set(threads[: round(ROLE_SEED_SHARE * len(threads))])
    df["role"] = df["thread_id"].map(lambda t: "seed" if t in seed_threads else "evaluation")
    df["item_id"] = [f"sup_{i:03d}" for i in range(1, len(df) + 1)]
    df["source"] = df["platform"]
    out = df[["item_id", "text", "source", "thread_id", "collected_at", "role"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "supplementary.csv", index=False, encoding="utf-8")

    # check: no thread in both roles
    roles_per_thread = out.groupby("thread_id")["role"].nunique()
    both = roles_per_thread[roles_per_thread > 1]
    by_role = out.groupby("role").agg(rows=("item_id", "size"), threads=("thread_id", "nunique"))
    check = (
        f"threads: {len(roles_per_thread)}\n"
        f"threads in both roles: {len(both)}\n"
        f"{by_role.to_string()}\n"
        + ("OK: every thread has exactly one role" if both.empty else f"FAIL: {list(both.index)}")
    )
    print(check)

    d = pd.DataFrame(dropped, columns=["source_item_id", "reason", "detail"])
    counts = d["reason"].value_counts().reindex(["exact duplicate", "under 10 words", "non-English", "near-duplicate"], fill_value=0)
    raw_counts = pd.concat([pd.read_csv(f, dtype=str, keep_default_na=False).assign(file=f.name) for f in files])["file"].value_counts().sort_index()
    per_source = out.groupby(["source", "role"]).size().unstack(fill_value=0)
    lines = [
        "# S02 cleaning log",
        "",
        f"Script: `src/scrape/clean_supplementary.py` (role seed {ROLE_SEED}, {ROLE_SEED_SHARE:.0%} of threads -> seed).",
        "",
        "## Input",
        "",
        "| Raw file | Rows |",
        "|---|---|",
        *[f"| `raw/{k}` | {v} |" for k, v in raw_counts.items()],
        f"| **Total** | **{n0}** |",
        "",
        "## Rows dropped",
        "",
        "| Step | Rule | Dropped |",
        "|---|---|---|",
        f"| 1 | exact duplicate (same text after whitespace normalisation) | {counts['exact duplicate']} |",
        f"| 3 | under {MIN_WORDS} words (after scrubbing) | {counts['under 10 words']} |",
        f"| 4 | non-English (English stop-word share < {MIN_STOP_SHARE}) | {counts['non-English']} |",
        f"| 5 | near-duplicate (character 5-gram Jaccard >= {NEAR_DUP_JACCARD}) | {counts['near-duplicate']} |",
        f"| | **Total dropped** | **{len(d)}** |",
        "",
        f"**Kept: {len(out)} of {n0}. Discard rate: {len(d) / n0:.1%}.**",
        "",
        "The S01 scrapers already keep only 10-250-word, first-person posts from English-language sites, so",
        "these S02 filters have little left to remove. Margins on the kept rows: shortest "
        f"{int(out['text'].str.split().str.len().min())} words; lowest English stop-word share "
        f"{out['text'].map(stop_share).min():.2f}; most similar pair Jaccard < {NEAR_DUP_JACCARD} by construction.",
        "",
        "## Scrubbed (step 2, rows kept)",
        "",
        f"- Person names -> `[NAME]` (spaCy `en_core_web_sm`): {int(df['n_names'].sum())} names in {int((df['n_names'] > 0).sum())} rows",
        f"- Links -> `[LINK]`: {int(out['text'].str.count(re.escape('[LINK]')).sum())}",
        f"- Emails -> `[EMAIL]`: {int(out['text'].str.count(re.escape('[EMAIL]')).sum())}",
        f"- Handles -> `[HANDLE]`: {int(out['text'].str.count(re.escape('[HANDLE]')).sum())}",
        "",
        "## Output by source and role",
        "",
        "| Source | evaluation | seed |",
        "|---|---|---|",
        *[f"| {s} | {r.get('evaluation', 0)} | {r.get('seed', 0)} |" for s, r in per_source.iterrows()],
        "",
        "## Check: no thread in both roles",
        "",
        "```",
        check,
        "```",
        "",
        "## Dropped rows",
        "",
        "| source_item_id | Reason | Detail |",
        "|---|---|---|",
        *[f"| {r.source_item_id} | {r.reason} | {r.detail} |" for r in d.itertuples()],
        "",
    ]
    (out_dir / "cleaning_log.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
