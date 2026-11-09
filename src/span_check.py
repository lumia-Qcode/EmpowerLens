"""
T11 data check — how do `Distorted part` spans sit inside `Patient Question`?

Run from repo root:   python -m src.span_check          (put this file in src/)
Reads Annotated_data.csv only — no splits, no test.csv, no training.
Writes results/span_check_rows.csv (per row) and results/span_check_summary.txt
(the block to paste back). v2: adds head/tail ANCHOR matching for spans the
annotators edited (match order: exact > normalized > anchor > fuzzy > none),
reports how far anchored regions stretch beyond the quoted span (contiguous
vs joined), and compares located-subset stats to expose selection bias.
Sentence rule: regex splitter that protects common
abbreviations/initials, and treats '...' as a boundary only before an uppercase
letter; blank lines are boundaries (single newlines = hard-wrap, treated as a space). A sentence counts as "hit" by the span if they
share >=1 char ("any") — also reported at >=20% of the sentence ("sub").
"""

import re
import difflib
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_raw, LABEL_CANON, TEXT_COL, DOMINANT_COL, SECONDARY_COL, ID_COL

SPAN_COL = "Distorted part"
ANCHOR_K = 6            # words in head/tail anchors (span needs >= 2*K words)
FUZZY_MIN = 0.80        # longest common block / span length to accept a fuzzy match
SUB_FRAC = 0.20         # "sub" rule: sentence counts if span covers >=20% of it
WHOLE_FRAC = 0.90       # sentence counts as "whole" if span covers >=90% of it
ABBREV = {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "e.g", "i.e", "inc", "ltd", "approx", "fig"}
OUT_DIR = Path("results")


def split_sentences(text):
    """Return list of (start, end) char offsets, whitespace-trimmed."""
    cuts = []
    for m in re.finditer(r'[.!?]+["\')\]\u201d\u2019]*\s+|\n\s*\n', text):
        g = m.group()
        if g.strip() == "":                      # blank-line boundary
            cuts.append(m.end())
            continue
        nxt = text[m.end():m.end() + 1]
        tok = re.search(r"(\S+)$", text[:m.start()])
        tok = tok.group(1).lower().lstrip("(\"'[\u201c\u2018") if tok else ""
        if g.startswith("...") and not nxt.isupper():
            continue                              # mid-sentence ellipsis
        if g.count(".") == 1 and "!" not in g and "?" not in g:
            if tok in ABBREV or re.fullmatch(r"[A-HJ-Za-hj-z]", tok):
                continue                          # abbreviation / initial
        cuts.append(m.end())
    bounds = [0] + cuts + [len(text)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        seg = text[a:b]
        s = a + (len(seg) - len(seg.lstrip()))
        e = b - (len(seg) - len(seg.rstrip()))
        if e > s:
            out.append((s, e))
    return out


def norm_map(s):
    """Normalise (quotes, dashes, case, whitespace runs) and keep a map back to original offsets."""
    rep = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u2014": "-", "\u2013": "-", "\u2026": "..."}
    out, idx, prev_space = [], [], False
    for i, ch in enumerate(s):
        for c in rep.get(ch, ch):
            if c.isspace():
                if prev_space:
                    continue
                c, prev_space = " ", True
            else:
                prev_space = False
                low = c.lower()
                c = low if len(low) == 1 else c
            out.append(c)
            idx.append(i)
    return "".join(out), idx


def locate_span(text, span):
    """Return (start, end, kind, anchor_ratio); kind: exact / normalized / anchor / fuzzy / none.
    anchor_ratio = anchored region length / quoted span length (~1 = contiguous quote, >>1 = joined/edited)."""
    span = str(span).strip()
    if not span:
        return None, None, "none", None
    i = text.find(span)
    if i >= 0:
        return i, i + len(span), "exact", None
    t, ti = norm_map(text)
    p = norm_map(span)[0].strip()
    i = t.find(p)
    if i >= 0:
        return ti[i], ti[i + len(p) - 1] + 1, "normalized", None
    w = p.split(" ")
    if len(w) >= 2 * ANCHOR_K:
        head, tail = " ".join(w[:ANCHOR_K]), " ".join(w[-ANCHOR_K:])
        i = t.find(head)
        if i >= 0:
            j = t.find(tail, i)
            if j >= 0:
                e = j + len(tail)
                return ti[i], ti[e - 1] + 1, "anchor", (e - i) / len(p)
    m = difflib.SequenceMatcher(None, t, p, autojunk=False).find_longest_match(0, len(t), 0, len(p))
    if m.size / len(p) >= FUZZY_MIN:
        s0 = max(0, m.a - m.b)
        e0 = min(len(t), s0 + len(p))
        return ti[s0], ti[e0 - 1] + 1, "fuzzy", None
    return None, None, "none", None


def mark(text, sents, s, e, width=420):
    """Text with sentence ends as ' ¦ ' and the span as [[ ... ]] for eyeballing."""
    ins = [(b, " ¦ ") for _, b in sents[:-1]] + [(s, "[["), (e, "]]")]
    for pos, tag in sorted(ins, key=lambda x: -x[0]):
        text = text[:pos] + tag + text[pos:]
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width] + " ..."


def pct(a, b):
    return f"{a} ({100 * a / b:.1f}%)" if b else "0 (n/a)"


def main():
    df = load_raw("Annotated_data.csv")
    recs = []
    for _, r in df.iterrows():
        text = r[TEXT_COL]
        span = r.get(SPAN_COL)
        dom = LABEL_CANON[str(r[DOMINANT_COL]).strip()]
        sents = split_sentences(text)
        has_span = pd.notna(span) and str(span).strip() != ""
        rec = dict(id=r[ID_COL], dominant=dom, distorted=dom != "no_distortion",
                   has_secondary=pd.notna(r[SECONDARY_COL]), has_span=has_span,
                   n_sent=len(sents), text_words=len(text.split()), match="no_span",
                   text=text, span=str(span) if has_span else "")
        if has_span:
            s, e, kind, ar = locate_span(text, span)
            rec["match"] = kind
            rec["anchor_ratio"] = ar
            rec["span_words"] = len(str(span).split())
            rec["multi_fragment"] = bool(re.search(r"\.\.\.|…|\n", str(span).strip()))
            if s is not None:
                ov = [max(0, min(b, e) - max(a, s)) for a, b in sents]
                cov = [o / (b - a) for o, (a, b) in zip(ov, sents)]
                gold_any = [i for i, o in enumerate(ov) if o > 0]
                gold_sub = [i for i, c in enumerate(cov) if c >= SUB_FRAC]
                gold = gold_any
                if kind == "anchor":
                    # edited/joined quote: gold = sentences whose full text appears in the quote
                    pn = norm_map(str(span))[0].strip()
                    cont = [i for i in gold_any if norm_map(text[sents[i][0]:sents[i][1]])[0].strip() in pn]
                    rec["a_region_sent"], rec["a_contained"] = len(gold_any), len(cont)
                    gold = cont if cont else gold_any
                n_any = len(gold)
                noncontig = int(gold[-1] - gold[0] + 1 > len(gold))
                if n_any == 1:
                    stype = "whole_sentence" if cov[gold[0]] >= WHOLE_FRAC else "sub_sentence"
                else:
                    stype = "multi_sentence"
                rec.update(start=s, end=e, n_any=n_any, n_sub=len(gold_sub), span_type=stype, noncontig=noncontig,
                           span_share=(e - s) / len(text),
                           first_hit=int(0 in gold), last_hit=int(len(sents) - 1 in gold),
                           rand_hit=len(gold) / len(sents), marked=mark(text, sents, s, e))
        recs.append(rec)
    d = pd.DataFrame(recs)

    dist = d[d.distorted]
    ds = dist[dist.has_span]
    loc = ds[ds.match.isin(["exact", "normalized", "anchor", "fuzzy"])].copy()
    multi = loc[loc.n_sent >= 2]

    L = []
    out = L.append
    out("=== T11 SPAN DATA CHECK ===")
    out("\n[A] Coverage")
    out(f"rows total               : {len(d)}")
    out(f"distorted rows           : {len(dist)}")
    out(f"  with a span            : {pct(len(ds), len(dist))}")
    out(f"  missing span           : {pct(len(dist) - len(ds), len(dist))}")
    out(f"no_distortion with span  : {int(d[~d.distorted].has_span.sum())} of {int((~d.distorted).sum())}")

    out("\n[B] Span match inside text (distorted rows with a span)")
    for k in ("exact", "normalized", "anchor", "fuzzy", "none"):
        out(f"  {k:<11}: {pct(int((ds.match == k).sum()), len(ds))}")

    anc = ds[ds.match == "anchor"]
    out(f"\n[B2] Anchored spans (n={len(anc)}): region length / quoted span length")
    if len(anc):
        r_ = anc.anchor_ratio
        out(f"  median {r_.median():.2f} | p90 {r_.quantile(.9):.2f} | <=1.15 (contiguous edit): {pct(int((r_ <= 1.15).sum()), len(anc))}"
            f" | 1.15-1.5: {pct(int(((r_ > 1.15) & (r_ <= 1.5)).sum()), len(anc))} | >1.5 (likely joined/non-contiguous): {pct(int((r_ > 1.5).sum()), len(anc))}")

    if len(anc):
        out("[B3] Anchored rows: sentences inside the anchored region vs sentences whose full text appears in the quoted span")
        out(f"  region sentences {int(anc.a_region_sent.sum())} | contained in quote {int(anc.a_contained.sum())} "
            f"({100 * anc.a_contained.sum() / anc.a_region_sent.sum():.1f}%) | rows where ALL region sentences are contained: "
            f"{pct(int((anc.a_contained == anc.a_region_sent).sum()), len(anc))} | rows with zero contained: {pct(int((anc.a_contained == 0).sum()), len(anc))}")
        out("  (contained << region => quote is a non-contiguous subset; labelling 'every sentence between anchors' would over-label)")

    out(f"\n[C] Text structure (located spans, n={len(loc)})")
    out(f"sentences/text   mean {loc.n_sent.mean():.2f}  median {loc.n_sent.median():.0f}  p90 {loc.n_sent.quantile(.9):.0f}  max {loc.n_sent.max()}")
    out(f"single-sentence texts (selection trivial): {pct(int((loc.n_sent == 1).sum()), len(loc))}")
    out(f"texts with >=3 sentences                 : {pct(int((loc.n_sent >= 3).sum()), len(loc))}")
    out(f"words/text       median {loc.text_words.median():.0f}  p90 {loc.text_words.quantile(.9):.0f}")

    out("\n[D] Span vs sentences (located spans)")
    for k in ("sub_sentence", "whole_sentence", "multi_sentence"):
        out(f"  {k:<15}: {pct(int((loc.span_type == k).sum()), len(loc))}")
    out(f"gold sentences per row (overlap rule; containment rule for anchored rows): 1 -> {pct(int((loc.n_any == 1).sum()), len(loc))} | 2 -> {pct(int((loc.n_any == 2).sum()), len(loc))} | 3+ -> {pct(int((loc.n_any >= 3).sum()), len(loc))}")
    nz = loc[loc.match != "anchor"]
    out(f"rows where 'any' and 'sub>=20%' rules disagree on count (non-anchored): {pct(int((nz.n_any != nz.n_sub).sum()), len(nz))}")
    m2 = loc[loc.n_any >= 2]
    out(f"rows with >=2 gold sentences that are NON-adjacent (gap between them): {pct(int(m2.noncontig.sum()), len(m2))}"
        f" | all located rows: {pct(int(loc.noncontig.sum()), len(loc))}")
    out(f"multi-fragment spans (ellipsis/newline in span)       : {pct(int(ds.multi_fragment.sum()), len(ds))}")

    out("\n[E] Span size")
    out(f"span words       median {loc.span_words.median():.0f}  mean {loc.span_words.mean():.1f}  p90 {loc.span_words.quantile(.9):.0f}")
    out(f"span share of text chars  median {loc.span_share.median():.2f}  | spans >=90% of text: {pct(int((loc.span_share >= .9).sum()), len(loc))}")

    out("\n[F] Trivial selector baselines (hit = picked sentence overlaps gold span)")
    out(f"all located rows  : first {loc.first_hit.mean():.3f} | last {loc.last_hit.mean():.3f} | random {loc.rand_hit.mean():.3f}")
    out(f"multi-sentence only (n_sent>=2, n={len(multi)}): first {multi.first_hit.mean():.3f} | last {multi.last_hit.mean():.3f} | random {multi.rand_hit.mean():.3f}")

    out("\n[G] Cross-boundary (multi_sentence) rate")
    for flag in (False, True):
        sub = loc[loc.has_secondary == flag]
        out(f"  secondary={str(flag):<5}: {pct(int((sub.span_type == 'multi_sentence').sum()), len(sub))}")
    out("  by dominant class (n | multi% | sub-sentence%):")
    for cls, sub in loc.groupby("dominant"):
        out(f"    {cls:<20} {len(sub):>4} | {100 * (sub.span_type == 'multi_sentence').mean():5.1f} | {100 * (sub.span_type == 'sub_sentence').mean():5.1f}")

    out("\n[I] Selection-bias check: stats by how the span was located")
    out("  subset                    n   span_words(med)  n_sent(med)  multi%  whole%  sub%")
    for name, sub in (("exact+normalized", loc[loc.match.isin(["exact", "normalized"])]),
                      ("anchor+fuzzy", loc[loc.match.isin(["anchor", "fuzzy"])]),
                      ("ALL located", loc)):
        if len(sub):
            out(f"  {name:<20} {len(sub):>6}  {sub.span_words.median():>14.0f}  {sub.n_sent.median():>11.0f}  "
                f"{100 * (sub.span_type == 'multi_sentence').mean():5.1f}  {100 * (sub.span_type == 'whole_sentence').mean():5.1f}  {100 * (sub.span_type == 'sub_sentence').mean():5.1f}")
    out(f"  still unlocated: {pct(int((ds.match == 'none').sum()), len(ds))}")

    rs = np.random.default_rng(42)
    def sample(frame, k):
        return frame.iloc[rs.permutation(len(frame))[:k]]
    out("\n[H] Eyeball samples  ( ¦ = sentence boundary, [[ ]] = located span )")
    out("-- multi-sentence spans --")
    for _, r in sample(loc[loc.span_type == "multi_sentence"], 6).iterrows():
        out(f"id {r.id} ({r.dominant}): {r.marked}\n")
    out("-- anchored spans with ratio > 1.5 (joined/non-contiguous?) --")
    for _, r in sample(loc[(loc.match == "anchor") & (loc.anchor_ratio > 1.5)], 4).iterrows():
        out(f"id {r.id} ({r.dominant}, ratio {r.anchor_ratio:.2f}): {r.marked}\n")
    out("-- unlocated spans (match = none) --")
    for _, r in sample(ds[ds.match == "none"], 6).iterrows():
        out(f"id {r.id}: SPAN={r.span[:160]!r}\n   TEXT={' '.join(r.text.split())[:240]}\n")
    out("-- texts with most sentences (check for over-splitting) --")
    for _, r in loc.nlargest(3, "n_sent").iterrows():
        out(f"id {r.id} (n_sent={r.n_sent}): {r.marked}\n")

    OUT_DIR.mkdir(exist_ok=True)
    d.drop(columns=["marked"], errors="ignore").to_csv(OUT_DIR / "span_check_rows.csv", index=False, encoding="utf-8")
    (OUT_DIR / "span_check_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nWrote {OUT_DIR / 'span_check_summary.txt'} and {OUT_DIR / 'span_check_rows.csv'}")


if __name__ == "__main__":
    main()