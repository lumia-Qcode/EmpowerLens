"""
T12 - build sentence-level labels from 'Distorted part' spans.

Run from repo root:  python -m src.build_sentence_labels      (put this file in src/)
Reads Annotated_data.csv only. Reuses T11's split_sentences / locate_span so the
labels are exactly the ones T11's numbers describe.

Labelling rule (decided in T11):
  exact / normalized / fuzzy rows : sentence = 1 if it shares >= 1 char with the located span
  anchor rows (edited/joined quote): sentence = 1 only if its full text appears inside the quote
                                     (if none do, fall back to overlap and flag the row)
  unlocated rows (match = none)    : excluded, listed in excluded_rows.csv
  no_distortion rows (no span)     : excluded from the selector dataset, listed in excluded_rows.csv

Outputs (results/):
  sentence_labels.csv   one row per sentence: id, sent_idx, n_sent, sent_start, sent_end, sentence,
                        label, match, label_rule, dominant, has_secondary
  excluded_rows.csv     id, reason
  t12_label_report.md   counts to paste into the report
Splitting into train/val/test is NOT done here: join on `id` to your existing row-level split
so all sentences of a text stay in the same split.
"""
from pathlib import Path

import pandas as pd

from src.data import load_raw, LABEL_CANON, TEXT_COL, DOMINANT_COL, SECONDARY_COL, ID_COL
from src.span_check import split_sentences, locate_span, norm_map, SPAN_COL

OUT_DIR = Path("results")
splitter = split_sentences      # swap for a PySBD wrapper returning (start, end) offsets to test robustness


def label_row(text, span):
    """Return (sentences, labels, match_kind, rule) or None if the span is unlocated."""
    s, e, kind, _ = locate_span(text, span)
    if s is None:
        return None
    sents = splitter(text)
    overlap = [i for i, (a, b) in enumerate(sents) if min(b, e) - max(a, s) > 0]
    if kind == "anchor":
        pn = norm_map(str(span))[0].strip()
        gold = [i for i in overlap if norm_map(text[sents[i][0]:sents[i][1]])[0].strip() in pn]
        rule = "containment"
        if not gold:
            gold, rule = overlap, "containment_fallback_overlap"
    else:
        gold, rule = overlap, "overlap"
    labels = [int(i in set(gold)) for i in range(len(sents))]
    return sents, labels, kind, rule


def main():
    df = load_raw("Annotated_data.csv")
    sent_rows, excluded, row_meta = [], [], []
    for _, r in df.iterrows():
        text, rid = r[TEXT_COL], r[ID_COL]
        dom = LABEL_CANON[str(r[DOMINANT_COL]).strip()]
        span = r.get(SPAN_COL)
        has_span = pd.notna(span) and str(span).strip() != ""
        if dom == "no_distortion" or not has_span:
            excluded.append((rid, "no_distortion" if dom == "no_distortion" else "distorted_but_no_span"))
            continue
        res = label_row(text, span)
        if res is None:
            excluded.append((rid, "span_unlocated"))
            continue
        sents, labels, kind, rule = res
        row_meta.append(dict(id=rid, match=kind, rule=rule, n_sent=len(sents), n_pos=sum(labels)))
        for i, ((a, b), y) in enumerate(zip(sents, labels)):
            sent_rows.append(dict(id=rid, sent_idx=i, n_sent=len(sents), sent_start=a, sent_end=b,
                                  sentence=text[a:b], label=y, match=kind, label_rule=rule,
                                  dominant=dom, has_secondary=pd.notna(r[SECONDARY_COL])))

    sl, ex, rm = pd.DataFrame(sent_rows), pd.DataFrame(excluded, columns=["id", "reason"]), pd.DataFrame(row_meta)
    assert (rm.n_pos >= 1).all(), "every kept row must have at least one positive sentence"

    OUT_DIR.mkdir(exist_ok=True)
    sl.to_csv(OUT_DIR / "sentence_labels.csv", index=False, encoding="utf-8")
    ex.to_csv(OUT_DIR / "excluded_rows.csv", index=False, encoding="utf-8")

    n_dist = len(rm) + int((ex.reason != "no_distortion").sum())
    L = ["# T12 sentence-label report", "",
         f"Rows in corpus: {len(df)} | distorted rows with a span: {n_dist} | kept for selector: {len(rm)}",
         f"Sentences: {len(sl)} (positive {int(sl.label.sum())}, {100 * sl.label.mean():.1f}%)", "",
         "## How spans were located (distorted rows)"]
    for k in ("exact", "normalized", "anchor", "fuzzy"):
        L.append(f"- {k}: {int((rm.match == k).sum())} ({100 * (rm.match == k).sum() / n_dist:.1f}%)")
    nun = int((ex.reason == "span_unlocated").sum())
    L += [f"- none (excluded): {nun} ({100 * nun / n_dist:.1f}%)", "",
          "## Positives per row",
          f"- 1: {int((rm.n_pos == 1).sum())} | 2: {int((rm.n_pos == 2).sum())} | 3+: {int((rm.n_pos >= 3).sum())}",
          f"- anchor rows that fell back to overlap (no sentence fully contained): {int((rm.rule == 'containment_fallback_overlap').sum())}",
          f"- excluded no_distortion rows: {int((ex.reason == 'no_distortion').sum())}",
          "", "## Rule", "Overlap (>=1 char) for exact/normalized/fuzzy; containment for anchor rows; "
          "unlocated and no_distortion rows excluded. Sentence splitter: T11 regex (src/span_check.split_sentences)."]
    (OUT_DIR / "t12_label_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()