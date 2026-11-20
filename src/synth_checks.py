"""S07: synthetic leak + overlap checks.

1. Keyword check. A fixed word list per distortion type labels each row with
   whichever type's words appear most (no model, no training). Its macro-F1
   against the critic panel's labels on the synthetic rows is compared with its
   macro-F1 against the gold labels on the benchmark val split. If the list does
   much better on synthetic than on real text, the generator is writing the
   give-away words into every example and a model would learn the words, not the
   reasoning.

2. Overlap check. Word-Jaccard (src/synth/dedup.jaccard) between every synthetic
   row and four sets: the generator seeds, the supplementary-evaluation rows,
   the survey responses (optional, --survey; participant data, never committed)
   and the benchmark test split. Pairs above 0.6 are flagged and the flagged
   synthetic rows are dropped from a clean copy.

   Documented exception: this script reads the TEXT of data/splits/test.csv for
   the overlap check only (CLAUDE.md otherwise reserves test.csv for
   src/evaluate.py). It never reads test labels and never selects a model or a
   threshold; it can only remove synthetic training rows that copy a test row.

    venv\\Scripts\\python.exe -m src.synth_checks
    venv\\Scripts\\python.exe -m src.synth_checks --survey D3/survey_responses.csv
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.data import DOMINANT_COL, LABEL_CANON, MC_CLASSES, TEXT_COL, _canon
from src.synth.dedup import jaccard

SYN = Path("data/synthetic/synthetic_train_en.jsonl")
CLEAN = Path("data/synthetic/synthetic_train_en_clean.jsonl")
SEEDS = Path("data/synthetic_gen/seeds/entrepreneurial_seeds.jsonl")
SUPP = Path("data/supplementary/supplementary_labelled.csv")
VAL = Path("data/splits/val.csv")
TEST = Path("data/splits/test.csv")
OVERLAP_CSV = Path("data/synthetic/overlap_flags.csv")
OUT = Path("results/results_synth_checks")
THRESHOLD = 0.6
SEED = 42

# A-priori give-away words per type, written before looking at the S06 rows:
# the brief's examples plus the lexical markers of FYP_PLAN section 3.5
# (MARKER_LEXICON in src/cluster_label_errors.py). Order breaks ties.
KEYWORDS: dict[str, list[str]] = {
    "should_statements": ["should", "shouldn't", "should have", "must", "have to", "has to", "ought",
                          "supposed to", "need to"],
    "all_or_nothing": ["always", "never", "completely", "totally", "entirely", "total failure", "perfect",
                       "nothing", "everything", "all or nothing"],
    "mind_reading": ["they think", "he thinks", "she thinks", "must think", "they must", "thinks i",
                     "think i'm", "think i am", "judging me", "see me as", "they'll think"],
    "fortune_telling": ["will never", "never going to", "going to fail", "will fail", "won't work",
                        "doomed", "it's over", "probably won't", "will be a disaster"],
    "overgeneralization": ["every time", "everyone", "nobody", "no one", "all of them", "keeps happening",
                           "again and again", "this always"],
    "labeling": ["i'm a failure", "i am a failure", "i'm such a", "loser", "idiot", "stupid", "worthless",
                 "useless", "fraud", "failure"],
    "personalization": ["my fault", "because of me", "blame myself", "on me", "i caused", "i let",
                        "i'm responsible", "i am responsible"],
    "emotional_reasoning": ["i feel", "feel like", "feels like", "i felt", "my gut", "feeling"],
    "magnification": ["disaster", "catastrophe", "ruined", "end of everything", "the end", "collapse",
                      "terrible", "awful", "worst", "huge"],
    "mental_filter": ["only", "just one", "all i", "focus on", "can't see past", "can't stop thinking"],
}

# Post-hoc additions: phrases seen in the S06 output (docs/synth_run.md). Fitted
# to what we saw, so they're a supplementary result, never the headline.
POST_HOC: dict[str, list[str]] = {
    "mental_filter": ["fixat", "ignor", "replay", "only thing that matters", "couldn't stop thinking",
                      "drown out", "all i could see"],
    "labeling": ["not cut out"],
    "fortune_telling": ["fall apart", "never catch up", "already over"],
    "magnification": ["homeless", "collapsing", "lost our chance"],
    "emotional_reasoning": ["my gut said", "that feeling is"],
}


def _norm(text: str) -> str:
    return str(text).lower().replace("’", "'").replace("‘", "'")


def _patterns(lists: dict[str, list[str]]):
    # Word boundary at the start; stems like "fixat" / "ignor" may continue.
    return {t: [re.compile(r"(?<![a-z'])" + re.escape(w) + (r"" if w in ("fixat", "ignor") else r"(?![a-z])"))
                for w in ws] for t, ws in lists.items()}


def merge_lists(a, b):
    out = {t: list(ws) for t, ws in a.items()}
    for t, ws in b.items():
        out.setdefault(t, [])
        out[t] += [w for w in ws if w not in out[t]]
    return out


def keyword_hits(text: str, pats) -> dict[str, int]:
    s = _norm(text)
    return {t: sum(len(p.findall(s)) for p in ps) for t, ps in pats.items()}


def keyword_label(text: str, pats) -> str:
    hits = keyword_hits(text, pats)
    best = max(hits.values(), default=0)
    if best == 0:
        return "no_distortion"
    return next(t for t, h in hits.items() if h == best)      # dict order breaks ties


def macro_scores(y_true, y_pred) -> dict:
    return {
        "macro_f1": round(f1_score(y_true, y_pred, labels=MC_CLASSES, average="macro", zero_division=0), 4),
        "macro_f1_10": round(f1_score(y_true, y_pred, labels=MC_CLASSES[1:], average="macro", zero_division=0), 4),
    }


def chance_scores(y_true, n=100, seed=SEED) -> dict:
    """Stratified-random reference: the true labels shuffled (same class mix), averaged."""
    rng = np.random.default_rng(seed)
    y = np.array(y_true)
    s = [macro_scores(y, rng.permutation(y)) for _ in range(n)]
    return {k: round(float(np.mean([x[k] for x in s])), 4) for k in s[0]}


def openers(texts) -> dict:
    first = Counter(_norm(t).split()[0] for t in texts if str(t).split())
    two = Counter(" ".join(_norm(t).split()[:2]) for t in texts if len(str(t).split()) >= 2)
    n = len(texts)
    return {"n": n, "distinct_first_words": len(first), "distinct_two_word_starts": len(two),
            "top_two_word_starts": two.most_common(5),
            "share_top3_two_word_starts": round(sum(c for _, c in two.most_common(3)) / n, 3) if n else 0}


def overlaps(syn: list[dict], sets: dict[str, list[tuple[str, str]]], threshold=THRESHOLD) -> pd.DataFrame:
    rows = []
    for r in syn:
        for name, items in sets.items():
            for oid, text in items:
                j = jaccard(r["text"], text)
                if j > threshold:
                    rows.append({"syn_id": r["id"], "set": name, "other_id": oid, "jaccard": round(j, 3)})
    return pd.DataFrame(rows, columns=["syn_id", "set", "other_id", "jaccard"])


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_survey(path: Path, min_words=10) -> list[tuple[str, str]]:
    """Free-text answers from a Google Form export: every cell of the answer columns.

    Answer columns are those whose median answer is at least 5 words (drops
    timestamp / name / age / gender columns). Answers under 10 words are skipped
    (the D04 rule). Only anonymous ids ("survey_r<row>_q<col>") leave this function.
    """
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):          # Google Forms "Download -> .xlsx"
        df = pd.read_excel(path, dtype=str)                # first sheet = the form responses
    else:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    out = []
    for ci, col in enumerate(df.columns):
        wc = df[col].dropna().str.split().str.len()
        if wc.empty or wc.median() < 5:
            continue
        for ri, v in df[col].items():
            if isinstance(v, str) and len(v.split()) >= min_words:
                out.append((f"survey_r{ri + 2}_q{ci + 1}", v))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S07 synthetic keyword-leak and overlap checks")
    ap.add_argument("--input", default=str(SYN))
    ap.add_argument("--survey", default=None, help="CSV export of the survey (local only, never committed)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args(argv)

    syn = load_jsonl(Path(args.input))
    syn_texts = [r["text"] for r in syn]
    syn_y = [_canon(r["dominant_distortion"]) for r in syn]
    val = pd.read_csv(VAL, encoding="utf-8-sig")
    val_texts = val[TEXT_COL].astype(str).tolist()
    val_y = [_canon(x) for x in val[DOMINANT_COL]]

    # ---- 1. keyword check ---------------------------------------------------- #
    results, per_type = [], []
    for name, lists in (("a_priori", KEYWORDS), ("post_hoc", merge_lists(KEYWORDS, POST_HOC))):
        pats = _patterns(lists)
        for split, texts, y in (("synthetic", syn_texts, syn_y), ("benchmark_val", val_texts, val_y)):
            pred = [keyword_label(t, pats) for t in texts]
            results.append({"list": name, "split": split, "n": len(texts), **macro_scores(y, pred),
                            **{f"chance_{k}": v for k, v in chance_scores(y).items()}})
            if name == "a_priori":
                for t in KEYWORDS:
                    idx = [i for i, lab in enumerate(y) if lab == t]
                    hit = [keyword_hits(texts[i], pats)[t] > 0 for i in idx]
                    per_type.append({"split": split, "type": t, "n": len(idx),
                                     "rows_with_own_keyword": round(float(np.mean(hit)), 3) if hit else None})
    kw = pd.DataFrame(results)
    pt = pd.DataFrame(per_type).pivot(index="type", columns="split", values=["n", "rows_with_own_keyword"])
    op = {"synthetic": openers(syn_texts), "benchmark_val": openers(val_texts)}

    # ---- 2. overlap check ---------------------------------------------------- #
    seeds = [(s["id"], s["text"]) for s in load_jsonl(SEEDS)]
    supp = pd.read_csv(SUPP, dtype=str)
    supp_eval = [(r.item_id, r.text) for r in supp[supp.role == "evaluation"].itertuples()]
    test = pd.read_csv(TEST, encoding="utf-8-sig", usecols=["Id_Number", TEXT_COL])   # texts only, see docstring
    test_items = [(str(i), t) for i, t in zip(test["Id_Number"], test[TEXT_COL].astype(str))]
    sets = {"seeds": seeds, "supplementary_evaluation": supp_eval, "test": test_items}
    survey_status = "not checked: no --survey file given (survey data is not in the repo)"
    if args.survey:
        sv = load_survey(Path(args.survey))
        sets["survey"] = sv
        survey_status = f"checked: {len(sv)} answers of 10+ words from {Path(args.survey).name}"
    ov = overlaps(syn, sets, args.threshold)
    flagged = set(ov.syn_id)
    clean = [r for r in syn if r["id"] not in flagged]

    OVERLAP_CSV.parent.mkdir(parents=True, exist_ok=True)
    ov.to_csv(OVERLAP_CSV, index=False, encoding="utf-8")
    with CLEAN.open("w", encoding="utf-8") as fh:
        for r in clean:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    OUT.mkdir(parents=True, exist_ok=True)
    kw.to_csv(OUT / "keyword_scores.csv", index=False)
    pt.to_csv(OUT / "keyword_hits_per_type.csv")
    max_j = {name: round(max((jaccard(r["text"], t) for r in syn for _, t in items), default=0.0), 3)
             for name, items in sets.items()}
    summary = {
        "n_synthetic": len(syn), "threshold": args.threshold,
        "set_sizes": {k: len(v) for k, v in sets.items()},
        "flagged_per_set": {k: sorted(set(ov[ov.set == k].syn_id)) for k in sets},
        "max_jaccard_per_set": max_j,
        "n_flagged_rows": len(flagged), "n_clean": len(clean),
        "clean_per_type": dict(Counter(r["dominant_distortion"] for r in clean).most_common()),
        "survey": survey_status, "openers": op,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(kw.to_string(index=False))
    print("\nrows whose own type's keyword appears (a-priori list):\n", pt.to_string())
    print("\nopeners:", json.dumps(op, indent=1))
    print("\noverlap > %.1f:" % args.threshold, {k: len(v) for k, v in summary["flagged_per_set"].items()},
          "| max Jaccard per set:", max_j)
    print(f"survey: {survey_status}")
    print(f"clean rows: {len(clean)} -> {CLEAN}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
