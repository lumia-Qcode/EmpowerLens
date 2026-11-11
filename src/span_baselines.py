"""
D10 - does the trained span selector (D03) beat simple sentence-picking rules?

RUN (from repo root, no training, reads committed files only):
    venv\\Scripts\\python.exe -m src.span_baselines

Input : results/results_sentence_label/sentence_labels_{val,test}.csv   (gold sentence labels, T12)
        results/results_span_selector/predictions_{val,test}_<seed>.csv (D03 selector output)
        results/results_span_selector/eval_{val,test}_<seed>.json       (cross-check)
Output: results/results_span_selector/baselines.csv and baselines.md

Rules compared: always the FIRST sentence, always the LAST, one RANDOM sentence per row
(numpy default_rng, seed 42). All are scored with D03's own metric code
(src.train_span_selector.score_part): sentence-level precision/recall/F1 over all sentences,
plus row hit rate (share of rows where >= 1 picked sentence is gold).
Extra row: the selector restricted to its top-scoring sentence per row (same 1-pick budget as
the rules). The selector's thresholds were tuned on val per seed (D03); the rules need no tuning.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_span_selector import ID_COL, LABELS_DIR, OUT_DIR, load_part, score_part

SEEDS = (42, 1337, 2024)
RANDOM_SEED = 42
METRICS = ("precision", "recall", "f1", "row_hit_rate")
SELECTOR = "trained selector (mean ± std, 3 seeds)"
TOP1 = "extra: trained selector, top-1 sentence per row"
RULES = ("first sentence", "last sentence", "random sentence (seed 42)")


def pick_first(df):
    return (df["sent_idx"] == 0).astype(int).to_numpy()


def pick_last(df):
    return (df["sent_idx"] == df.groupby(ID_COL)["sent_idx"].transform("max")).astype(int).to_numpy()


def pick_random(df, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    sel = np.zeros(len(df), int)
    for rid in sorted(df[ID_COL].unique()):          # fixed row order -> reproducible draw
        pos = np.flatnonzero(df[ID_COL].to_numpy() == rid)
        sel[rng.choice(pos)] = 1
    return sel


def pick_top1(df, score):
    s = pd.Series(np.asarray(score, float), index=df.index)
    top = s.groupby(df[ID_COL]).idxmax()
    sel = np.zeros(len(df), int)
    sel[df.index.get_indexer(top.values)] = 1
    return sel


def score_sel(df, sel):
    return score_part(df, np.asarray(sel, float), 1.0)


def selector_runs(df, part):
    """Per-seed results for the thresholded selector and its top-1 variant."""
    thr_runs, top_runs = [], []
    for seed in SEEDS:
        pred = pd.read_csv(OUT_DIR / f"predictions_{part}_{seed}.csv", encoding="utf-8")
        merged = df[[ID_COL, "sent_idx"]].merge(pred, on=[ID_COL, "sent_idx"], how="left", validate="1:1")
        if merged["score"].isna().any():
            raise SystemExit(f"predictions_{part}_{seed}.csv does not cover every gold sentence")
        gold_pred = (merged["gold"] == "yes").astype(int).to_numpy()
        if not np.array_equal(gold_pred, df["label"].to_numpy()):
            raise SystemExit(f"gold column in predictions_{part}_{seed}.csv disagrees with the T12 labels")
        res = score_sel(df, (merged["selected"] == "yes").to_numpy())
        ref = json.loads((OUT_DIR / f"eval_{part}_{seed}.json").read_text(encoding="utf-8"))
        for k in METRICS:
            assert abs(res[k] - ref[k]) < 1e-9, f"{part} seed {seed} {k}: {res[k]} != eval json {ref[k]}"
        thr_runs.append(res)
        top_runs.append(score_sel(df, pick_top1(df, merged["score"].to_numpy())))
    return thr_runs, top_runs


def summarise(runs, split, method):
    row = {"split": split, "method": method}
    for k in METRICS:
        vals = np.array([r[k] for r in runs])
        row[k] = float(vals.mean())
        row[f"{k}_std"] = float(vals.std(ddof=1)) if len(runs) > 1 else np.nan
    for k in ("n_rows", "n_sentences", "n_gold_sentences"):
        row[k] = runs[0][k]
    row["n_selected"] = float(np.mean([r["n_selected"] for r in runs]))
    row["n_seeds"] = len(runs)
    return row


def expected_random(df):
    """Analytic expectation of the one-random-sentence rule (not just the seed-42 draw)."""
    g = df.groupby(ID_COL)["label"].agg(["sum", "count"])
    tp = (g["sum"] / g["count"]).sum()          # expected true positives
    p, r = tp / len(g), tp / df["label"].sum()
    hit = (g["sum"] / g["count"]).mean()        # one pick per row: P(hit) = gold share of the row
    return p, r, 2 * p * r / (p + r), float(hit)


def fmt(row, k):
    v, s = row[k], row[f"{k}_std"]
    return f"{v:.3f}" if pd.isna(s) else f"{v:.3f} ± {s:.3f}"


def verdict(rows):
    sel = rows[0]
    rules = [r for r in rows if r["method"] in RULES]
    lines = []
    for k, name in (("f1", "sentence F1"), ("row_hit_rate", "row hit rate")):
        best = max(rules, key=lambda r: r[k])
        margin = sel[k] - best[k]
        beats = margin > sel[f"{k}_std"]
        txt = (f"On {name}, the best rule is **{best['method']}** ({best[k]:.3f}); the selector is "
               f"{abs(margin):.3f} {'above' if margin >= 0 else 'below'} it (selector std {sel[f'{k}_std']:.3f}), "
               f"so it **{'does' if beats else 'does not'}** beat the best rule by more than its std.")
        if best["method"] == "last sentence" and not beats:
            txt += (" The selector barely beats (or does not beat) always taking the last sentence: people tend to "
                    "put the distorted conclusion at the end of their reflection.")
        lines.append(txt)
    top = next(r for r in rows if r["method"] == TOP1)
    best = max(rules, key=lambda r: r["precision"])
    lines.append(
        f"Equal budget (one sentence per row): the selector's top-1 pick is gold in {top['precision']:.3f} ± "
        f"{top['precision_std']:.3f} of rows vs {best['precision']:.3f} for the best rule ({best['method']}), "
        + ("so the advantage is not only an effect of picking more sentences."
           if top["precision"] - best["precision"] > top["precision_std"] else
           "so with an equal budget the selector does not beat the best rule by more than its std."))
    return lines


def main():
    excl = pd.read_csv(LABELS_DIR / "excluded_rows.csv", encoding="utf-8")
    n_unlocated = int((excl["reason"] != "no_distortion").sum())
    all_rows, md_blocks = [], []
    for part in ("val", "test"):
        df = load_part(part)
        thr_runs, top_runs = selector_runs(df, part)
        rows = [summarise(thr_runs, part, SELECTOR)]
        for name, fn in zip(RULES, (pick_first, pick_last, pick_random)):
            sel = fn(df)
            assert sel.sum() == df[ID_COL].nunique(), name
            rows.append(summarise([score_sel(df, sel)], part, name))
        rows.append(summarise(top_runs, part, TOP1))
        all_rows += rows
        ep, er, ef, eh = expected_random(df)
        r0 = rows[0]
        md_blocks += [
            f"## {part}", "",
            f"Scored: **{r0['n_rows']} rows, {r0['n_sentences']:,} sentences** ({r0['n_gold_sentences']} gold, "
            f"{100 * r0['n_gold_sentences'] / r0['n_sentences']:.1f}%).", "",
            "| Method | Precision | Recall | Sentence F1 | Row hit rate | Sentences picked |",
            "| --- | --- | --- | --- | --- | --- |",
            *[f"| {r['method']} | {fmt(r, 'precision')} | {fmt(r, 'recall')} | {fmt(r, 'f1')} | "
              f"{fmt(r, 'row_hit_rate')} | {r['n_selected']:.0f} |" for r in rows],
            "",
            *verdict(rows), "",
            f"Random rule, analytic expectation over all draws: precision {ep:.3f}, recall {er:.3f}, F1 {ef:.3f}, "
            f"row hit rate {eh:.3f} (the seed-42 row above is one draw). \"Last sentence\" is gold in "
            f"{rows[2]['precision']:.3f} of rows, {rows[2]['precision'] - ep:+.3f} vs that chance level, and "
            f"\"first sentence\" {rows[1]['precision'] - ep:+.3f}.", "",
        ]
    out = pd.DataFrame(all_rows)
    cols = ["split", "method"] + [c for k in METRICS for c in (k, f"{k}_std")] + \
           ["n_rows", "n_sentences", "n_gold_sentences", "n_selected", "n_seeds"]
    out[cols].round(4).to_csv(OUT_DIR / "baselines.csv", index=False, encoding="utf-8")

    md = [
        "# D10 - Span selector vs simple baselines", "",
        "Does the trained sentence selector (D03, MentalRoBERTa, one sentence at a time) beat simple rules? "
        "Rules: always pick the **first** sentence, always the **last**, or one **random** sentence per row (seed 42). "
        "All methods are scored with the same code as D03 (`score_part` in `src/train_span_selector.py`) against the "
        "T12 sentence labels in `results/results_sentence_label/`: sentence-level precision, recall and F1 over all "
        "sentences, and **row hit rate** = share of rows where at least one picked sentence is a gold sentence.", "",
        f"**{n_unlocated} rows were excluded** because their quoted distorted part couldn't be found in the text "
        "(`excluded_rows.csv`); No-Distortion rows have no gold sentence and are not scored.", "",
        "The selector is the mean ± std (n−1) over seeds 42, 1337, 2024, each with its own threshold chosen on val; "
        "the rules need no tuning and are run once. The selector picks about 2.6 sentences per row while every rule "
        "picks exactly one, which raises its recall and lowers its precision; the **extra** top-1 row restricts the "
        "selector to its single highest-scoring sentence per row, so it can be compared with the rules on an equal "
        "budget.", "",
        *md_blocks,
        "Source: `baselines.csv` (same numbers, unrounded std). Script: `python -m src.span_baselines`.", "",
    ]
    (OUT_DIR / "baselines.md").write_text("\n".join(md), encoding="utf-8")
    print(out[["split", "method", "precision", "recall", "f1", "f1_std", "row_hit_rate", "n_selected"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
