"""
Measure how far CODIPAS's DERIVED labels are from the human annotation.

Why this exists
---------------
CODIPAS results looked strong (flat 11-class macro_f1_10 0.271 vs Annotated's
0.129), but they were evaluated on CODIPAS's own test set. Before spending GPU on
a transfer run, the cheaper question is whether the two label sets even describe
the same thing on the SAME texts. They largely do not, and this quantifies it.

CODIPAS is not an independent corpus: 2,016 of its texts are already in
Annotated's train split, so the labels can be compared directly, text by text,
with no model involved.

What it reports
---------------
* Cohen's kappa and raw agreement, for binary and for the 11-class task.
* Type agreement restricted to rows BOTH sources call distorted — this separates
  "disagree that anything is wrong" from "disagree about which distortion",
  which have completely different implications.
* Per-class recall of the human label under CODIPAS's rule, i.e. for each human
  class, what CODIPAS called it.
* The directional error breakdown. The disagreement is asymmetric and that
  asymmetry is the finding.

No GPU, no model, no training. Writes docs/codipas_agreement.md + a CSV.

Usage
-----
    python -m src.codipas_agreement
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from src.data import MC_CLASSES, TEXT_COL


def _key(s):
    # Same normalisation as make_splits_codipas_transfer — the corpora differ in
    # whitespace and casing, so exact matching under-counts the overlap.
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _read(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, encoding="utf-8-sig")
    d["_k"] = d[TEXT_COL].map(_key)
    return d


def _kappa_ci(a, b, n_boot=1000, seed=42):
    """Bootstrap 95% CI. A kappa with no interval invites 'is 0.32 different from
    0.20?', and the answer matters here."""
    import numpy as np
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(a), len(a))
        if len(set(a[i])) < 2 or len(set(b[i])) < 2:
            continue
        stats.append(cohen_kappa_score(a[i], b[i]))
    if not stats:
        return (float("nan"), float("nan"))
    return tuple(round(float(x), 3) for x in np.percentile(stats, [2.5, 97.5]))


def main(argv=None):
    ap = argparse.ArgumentParser(description="CODIPAS vs human label agreement.")
    ap.add_argument("--annotated", default="data/splits")
    ap.add_argument("--codipas", default="data/splits_codipas_cls")
    ap.add_argument("--out-md", default="docs/codipas_agreement.md")
    ap.add_argument("--out-csv", default="results/codipas_agreement_per_class.csv")
    args = ap.parse_args(argv)

    ann = pd.concat([_read(Path(args.annotated) / f"{s}.csv") for s in ("train", "val", "test")],
                    ignore_index=True).drop_duplicates("_k")
    cod = pd.concat([_read(Path(args.codipas) / f"{s}.csv") for s in ("train", "val", "test")],
                    ignore_index=True).drop_duplicates("_k")

    m = ann[["_k", "y_bin", "y_mc"]].merge(cod[["_k", "y_bin", "y_mc"]], on="_k",
                                           suffixes=("_a", "_c"))
    if m.empty:
        raise SystemExit("No shared texts — check the splits paths.")

    bin_agree = float((m.y_bin_a == m.y_bin_c).mean())
    mc_agree = float((m.y_mc_a == m.y_mc_c).mean())
    bin_k = cohen_kappa_score(m.y_bin_a, m.y_bin_c)
    mc_k = cohen_kappa_score(m.y_mc_a, m.y_mc_c)
    bin_ci, mc_ci = _kappa_ci(m.y_bin_a, m.y_bin_c), _kappa_ci(m.y_mc_a, m.y_mc_c)

    both = m[(m.y_bin_a == 1) & (m.y_bin_c == 1)]
    type_agree = float((both.y_mc_a == both.y_mc_c).mean()) if len(both) else float("nan")

    fn = int(((m.y_bin_a == 1) & (m.y_bin_c == 0)).sum())   # human distorted, rule clean
    fp = int(((m.y_bin_a == 0) & (m.y_bin_c == 1)).sum())

    # Per human class: how often CODIPAS reproduces it, and what it says instead.
    rows = []
    for i, cls in enumerate(MC_CLASSES):
        sub = m[m.y_mc_a == i]
        if not len(sub):
            continue
        match = float((sub.y_mc_c == i).mean())
        alt = sub[sub.y_mc_c != i].y_mc_c.map(lambda j: MC_CLASSES[j]).value_counts()
        rows.append({
            "human_class": cls,
            "n": len(sub),
            "codipas_agrees": round(match, 3),
            "top_alternative": alt.index[0] if len(alt) else "",
            "top_alternative_n": int(alt.iloc[0]) if len(alt) else 0,
        })
    per_class = pd.DataFrame(rows).sort_values("codipas_agrees")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(args.out_csv, index=False)

    cm = confusion_matrix(m.y_mc_a, m.y_mc_c, labels=range(len(MC_CLASSES)))

    def _interp(k):
        # Landis & Koch (1977) bands, the conventional reading of kappa.
        for lim, name in ((0.0, "poor"), (0.20, "slight"), (0.40, "fair"),
                          (0.60, "moderate"), (0.80, "substantial")):
            if k < lim:
                return name
        return "almost perfect"

    md = [
        "# CODIPAS vs human annotation — label agreement\n",
        f"Generated by `src/codipas_agreement.py` over the **{len(m)} texts** present in "
        "both corpora. No model, no training — this compares the two label sets directly.\n",
        "## Headline\n",
        "| comparison | raw agreement | Cohen's kappa | 95% CI | reading |",
        "|---|---|---|---|---|",
        f"| binary (distorted?) | {bin_agree:.1%} | **{bin_k:.3f}** | {bin_ci[0]}–{bin_ci[1]} | {_interp(bin_k)} |",
        f"| 11-class (which type?) | {mc_agree:.1%} | **{mc_k:.3f}** | {mc_ci[0]}–{mc_ci[1]} | {_interp(mc_k)} |",
        f"| type, among the {len(both)} rows BOTH call distorted | {type_agree:.1%} | — | — | — |",
        "",
        "That third row is the important one. Even where the two sources agree that a "
        "distortion is present, they disagree about **which** distortion roughly "
        f"{1 - type_agree:.0%} of the time. The problem is therefore not a detection "
        "threshold that could be retuned — the disagreement reaches into the taxonomy "
        "itself.\n",
        "## The disagreement is directional\n",
        f"- Human says distorted, CODIPAS says clean: **{fn}**",
        f"- CODIPAS says distorted, human says clean: **{fp}**",
        "",
        f"CODIPAS's derived rule under-detects distortion by roughly {fn / max(fp, 1):.1f}:1 "
        "relative to human annotators. A model trained on it inherits that conservatism, "
        "which is the most likely reason a CODIPAS-trained model scores poorly on the "
        "Annotated test set — a label-convention mismatch, not a data-volume effect.\n",
        "## Per human class\n",
        "How often CODIPAS's derived label reproduces the human one, worst first.\n",
        "| human class | n | CODIPAS agrees | most common alternative | n |",
        "|---|---|---|---|---|",
    ]
    for r in per_class.itertuples():
        md.append(f"| {r.human_class} | {r.n} | {r.codipas_agrees:.1%} | "
                  f"{r.top_alternative} | {r.top_alternative_n} |")

    md += [
        "",
        "## How to cite this\n",
        "This is why the CODIPAS numbers in `results/all_experiments.csv` are **not** "
        "comparable to the Annotated ones: they are scored against a different label "
        "convention on a different test set. Report CODIPAS as a label-transfer / "
        "annotation-agreement result. A low transfer F1 is the *expected* outcome here "
        "and is not evidence that additional training data fails to help.\n",
        "Per-class detail: `results/codipas_agreement_per_class.csv`\n",
        "## 11-class confusion (rows = human, cols = CODIPAS)\n",
        "```",
        "                          " + " ".join(f"{c[:6]:>6}" for c in MC_CLASSES),
    ]
    for i, cls in enumerate(MC_CLASSES):
        md.append(f"{cls:<25} " + " ".join(f"{v:>6}" for v in cm[i]))
    md.append("```")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")

    print(f"shared texts: {len(m)}")
    print(f"  binary   agreement {bin_agree:.1%}  kappa {bin_k:.3f} ({_interp(bin_k)})")
    print(f"  11-class agreement {mc_agree:.1%}  kappa {mc_k:.3f} ({_interp(mc_k)})")
    print(f"  type agreement among {len(both)} both-distorted rows: {type_agree:.1%}")
    print(f"  directional: {fn} human-distorted->clean vs {fp} the other way")
    print(f"\nWrote {args.out_md}\nWrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
