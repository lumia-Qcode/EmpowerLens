"""
D07 - inter-annotator agreement tables on the 60 overlap rows.

Run from repo root:  python reannotation_2026-09-27/scripts/agreement_tables.py
Reads  outputs/relabel/merged_labels.csv   (from relabel_agreement.py)
Writes outputs/relabel/agreement_tables.csv and agreement_tables.md

Comparisons: all four annotators; psych only (Laiba, Reesha, Hurema); each of the 6 pairs.
Levels: type (11 labels, calibrated pessimism counted as No Distortion) and binary (distorted or not).
Rule for blanks: for each comparison, only overlap rows where EVERY annotator in that comparison
gave a label are used (complete cases); the count used is in `n_rows`.
Raw % agreement: for pairs, share of rows with identical labels; for groups, the mean over all
annotator pairs of that share ("pairwise"), and `pct_all_agree` = rows where everyone agrees.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

OUT = Path(__file__).resolve().parent.parent / "outputs" / "relabel"
NAMES = {1: "Laiba", 2: "Reesha", 3: "Hurema", 4: "CS majors"}
PSYCH = [1, 2, 3]
SEED, N_BOOT = 42, 1000


def fleiss_kappa(wide):
    """wide: items x raters, every cell a label (complete cases only)."""
    arr = wide.to_numpy()
    cats = np.unique(arr)
    table = (arr[:, :, None] == cats[None, None, :]).sum(axis=1).astype(float)
    n = table.sum(axis=1)[0]
    p_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_j = table.sum(axis=0) / table.sum()
    p_e = (p_j ** 2).sum()
    return float("nan") if np.isclose(p_e, 1) else (p_i.mean() - p_e) / (1 - p_e)


def pct_pairwise(wide):
    cols = list(wide.columns)
    return float(np.mean([(wide[a] == wide[b]).mean() for a, b in combinations(cols, 2)]))


def kappa(wide):
    """Cohen for 2 columns, Fleiss for 3+."""
    if wide.shape[1] == 2:
        a, b = wide.columns
        if wide[a].nunique() == 1 and wide[b].nunique() == 1 and wide[a].iloc[0] == wide[b].iloc[0]:
            return float("nan")
        return float(cohen_kappa_score(wide[a], wide[b]))
    return float(fleiss_kappa(wide))


def boot_ci(wide):
    rng = np.random.default_rng(SEED)
    ks = [kappa(wide.iloc[rng.integers(0, len(wide), len(wide))]) for _ in range(N_BOOT)]
    return tuple(float(x) for x in np.nanpercentile(ks, [2.5, 97.5]))


def main():
    m = pd.read_csv(OUT / "merged_labels.csv", encoding="utf-8")
    ov = m[m["section"] == "overlap"]
    n_items = ov["Row ID"].nunique()
    comps = [("all four annotators", [1, 2, 3, 4]), ("psychology only (Laiba, Reesha, Hurema)", PSYCH)]
    comps += [(f"{NAMES[a]} vs {NAMES[b]}", [a, b]) for a, b in combinations([1, 2, 3, 4], 2)]
    levels = {"type (11 labels, CP = No Distortion)": "primary_collapsed",
              "binary (distorted or not)": "primary_binary"}
    rows = []
    for cname, ann in comps:
        for lname, col in levels.items():
            wide = ov.pivot_table(index="Row ID", columns="annotator", values=col, aggfunc="first")[ann].dropna()
            k = kappa(wide); lo, hi = boot_ci(wide)
            rows.append({"comparison": cname, "label level": lname,
                         "kappa type": "Fleiss" if len(ann) > 2 else "Cohen",
                         "kappa value": round(k, 3), "ci95_low": round(lo, 3), "ci95_high": round(hi, 3),
                         "raw % agreement": round(100 * pct_pairwise(wide), 1),
                         "raw % all agree": round(100 * (wide.nunique(axis=1) == 1).mean(), 1),
                         "number of rows": len(wide), "rows dropped (blank label)": n_items - len(wide)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "agreement_tables.csv", index=False, encoding="utf-8")

    # who left overlap rows without a label (blank cell or row missing from the workbook)
    labelled = ov[ov["Primary"].notna()].groupby("Row ID")["annotator"].apply(set)
    gaps = {r: sorted({1, 2, 3, 4} - labelled.get(r, set())) for r in ov["Row ID"].unique()}
    gaps = {r: g for r, g in gaps.items() if g}
    missing_rows = list(gaps)
    blank_txt = "; ".join(f"row {r}: " + ", ".join(NAMES[a] for a in g) for r, g in gaps.items())
    md = ["# D07 - Inter-annotator agreement on the 60 overlap rows", "",
          f"Source: `merged_labels.csv`, overlap section ({n_items} rows, 4 annotators). "
          "Type = 11 labels with *No Distortion (calibrated pessimism)* counted as No Distortion; binary = distorted or not.", "",
          f"**Blank labels:** {len(missing_rows)} of {n_items} overlap rows were left without a label by at least one annotator "
          f"({blank_txt}). In each comparison only rows where every annotator in it has a label are used, so "
          "the 'all four' and 'psychology only' groups use fewer rows than a pair that avoids the blank annotator "
          "(`number of rows` column). The CS-majors blanks filled from the original label (81 rows) are all outside the overlap set.", "",
          "Kappa subtracts the agreement expected by chance, so it is always lower than raw % agreement on the same labels. "
          "Raw % agreement is the mean over annotator pairs of the share of rows with identical labels (the 'raw % all agree' column "
          "is the stricter share where everyone agrees). The original paper reports 33.7% (type) and 61% (binary); compare against the "
          "raw % columns, not the kappas. 95% CIs are bootstrap over rows (1,000 resamples, seed 42).", "",
          t.to_markdown(index=False), ""]
    (OUT / "agreement_tables.md").write_text("\n".join(md), encoding="utf-8")
    print(t.to_string(index=False)); print("blanks:", blank_txt)


if __name__ == "__main__":
    main()
