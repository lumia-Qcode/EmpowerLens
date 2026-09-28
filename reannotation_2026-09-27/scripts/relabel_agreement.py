"""
Merge returned annotator workbooks and compute inter-annotator agreement.

Task 4 of RELABEL_SPLIT_TASK.md — run this *after* the labelling session:

    venv\\Scripts\\python.exe reannotation_2026-09-27/scripts/relabel_agreement.py

Inputs : outputs/relabel/annotator_*.xlsx (the returned workbooks)
         outputs/relabel/annotator_master_key.xlsx
Outputs: outputs/relabel/merged_labels.csv
         outputs/relabel/unsure_for_adjudication.csv
         outputs/relabel/agreement_report.md

Agreement is computed on the overlap set only; sections 2 and 3 are
single-annotated. Fleiss' kappa is implemented here rather than imported, so the
script needs no statsmodels.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics import cohen_kappa_score

SEED = 42
N_BOOTSTRAP = 1000
CP_LABEL = "No Distortion (calibrated pessimism)"
NO_DISTORTION = "No Distortion"

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs" / "relabel"

DISTORTED_NEGATIVES = {NO_DISTORTION, CP_LABEL}


# --------------------------------------------------------------------------
def read_workbooks() -> pd.DataFrame:
    rows, invalid = [], []
    for path in sorted(OUT.glob("annotator_*.xlsx")):
        if path.name == "annotator_master_key.xlsx":
            continue
        m = re.match(r"annotator_(\d+)", path.name)
        annotator = int(m.group(1)) if m else -1
        ws = load_workbook(path, data_only=True).worksheets[0]
        header = [c.value for c in ws[1]]
        idx = {name: i for i, name in enumerate(header)}
        for r in ws.iter_rows(min_row=2, values_only=True):
            if r[idx["Row ID"]] is None:
                continue
            primary = (r[idx["Primary"]] or "").strip() or None
            secondary = (r[idx["Secondary"]] or "").strip() or None
            raw_unsure = r[idx["Unsure"]]
            # Accepts a checkbox (True/False), a typed TRUE/Yes/x, or blank.
            if isinstance(raw_unsure, bool):
                unsure = "Yes" if raw_unsure else None
            else:
                text = str(raw_unsure or "").strip().lower()
                unsure = "Yes" if text in {"true", "yes", "y", "x", "1", "☑"} else None
            note = r[idx["Notes"]]
            reason = None
            if primary is None:
                reason = "Primary is blank"
            elif secondary is not None and secondary == primary:
                reason = "Secondary repeats Primary"
            elif secondary is not None and primary in DISTORTED_NEGATIVES:
                reason = f"Secondary set while Primary is '{primary}'"
            if reason:
                invalid.append({"file": path.name, "Row ID": r[idx["Row ID"]],
                                "annotator": annotator, "reason": reason})
            rows.append({
                "Row ID": int(r[idx["Row ID"]]),
                "annotator": annotator,
                "Primary": primary,
                "Secondary": secondary,
                "Unsure": unsure,
                "Notes": note,
                "invalid_reason": reason,
            })
    if not rows:
        raise SystemExit(f"no annotator workbooks with labels found in {OUT}")
    return pd.DataFrame(rows), pd.DataFrame(invalid)


def fleiss_kappa(table: np.ndarray) -> float:
    """table[i, j] = number of raters assigning item i to category j."""
    n_items, _ = table.shape
    n_raters = table.sum(axis=1)
    if not np.all(n_raters == n_raters[0]):
        # keep only items rated by the modal number of raters
        keep = n_raters == np.bincount(n_raters).argmax()
        table, n_raters = table[keep], n_raters[keep]
        n_items = table.shape[0]
    n = n_raters[0]
    if n < 2 or n_items == 0:
        return float("nan")
    p_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    p_bar = p_i.mean()
    p_j = table.sum(axis=0) / (n_items * n)
    p_e = (p_j ** 2).sum()
    return float("nan") if np.isclose(p_e, 1) else (p_bar - p_e) / (1 - p_e)


def to_table(wide: pd.DataFrame, categories: list[str]) -> np.ndarray:
    cat_idx = {c: i for i, c in enumerate(categories)}
    table = np.zeros((len(wide), len(categories)))
    for i, (_, row) in enumerate(wide.iterrows()):
        for v in row.dropna():
            table[i, cat_idx[v]] += 1
    return table


def view_metrics(wide: pd.DataFrame, discipline: dict[int, str]):
    """wide: one row per item, one column per annotator, values = labels."""
    categories = sorted({v for v in wide.to_numpy().ravel() if isinstance(v, str)})
    table = to_table(wide, categories)

    pairwise_rows = []
    for _, row in wide.iterrows():
        vals = [v for v in row.tolist() if isinstance(v, str)]
        pairs = list(combinations(vals, 2))
        if pairs:
            pairwise_rows.append(sum(a == b for a, b in pairs) / len(pairs))
    pct_pairwise = float(np.mean(pairwise_rows)) if pairwise_rows else float("nan")
    pct_full = float(np.mean([row.dropna().nunique() == 1 for _, row in wide.iterrows()]))

    cohens, by_discipline = {}, {"psych-psych": [], "cs-cs": [], "psych-cs": []}
    for a, b in combinations(wide.columns, 2):
        sub = wide[[a, b]].dropna()
        if len(sub) < 2:
            continue
        k = cohen_kappa_score(sub[a], sub[b])
        cohens[f"{a}-{b}"] = k
        da, db = discipline.get(a, "?"), discipline.get(b, "?")
        key = "psych-psych" if da == db == "psych" else (
            "cs-cs" if da == db == "cs" else "psych-cs")
        by_discipline[key].append(k)

    return {
        "n_items": len(wide),
        "categories": categories,
        "pct_pairwise": pct_pairwise,
        "pct_full": pct_full,
        "fleiss": fleiss_kappa(table),
        "cohens": cohens,
        "cohen_min": min(cohens.values()) if cohens else float("nan"),
        "cohen_max": max(cohens.values()) if cohens else float("nan"),
        "lights": float(np.mean(list(cohens.values()))) if cohens else float("nan"),
        "by_discipline": {k: (float(np.mean(v)) if v else float("nan"))
                          for k, v in by_discipline.items()},
        "label_distribution": pd.Series(
            [v for v in wide.to_numpy().ravel() if isinstance(v, str)]
        ).value_counts().to_dict(),
    }


def bootstrap(wide: pd.DataFrame, discipline: dict[int, str]):
    rng = np.random.default_rng(SEED)
    stats = {"pct_pairwise": [], "fleiss": [], "lights": []}
    n = len(wide)
    for _ in range(N_BOOTSTRAP):
        sample = wide.iloc[rng.integers(0, n, n)]
        m = view_metrics(sample, discipline)
        for k in stats:
            stats[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
            for k, v in stats.items()}


def fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def main() -> int:
    labels, invalid = read_workbooks()
    key = pd.read_excel(OUT / "annotator_master_key.xlsx")
    key_cols = [c for c in ["Row ID", "annotator", "annotator_name", "discipline",
                            "section", "split", "bucket", "original_dominant",
                            "original_secondary"] if c in key.columns]
    merged = labels.merge(key[key_cols], on=["Row ID", "annotator"], how="left")

    merged["primary_collapsed"] = merged["Primary"].replace({CP_LABEL: NO_DISTORTION})
    merged["primary_binary"] = np.where(
        merged["primary_collapsed"].isin([NO_DISTORTION]), "none", "distorted")
    merged.loc[merged["Primary"].isna(), "primary_binary"] = np.nan

    merged.to_csv(OUT / "merged_labels.csv", index=False, encoding="utf-8")
    unsure = merged[merged["Unsure"].notna()]
    unsure.to_csv(OUT / "unsure_for_adjudication.csv", index=False, encoding="utf-8")

    discipline = (key.drop_duplicates("annotator")
                  .set_index("annotator")["discipline"].to_dict()
                  if "discipline" in key.columns else {})

    overlap = merged[merged["section"] == "overlap"]
    lines = [
        "# Inter-annotator agreement (overlap set)",
        "",
        f"- Rows merged: **{len(merged)}**",
        f"- Overlap rows: **{overlap['Row ID'].nunique()}** "
        f"x {overlap['annotator'].nunique()} annotators",
        f"- Invalid rows (not dropped): **{len(invalid)}**",
        f"- Unsure rows flagged for adjudication: **{len(unsure)}**",
        "",
    ]
    if len(invalid):
        lines += ["## Invalid rows", "", invalid.to_markdown(index=False), ""]

    views = {
        "12-way (with CP_LABEL)": "Primary",
        "11-way (CP collapsed)": "primary_collapsed",
        "binary": "primary_binary",
    }
    for view_name, col in views.items():
        wide = overlap.pivot_table(index="Row ID", columns="annotator",
                                   values=col, aggfunc="first")
        if wide.empty:
            continue
        m = view_metrics(wide, discipline)
        ci = bootstrap(wide, discipline)
        lines += [
            f"## {view_name}",
            "",
            f"- Items: {m['n_items']}, categories used: {len(m['categories'])}",
            "",
            "| Metric | Value | 95% CI |",
            "| --- | --- | --- |",
            f"| Percent agreement (pairwise) | {fmt(m['pct_pairwise'])} | "
            f"[{fmt(ci['pct_pairwise'][0])}, {fmt(ci['pct_pairwise'][1])}] |",
            f"| Percent agreement (full) | {fmt(m['pct_full'])} | — |",
            f"| Fleiss' kappa | {fmt(m['fleiss'])} | "
            f"[{fmt(ci['fleiss'][0])}, {fmt(ci['fleiss'][1])}] |",
            f"| Light's kappa | {fmt(m['lights'])} | "
            f"[{fmt(ci['lights'][0])}, {fmt(ci['lights'][1])}] |",
            f"| Cohen's kappa range | {fmt(m['cohen_min'])} – {fmt(m['cohen_max'])} | — |",
            "",
            "**Pairwise Cohen's kappa:** "
            + ", ".join(f"{k}: {fmt(v)}" for k, v in m["cohens"].items()),
            "",
            "**By discipline:** "
            + ", ".join(f"{k}: {fmt(v)}" for k, v in m["by_discipline"].items()),
            "",
            f"**Label distribution:** {m['label_distribution']}",
            "",
        ]

    # Rows originally labelled distorted that the majority now calls calibrated pessimism.
    cp_rows = []
    for rid, grp in merged[merged["Primary"].notna()].groupby("Row ID"):
        orig = str(grp["original_dominant"].iloc[0]).strip()
        if orig and orig != NO_DISTORTION:
            votes = (grp["Primary"] == CP_LABEL).sum()
            if votes > len(grp) / 2:
                cp_rows.append(rid)
    lines += [
        "## Calibrated pessimism",
        "",
        f"Rows originally labelled as a distortion that the majority marked "
        f"`{CP_LABEL}`: **{len(cp_rows)}**",
        "",
        f"IDs: {cp_rows}",
        "",
    ]

    (OUT / "agreement_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"merged rows        : {len(merged)}")
    print(f"invalid rows       : {len(invalid)}")
    print(f"unsure rows        : {len(unsure)}")
    print(f"overlap items      : {overlap['Row ID'].nunique()}")
    print(f"wrote              : {OUT / 'agreement_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
