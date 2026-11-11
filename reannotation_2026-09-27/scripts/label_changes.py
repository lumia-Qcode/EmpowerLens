"""
D08 - errors vs redefinitions, and the label change rate on the test rows.

Run from repo root:  venv\\Scripts\\python.exe reannotation_2026-09-27/scripts/label_changes.py
Reads  outputs/relabel/merged_labels.csv   (from relabel_agreement.py)
       Annotated_data.csv                  (repo root, read only: text for the check sheet)
Writes outputs/relabel/label_changes.csv, label_changes_summary.md, redefinition_check_20.xlsx

Votes: a label counts only if Primary is filled, it was NOT filled in from the original
label (CS-majors blanks, `filled_from_original`), and the annotator did not tick Unsure
(Unsure rows were never adjudicated). Majority = one 11-way label (calibrated pessimism
counted as No Distortion) holding more than half of a row's valid votes. A row "changed"
when that label differs from the original dominant label. Groups (rule on label pairs):
  distortion -> No Distortion  : redefinition if CP votes >= plain ND votes among the ND
                                 voters (tie = redefinition, the cautious choice), else genuine error
  No Distortion -> distortion  : genuine error
  type A -> original secondary : primary swap
  type A -> any other type     : genuine error
"""
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation
from scipy.stats import fisher_exact

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "relabel"
REPO = ROOT.parent
NO_DISTORTION = "No Distortion"
CP_LABEL = "No Distortion (calibrated pessimism)"
GROUPS = ["genuine error", "redefinition", "primary swap"]
SEED, N_CHECK = 42, 20
FLAGGED, UNFLAGGED = {"A", "B"}, "agree (not flagged)"
CHECK_QUESTION = "If calibrated pessimism were not an option, would you call this distorted?"


def collapse(label):
    return NO_DISTORTION if label == CP_LABEL else label


def classify(orig_dom, orig_sec, votes):
    """votes: the row's valid Primary labels (12-way). Returns (majority 11-way label, group).
    majority is None when no label has > half the votes; group is None when nothing changed."""
    if not votes:
        return None, None
    counts = pd.Series([collapse(v) for v in votes]).value_counts()
    if counts.iloc[0] * 2 <= len(votes):
        return None, None
    new = counts.index[0]
    if new == orig_dom:
        return new, None
    if new == NO_DISTORTION:
        n_cp = sum(v == CP_LABEL for v in votes)
        n_nd = sum(v == NO_DISTORTION for v in votes)
        return new, "redefinition" if n_cp >= n_nd else "genuine error"
    if orig_dom == NO_DISTORTION:
        return new, "genuine error"
    if isinstance(orig_sec, str) and new == orig_sec:
        return new, "primary swap"
    return new, "genuine error"


def rate(sub):
    n, x = len(sub), int(sub.sum())
    return x, n, f"{x} of {n} ({100 * x / n:.1f}%)" if n else "0 of 0 (n/a)"


def write_check_sheet(redef, path):
    raw = pd.read_csv(REPO / "Annotated_data.csv", encoding="utf-8-sig").set_index("Id_Number")
    multi = redef[redef["n_annotators"] >= 2]
    single = redef[redef["n_annotators"] < 2]
    pick = multi.sample(min(N_CHECK, len(multi)), random_state=SEED)
    if len(pick) < N_CHECK:
        pick = pd.concat([pick, single.sample(min(N_CHECK - len(pick), len(single)), random_state=SEED)])
    pick = pick.sample(frac=1, random_state=SEED)  # shuffle so multi-rater rows are not all on top
    wb = Workbook()
    ws = wb.active
    ws.title = "Check"
    ws.append(["Row ID", "Text", "Distorted part", CHECK_QUESTION, "Notes"])
    for c in ws[1]:
        c.font = Font(bold=True)
        c.alignment = Alignment(wrap_text=True, vertical="top")
    for rid in pick["Id_Number"]:
        ws.append([int(rid), raw.at[rid, "Patient Question"], raw.at[rid, "Distorted part"], None, None])
    dv = DataValidation(type="list", formula1='"Yes,No"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"D2:D{len(pick) + 1}")
    for col, width in zip("ABCDE", (8, 90, 50, 22, 30)):
        ws.column_dimensions[col].width = width
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    wb.save(path)
    return pick


def main():
    m = pd.read_csv(OUT / "merged_labels.csv", encoding="utf-8")
    m["filled_from_original"] = m["filled_from_original"].astype(str).str.lower() == "true"
    rows_all = m.drop_duplicates("Row ID").set_index("Row ID")
    labelled = m[m["Primary"].notna()]
    filled = labelled[labelled["filled_from_original"]]
    real = labelled[~labelled["filled_from_original"]]
    unsure = real[real["Unsure"].notna()]
    valid = real[real["Unsure"].isna()]

    recs, no_majority = [], []
    for rid, info in rows_all.iterrows():
        votes = valid.loc[valid["Row ID"] == rid, "Primary"].tolist()
        orig_dom, orig_sec = str(info["original_dominant"]).strip(), info["original_secondary"]
        new, group = classify(orig_dom, orig_sec, votes)
        if votes and new is None:
            no_majority.append(rid)
        recs.append({"Id_Number": rid, "original_label": orig_dom,
                     "original_secondary": orig_sec if isinstance(orig_sec, str) else "",
                     "new_majority_label": new, "n_annotators": len(votes),
                     "unanimous": ("n/a" if len(votes) < 2 else
                                   "yes" if len({collapse(v) for v in votes}) == 1 else "no"),
                     "group": group, "changed": group is not None, "has_majority": new is not None,
                     "binary_changed": (new is not None and
                                        (new == NO_DISTORTION) != (orig_dom == NO_DISTORTION)),
                     "votes": " | ".join(votes), "section": info["section"],
                     "split": info["split"], "bucket": info["bucket"]})
    allrows = pd.DataFrame(recs)
    changes = allrows[allrows["changed"]].drop(columns=["changed", "has_majority"])
    changes = changes.sort_values(["group", "Id_Number"]).reset_index(drop=True)
    cols = ["Id_Number", "original_label", "new_majority_label", "n_annotators", "unanimous", "group",
            "original_secondary", "binary_changed", "votes", "section", "split", "bucket"]
    changes[cols].to_csv(OUT / "label_changes.csv", index=False, encoding="utf-8")

    # ---- counts per group
    usable = allrows[allrows["has_majority"]]
    multi = usable[usable["n_annotators"] >= 2]
    unan = changes[changes["unanimous"] == "yes"]
    group_tbl = pd.DataFrame({
        "group": GROUPS,
        "majority rows (all)": [int((changes["group"] == g).sum()) for g in GROUPS],
        "of which single-annotator": [int(((changes["group"] == g) & (changes["n_annotators"] == 1)).sum())
                                      for g in GROUPS],
        "multi-annotator majority": [int(((changes["group"] == g) & (changes["n_annotators"] >= 2)).sum())
                                     for g in GROUPS],
        "unanimous (2+ annotators, all agree)": [int((unan["group"] == g).sum()) for g in GROUPS],
    })
    group_tbl.loc[len(group_tbl)] = ["total changed"] + [int(group_tbl[c].sum()) for c in group_tbl.columns[1:]]

    # ---- change rate on the test rows
    test = allrows[allrows["split"] == "test"]
    t_use = test[test["has_majority"]]
    flag = t_use[t_use["bucket"].isin(FLAGGED)]
    unfl = t_use[t_use["bucket"] == UNFLAGGED]
    n_flag_all = int(test["bucket"].isin(FLAGGED).sum())
    n_unfl_all = int((test["bucket"] == UNFLAGGED).sum())

    def excluded(bucket_mask):
        sub = test[bucket_mask & ~test["has_majority"]]
        ids = set(sub["Id_Number"])
        n_fill = len(ids & set(filled["Row ID"]))
        n_uns = len(ids & set(unsure["Row ID"]))
        return len(sub), n_fill, n_uns, len(sub) - n_fill - n_uns

    ex_flag = excluded(test["bucket"].isin(FLAGGED))
    ex_unfl = excluded(test["bucket"] == UNFLAGGED)
    xf, nf, sf = rate(flag["changed"])
    xu, nu, su = rate(unfl["changed"])
    _, p_type = fisher_exact([[xf, nf - xf], [xu, nu - xu]])
    bxf, _, bsf = rate(flag["binary_changed"])
    bxu, _, bsu = rate(unfl["binary_changed"])
    _, p_bin = fisher_exact([[bxf, nf - bxf], [bxu, nu - bxu]])
    test_groups = pd.crosstab(changes[changes["split"] == "test"]["bucket"].isin(FLAGGED)
                              .map({True: "flagged", False: "unflagged"}),
                              changes[changes["split"] == "test"]["group"])

    # ---- dropped / excluded rows
    uns_rows = set(unsure["Row ID"])
    uns_rows_lost = sorted(r for r in uns_rows if allrows.set_index("Id_Number").at[r, "n_annotators"] == 0)
    fill_rows = sorted(set(filled["Row ID"]))
    blank_rows = sorted(set(m.loc[m["Primary"].isna(), "Row ID"]))
    blank_lost = sorted(r for r in blank_rows
                        if allrows.set_index("Id_Number").at[r, "n_annotators"] == 0
                        and r not in uns_rows_lost and r not in fill_rows)

    # ---- reconciliation with agreement_report.md (161 CP rows)
    old_cp = []
    for rid, grp in labelled.groupby("Row ID"):
        orig = str(grp["original_dominant"].iloc[0]).strip()
        if orig != NO_DISTORTION and (grp["Primary"] == CP_LABEL).sum() > len(grp) / 2:
            old_cp.append(rid)
    redef_ids = set(changes.loc[changes["group"] == "redefinition", "Id_Number"])
    old_only, new_only = sorted(set(old_cp) - redef_ids), sorted(redef_ids - set(old_cp))
    old_only_unsure = [r for r in old_only if r in uns_rows]

    # ---- optional check sheet
    redef = changes[changes["group"] == "redefinition"]
    pick = write_check_sheet(redef, OUT / "redefinition_check_20.xlsx")

    n_rows = len(allrows)
    md = [
        "# D08 - Errors vs redefinitions, and the label change rate", "",
        f"Source: `merged_labels.csv` ({len(m)} label rows, {n_rows} distinct re-annotated rows). "
        "Script: `scripts/label_changes.py`. Per-row output: `label_changes.csv` (one row per changed label).", "",
        "**How the groups are decided.** This split is a *rule applied to the pair of labels* "
        "(original dominant label, our majority label), not a reading of the original annotators' minds. "
        "Our workbooks offered two no-distortion options, plain `No Distortion` and "
        "`No Distortion (calibrated pessimism)`, so the option an annotator picked tells us which kind of "
        "\"not distorted\" they meant.", "",
        "| Original -> our majority | Group |", "| --- | --- |",
        "| a distortion -> calibrated pessimism | redefinition |",
        "| a distortion -> plain No Distortion | genuine error |",
        "| No Distortion -> a distortion | genuine error |",
        "| type A -> type B, B = the original secondary | primary swap |",
        "| type A -> any other type B | genuine error |", "",
        "The majority label is on the 11-way scale (calibrated pessimism counted as No Distortion) and must hold "
        "**more than half** of a row's valid votes. When the majority is No Distortion, the row is a redefinition if "
        "calibrated-pessimism votes are at least as many as plain No-Distortion votes (a tie goes to redefinition), "
        "otherwise a genuine error. Counting the calibrated-pessimism group as redefinition is the cautious choice: "
        "it keeps us from claiming more errors than we can show.", "",
        "**Annotator count.** The 60 overlap rows have 3-4 annotators; every other row, including all 253 test rows, "
        "was labelled by **one** annotator, so its \"majority\" is that one label. `unanimous` is `yes`/`no` only for "
        "rows with 2+ valid votes and `n/a` otherwise.", "",
        "## Count per group", "",
        group_tbl.to_markdown(index=False), "",
        f"Rows with a usable majority label: **{len(usable)}** of {n_rows} "
        f"({len(multi)} of them with 2+ annotators). Rows with valid votes but no majority (split vote): "
        f"**{len(no_majority)}** {sorted(no_majority)}.", "",
        "## Change rate on the test split", "",
        "Flagged = triage buckets A + B; unflagged = bucket C (\"agree\"), the rows the model did not flag. "
        "The unflagged rate shows how often labels change under review anyway, so the flagged rate only means "
        "something compared with it.", "",
        "| Test rows | In split | Usable | Excluded (filled / Unsure / blank) | Label changed (11-way type) | Binary changed (distorted <-> not) |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| Flagged (A+B) | {n_flag_all} | {nf} | {ex_flag[0]} ({ex_flag[1]} / {ex_flag[2]} / {ex_flag[3]}) | {sf} | {bsf} |",
        f"| Unflagged (C) | {n_unfl_all} | {nu} | {ex_unfl[0]} ({ex_unfl[1]} / {ex_unfl[2]} / {ex_unfl[3]}) | {su} | {bsu} |",
        "",
        f"Fisher's exact test, flagged vs unflagged: type change p = {p_type:.3g}; binary change p = {p_bin:.3g}.", "",
        "Changed test rows by group:", "",
        test_groups.to_markdown(), "",
        "## Rows and votes left out", "",
        f"- **Unsure: {len(unsure)} labelled votes dropped** (on {len(uns_rows)} rows; {len(uns_rows_lost)} rows lost all "
        "their votes this way). Why: the annotator ticked Unsure and the adjudication file "
        "(`unsure_for_adjudication.csv`) was never filled in, so there is no settled label to count. "
        f"Rows lost: {uns_rows_lost}. (`agreement_report.md` lists {int(m['Unsure'].notna().sum())} Unsure ticks in total; "
        f"the other {int((m['Unsure'].notna() & m['Primary'].isna()).sum())} sit on blank Primary cells, which carry no vote anyway.)",
        f"- **Filled from the original label: {len(filled)} votes excluded** ({len(fill_rows)} rows). Why: these were "
        "CS-majors blanks that `relabel_agreement.py` filled with the original dataset label; nobody reviewed them, "
        "so counting them would make labels look unchanged when they were never checked. "
        f"{int(test['Id_Number'].isin(fill_rows).sum())} of these rows are in the test split.",
        f"- **Blank Primary** (no vote, already excluded in `agreement_report.md`): {len(blank_rows)} cells; rows left "
        f"with no vote at all because of a blank: {blank_lost}.", "",
        "## Reconciliation with `agreement_report.md`", "",
        f"`agreement_report.md` counts **{len(old_cp)}** rows originally labelled as a distortion where calibrated "
        "pessimism held a majority on the 12-way scale, counting every non-blank vote (Unsure and filled votes "
        f"included). Here the redefinition group has **{len(redef_ids)}** rows. "
        f"In the old count but not here: {len(old_only)} ({len(old_only_unsure)} because the only vote was Unsure "
        f"and is dropped here; the rest change group under the >half-of-valid-votes rule): {old_only}. "
        f"Here but not in the old count: {len(new_only)} (rows where CP and plain ND votes together form a "
        f"No-Distortion majority and CP is at least tied with plain ND): {new_only}.", "",
        "## Optional check: are the redefinitions really redefinitions?", "",
        f"`redefinition_check_20.xlsx` holds {len(pick)} rows from the redefinition group "
        f"(sampled with seed {SEED}: every multi-annotator redefinition row ({int((pick['n_annotators'] >= 2).sum())}) "
        "is included, the rest are single-annotator rows; order shuffled), with the text and distorted part only, no labels. Give it to one psychology annotator "
        f"with the question: *\"{CHECK_QUESTION}\"* Mostly **yes** means these rows really are redefinitions; "
        "mostly **no** means they were errors all along, and Chapter 9 should say so.", "",
        "**Status: pending** (sheet generated, not yet answered).", "",
    ]
    (OUT / "label_changes_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(group_tbl.to_string(index=False))
    print(f"flagged: {sf}   unflagged: {su}   p={p_type:.3g}")
    print(f"binary flagged: {bsf}   unflagged: {bsu}   p={p_bin:.3g}")
    print(f"old CP {len(old_cp)} vs redefinition {len(redef_ids)}")


if __name__ == "__main__":
    main()
