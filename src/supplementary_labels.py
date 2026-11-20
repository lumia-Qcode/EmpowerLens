"""S03a / S03b - labelling tooling for the scraped supplementary rows.

The labels themselves come from two people (Lumia and Izza), never from a model
(D3 brief S03). This script only prepares their workbooks and processes what
they hand back.

RUN (from the repo root), in this order:

    venv\\Scripts\\python.exe -m src.supplementary_labels sheets      # 1. build the two workbooks
    #   ... Lumia and Izza fill in data/supplementary/labelling/labels_<name>.xlsx ...
    venv\\Scripts\\python.exe -m src.supplementary_labels export      # 2. workbooks -> labels_<name>.csv
    venv\\Scripts\\python.exe -m src.supplementary_labels agreement   # 3. kappa on the 30 shared rows
    #   ... they discuss the disagreements and fill data/supplementary/adjudication.csv ...
    venv\\Scripts\\python.exe -m src.supplementary_labels merge       # 4. supplementary_labelled.csv

Assignment: 30 rows (seeded, stratified by role) go to BOTH annotators; the
other 470 are split in half by thread, so a whole thread goes to one person.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from sklearn.metrics import cohen_kappa_score

from src.data import LABEL_CANON

SEED = 42
N_SHARED = 30
ANNOTATORS = ("lumia", "izza")
NO_DIST = "No Distortion"
CP_LABEL = "No Distortion (calibrated pessimism)"  # same string as the 27 Sep relabelling workbooks
DOMINANT_CHOICES = list(LABEL_CANON) + [CP_LABEL]
SECONDARY_CHOICES = [k for k in LABEL_CANON if k != NO_DIST]
LABEL_COLS = ["dominant", "secondary", "distorted_part", "unsure", "notes"]
SHEET_COLS = ["item_id", "block", "text"] + LABEL_COLS

SUPP = Path("data/supplementary")
SUPP_CSV = SUPP / "supplementary.csv"
LAB_DIR = SUPP / "labelling"
CODEBOOK = "reannotation_2026-09-27/EmpowerLens_Relabelling_Codebook.pdf"
SHARED_FILL, OWN_FILL = "DDEBF7", "E2EFDA"


def is_distorted(label: str) -> bool:
    return label not in (NO_DIST, CP_LABEL)


# ---------------------------------------------------------------- assignment
def assign(df: pd.DataFrame, n_shared: int = N_SHARED, seed: int = SEED) -> dict:
    """Pick the shared rows (stratified by role) and split the rest by thread."""
    rng = np.random.default_rng(seed)
    shared = []
    roles = sorted(df.role.unique())
    for k, role in enumerate(roles):
        ids = sorted(df.loc[df.role == role, "item_id"])
        if k == len(roles) - 1:
            n = n_shared - len(shared)
        else:
            n = int(round(n_shared * len(ids) / len(df)))
        shared += list(rng.choice(ids, size=n, replace=False))
    shared = sorted(shared)

    rest = df[~df.item_id.isin(shared)]
    threads = sorted(rest.thread_id.unique())
    rng.shuffle(threads)
    sizes = rest.groupby("thread_id").size()
    out = {name: [] for name in ANNOTATORS}
    for t in threads:
        who = min(ANNOTATORS, key=lambda n: len(out[n]))
        out[who] += rest.loc[rest.thread_id == t, "item_id"].tolist()
    assert sum(sizes) == sum(len(v) for v in out.values())
    return {"shared": shared, **{n: sorted(v) for n, v in out.items()}}


# ---------------------------------------------------------------- workbooks
def write_workbook(path, rows: pd.DataFrame, shared_ids, annotator: str) -> None:
    """One labelling workbook: rows in the given order, label columns blank."""
    wb = Workbook()
    ws = wb.active
    ws.title = "Label"
    ws.append(SHEET_COLS)
    for c in range(1, len(SHEET_COLS) + 1):
        ws.cell(row=1, column=c).font = Font(bold=True)
    for i, (_, r) in enumerate(rows.iterrows(), start=2):
        block = "shared" if r.item_id in shared_ids else "own"
        ws.append([r.item_id, block, r.text, "", "", "", False, ""])
        fill = PatternFill("solid", fgColor=SHARED_FILL if block == "shared" else OWN_FILL)
        for c, name in enumerate(SHEET_COLS, start=1):
            cell = ws.cell(row=i, column=c)
            cell.fill = fill
            cell.alignment = Alignment(wrap_text=name in ("text", "distorted_part", "notes"), vertical="top")
    widths = {"item_id": 10, "block": 8, "text": 90, "dominant": 34, "secondary": 24,
              "distorted_part": 45, "unsure": 9, "notes": 30}
    for c, name in enumerate(SHEET_COLS, start=1):
        ws.column_dimensions[get_column_letter(c)].width = widths[name]
    ws.freeze_panes = "D2"

    last = max(ws.max_row, 2)
    col = {n: get_column_letter(SHEET_COLS.index(n) + 1) for n in SHEET_COLS}
    d_list, s_list = ",".join(DOMINANT_CHOICES), ",".join(SECONDARY_CHOICES)
    for lst in (d_list, s_list):
        if len(lst) > 255:
            raise SystemExit(f"dropdown list is {len(lst)} chars; Excel allows 255")
    dv_d = DataValidation(type="list", formula1=f'"{d_list}"', allow_blank=False, showErrorMessage=True,
                          error="Choose one of the labels in the dropdown.", errorTitle="Pick from the list")
    dv_s = DataValidation(type="list", formula1=f'"{s_list}"', allow_blank=True, showErrorMessage=True,
                          error="Choose one of the ten distortions, or leave blank.", errorTitle="Pick from the list")
    dv_u = DataValidation(type="list", formula1='"TRUE,FALSE"', allow_blank=True)
    for dv, n in ((dv_d, "dominant"), (dv_s, "secondary"), (dv_u, "unsure")):
        ws.add_data_validation(dv)
        dv.add(f"{col[n]}2:{col[n]}{last}")

    # Red row when: secondary equals dominant; secondary set on a no-distortion row;
    # a distortion with no distorted part quoted.
    d, s, p = col["dominant"], col["secondary"], col["distorted_part"]
    rule = FormulaRule(formula=[
        f'OR(AND(${s}2<>"",${s}2=${d}2),'
        f'AND(${s}2<>"",OR(${d}2="{NO_DIST}",${d}2="{CP_LABEL}")),'
        f'AND(${d}2<>"",${d}2<>"{NO_DIST}",${d}2<>"{CP_LABEL}",${p}2=""))'],
        fill=PatternFill("solid", fgColor="FFC7CE"))
    ws.conditional_formatting.add(f"A2:{get_column_letter(len(SHEET_COLS))}{last}", rule)

    info = wb.create_sheet("Instructions")
    lines = [
        f"S03 labelling workbook for: {annotator.capitalize()}",
        "",
        f"Codebook: {CODEBOOK} (the same one used for the 27 Sep relabelling).",
        "For every row fill: dominant (required), secondary (optional, a different distortion),",
        "distorted_part (copy the exact words from the text that carry the distortion; blank for No Distortion),",
        "unsure (TRUE if you are not confident), notes (anything worth discussing).",
        f"Use '{CP_LABEL}' when a negative expectation is supported by the facts in the text.",
        "",
        "Blue rows (block = shared) are labelled by BOTH of you. Do them first, and do NOT discuss",
        "them with each other until both of you have finished: they measure agreement (Cohen's kappa).",
        "Green rows (block = own) are yours alone.",
        "Red highlight = invalid combination (secondary equals dominant, secondary on a no-distortion row,",
        "or a distortion with no distorted_part).",
        "About 45 seconds per row. Do not use an AI model to label: these labels must come from people.",
        "Do not reorder, delete or edit item_id / block / text. Save the file in place when done.",
    ]
    for ln in lines:
        info.append([ln])
    info.column_dimensions["A"].width = 120
    info["A1"].font = Font(bold=True)
    wb.save(path)


def _clean(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v).strip()


def read_workbook(path) -> pd.DataFrame:
    ws = load_workbook(path, data_only=True)["Label"]
    rows = list(ws.iter_rows(values_only=True))
    header = list(rows[0])
    if header[:len(SHEET_COLS)] != SHEET_COLS:
        raise ValueError(f"{path}: unexpected columns {header}")
    recs = []
    for r in rows[1:]:
        if r[0] is None:
            continue
        rec = dict(zip(SHEET_COLS, r))
        u = rec["unsure"]
        recs.append({"item_id": _clean(rec["item_id"]), "dominant": _clean(rec["dominant"]),
                     "secondary": _clean(rec["secondary"]), "distorted_part": _clean(rec["distorted_part"]),
                     "unsure": u is True or _clean(u).upper() == "TRUE", "notes": _clean(rec["notes"])})
    return pd.DataFrame(recs, columns=["item_id"] + LABEL_COLS)


# ---------------------------------------------------------------- validation
def validate_labels(labels: pd.DataFrame, assigned_ids, texts: dict):
    """Return (errors, warnings). Errors stop the export; warnings are printed."""
    errors, warns = [], []
    lab = labels.copy()
    for c in ("dominant", "secondary", "distorted_part"):
        lab[c] = lab[c].fillna("").astype(str).str.strip()
    have = set(lab.loc[lab.dominant != "", "item_id"])
    missing = [i for i in assigned_ids if i not in have]
    if missing:
        errors.append(f"{len(missing)} assigned rows have no dominant label: {missing[:10]}"
                      + (" ..." if len(missing) > 10 else ""))
    for _, r in lab.iterrows():
        if r.dominant == "":
            continue
        if r.dominant not in DOMINANT_CHOICES:
            errors.append(f"{r.item_id}: dominant {r.dominant!r} is not a codebook label")
        if r.secondary:
            if r.secondary not in SECONDARY_CHOICES:
                errors.append(f"{r.item_id}: secondary {r.secondary!r} is not a distortion")
            elif r.secondary == r.dominant:
                errors.append(f"{r.item_id}: secondary equals dominant")
            elif not is_distorted(r.dominant):
                errors.append(f"{r.item_id}: secondary given on a no-distortion row")
        if r.dominant in DOMINANT_CHOICES and is_distorted(r.dominant):
            if not r.distorted_part:
                warns.append(f"{r.item_id}: distortion with no distorted_part")
            elif r.distorted_part not in texts.get(r.item_id, ""):
                warns.append(f"{r.item_id}: distorted_part is not an exact quote from the text")
    return errors, warns


# ---------------------------------------------------------------- agreement
def _kappa(a, b) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = cohen_kappa_score(a, b)
    return float(k)


def agreement(lumia: pd.DataFrame, izza: pd.DataFrame, shared_ids) -> dict:
    L = lumia.set_index("item_id").loc[list(shared_ids)]
    I = izza.set_index("item_id").loc[list(shared_ids)]
    fold = lambda s: s.replace({CP_LABEL: NO_DIST})
    binary = lambda s: s.map(lambda x: int(is_distorted(x)))
    out = {"n": len(shared_ids)}
    for name, f in (("type", lambda s: s), ("type_cp_folded", fold), ("binary", binary)):
        a, b = f(L.dominant).tolist(), f(I.dominant).tolist()
        out[name] = {"raw": float(np.mean([x == y for x, y in zip(a, b)])), "kappa": _kappa(a, b)}
    sec_l, sec_i = L.secondary.fillna(""), I.secondary.fillna("")
    dis = (L.dominant != I.dominant) | (sec_l != sec_i)
    out["disagreements"] = pd.DataFrame({
        "item_id": L.index[dis],
        "lumia_dominant": L.dominant[dis].values, "lumia_secondary": sec_l[dis].values,
        "izza_dominant": I.dominant[dis].values, "izza_secondary": sec_i[dis].values,
    })
    out["counts"] = pd.DataFrame({"lumia": L.dominant.value_counts(), "izza": I.dominant.value_counts()}
                                 ).fillna(0).astype(int)
    return out


def _fmt_k(k: float) -> str:
    return "n/a (only one label used)" if np.isnan(k) else f"{k:.3f}"


def render_agreement(res: dict) -> str:
    n = res["n"]
    rows = [("Type (11 labels + calibrated pessimism as its own label)", res["type"]),
            ("Type, calibrated pessimism folded into No Distortion", res["type_cp_folded"]),
            ("Binary (distorted vs not; calibrated pessimism = not distorted)", res["binary"])]
    md = ["# S03 - Agreement on the shared rows", "",
          f"Lumia and Izza each labelled the same {n} rows of `supplementary.csv` separately, before",
          "discussing them (rows chosen by `src/supplementary_labels.py sheets`, seed 42, stratified by role;",
          "IDs in `labelling/assignment.json`). Codebook: `" + CODEBOOK + "`.", "",
          "| Comparison | Raw agreement | Cohen's kappa |", "|---|---|---|"]
    for label, r in rows:
        md.append(f"| {label} | {r['raw']:.1%} ({round(r['raw'] * n)}/{n}) | {_fmt_k(r['kappa'])} |")
    md += ["", f"n = {n} rows. Kappa 0 = no better than chance, 1 = perfect. With only {n} rows the kappa is a",
           "rough estimate; type kappa is unstable when some labels appear once or not at all.", "",
           "## Label counts on the shared rows", "", "| Label | Lumia | Izza |", "|---|---|---|"]
    for lab, r in res["counts"].iterrows():
        md.append(f"| {lab} | {r.lumia} | {r.izza} |")
    d = res["disagreements"]
    md += ["", f"## Rows to settle by discussion ({len(d)})", "",
           "Dominant or secondary differs. Final labels go in `adjudication.csv`.", ""]
    if len(d):
        md += ["| item_id | Lumia | Izza |", "|---|---|---|"]
        for _, r in d.iterrows():
            lm = r.lumia_dominant + (f" + {r.lumia_secondary}" if r.lumia_secondary else "")
            iz = r.izza_dominant + (f" + {r.izza_secondary}" if r.izza_secondary else "")
            md.append(f"| {r.item_id} | {lm} | {iz} |")
    else:
        md.append("None.")
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- merge
def merge(supp: pd.DataFrame, lumia: pd.DataFrame, izza: pd.DataFrame, shared_ids,
          adjudication: pd.DataFrame | None) -> pd.DataFrame:
    shared_ids = list(shared_ids)
    res = agreement(lumia, izza, shared_ids)
    need = set(res["disagreements"].item_id)
    adj = {} if adjudication is None else adjudication.fillna("").set_index("item_id").to_dict("index")
    missing = sorted(need - {k for k, v in adj.items() if str(v.get("final_dominant", "")).strip()})
    if missing:
        raise ValueError(f"shared rows still disagree with no final label in adjudication.csv: {missing}")

    L, I = lumia.fillna("").set_index("item_id"), izza.fillna("").set_index("item_id")
    recs = []
    for iid in supp.item_id:
        if iid in shared_ids:
            src = L.loc[iid]
            dom, sec, part = src.dominant, src.secondary, src.distorted_part
            if iid in adj:
                a = adj[iid]
                dom = str(a["final_dominant"]).strip() or dom
                sec = str(a.get("final_secondary", "")).strip()
                part = str(a.get("final_distorted_part", "")).strip() or part
            by, unsure = "both", bool(L.loc[iid, "unsure"]) or bool(I.loc[iid, "unsure"])
        elif iid in L.index:
            r = L.loc[iid]
            dom, sec, part, by, unsure = r.dominant, r.secondary, r.distorted_part, "lumia", bool(r.unsure)
        elif iid in I.index:
            r = I.loc[iid]
            dom, sec, part, by, unsure = r.dominant, r.secondary, r.distorted_part, "izza", bool(r.unsure)
        else:
            raise ValueError(f"{iid} has no label from either annotator")
        if dom not in DOMINANT_CHOICES:
            raise ValueError(f"{iid}: final dominant {dom!r} is not a codebook label")
        cp = dom == CP_LABEL
        recs.append({"item_id": iid, "dominant": NO_DIST if cp else dom, "secondary": sec,
                     "distorted_part": part if is_distorted(dom) else "", "calibrated_pessimism": cp,
                     "y_bin": int(is_distorted(dom)), "labelled_by": by, "unsure": unsure})
    out = supp.merge(pd.DataFrame(recs), on="item_id", how="left", validate="one_to_one")
    assert len(out) == len(supp) and out.item_id.is_unique
    return out


# ---------------------------------------------------------------- CLI
def _load_supp() -> pd.DataFrame:
    return pd.read_csv(SUPP_CSV, encoding="utf-8-sig", dtype=str)


def _load_assignment() -> dict:
    return json.loads((LAB_DIR / "assignment.json").read_text(encoding="utf-8"))


def cmd_sheets(args) -> int:
    supp = _load_supp()
    paths = {n: LAB_DIR / f"labels_{n}.xlsx" for n in ANNOTATORS}
    if any(p.exists() for p in paths.values()) and not args.force:
        raise SystemExit("labelling workbooks already exist - they may hold labels. Use --force to overwrite.")
    LAB_DIR.mkdir(parents=True, exist_ok=True)
    a = assign(supp, N_SHARED, SEED)
    rng = np.random.default_rng(SEED + 1)
    by_id = supp.set_index("item_id")
    for n in ANNOTATORS:
        shared, own = list(a["shared"]), list(a[n])
        rng.shuffle(shared)
        order = shared + own  # own rows stay in id order so a thread's posts sit together
        write_workbook(paths[n], by_id.loc[order].reset_index(), set(a["shared"]), n)
        print(f"{paths[n]}: {len(shared)} shared + {len(own)} own = {len(order)} rows")
    meta = {"seed": SEED, "n_shared": N_SHARED, "source": str(SUPP_CSV).replace("\\", "/"),
            "shared_by_role": supp[supp.item_id.isin(a["shared"])].role.value_counts().to_dict(),
            **a}
    (LAB_DIR / "assignment.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    union = set(a["shared"]) | set(a["lumia"]) | set(a["izza"])
    print(f"union covers {len(union)} of {len(supp)} rows; shared by role {meta['shared_by_role']}")
    return 0


def cmd_export(args) -> int:
    supp, a = _load_supp(), _load_assignment()
    texts = dict(zip(supp.item_id, supp.text))
    ok = True
    for n in ANNOTATORS:
        lab = read_workbook(LAB_DIR / f"labels_{n}.xlsx")
        errors, warns = validate_labels(lab, a["shared"] + a[n], texts)
        for w in warns:
            print(f"[warn] {n}: {w}")
        for e in errors:
            print(f"[FAIL] {n}: {e}")
        if errors:
            ok = False
            continue
        out = SUPP / f"labels_{n}.csv"
        lab.to_csv(out, index=False, encoding="utf-8")
        print(f"wrote {out} ({len(lab)} rows, {len(warns)} warnings)")
    return 0 if ok else 1


def cmd_agreement(args) -> int:
    a = _load_assignment()
    lumia, izza = (pd.read_csv(SUPP / f"labels_{n}.csv", dtype=str, keep_default_na=False) for n in ANNOTATORS)
    res = agreement(lumia, izza, a["shared"])
    (SUPP / "agreement.md").write_text(render_agreement(res), encoding="utf-8")
    d = res["disagreements"].assign(final_dominant="", final_secondary="", final_distorted_part="", note="")
    d.to_csv(SUPP / "adjudication_template.csv", index=False, encoding="utf-8")
    for k in ("type", "type_cp_folded", "binary"):
        print(f"{k:15s} raw {res[k]['raw']:.1%}  kappa {_fmt_k(res[k]['kappa'])}")
    print(f"{len(d)} rows to settle -> fill {SUPP / 'adjudication_template.csv'} and save it as adjudication.csv")
    return 0


def cmd_merge(args) -> int:
    supp, a = _load_supp(), _load_assignment()
    lumia, izza = (pd.read_csv(SUPP / f"labels_{n}.csv", dtype=str, keep_default_na=False) for n in ANNOTATORS)
    for df in (lumia, izza):
        df["unsure"] = df.unsure.str.upper() == "TRUE"
    adj_path = SUPP / "adjudication.csv"
    adj = pd.read_csv(adj_path, dtype=str, keep_default_na=False) if adj_path.exists() else None
    out = merge(supp, lumia, izza, a["shared"], adj)
    path = SUPP / "supplementary_labelled.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"wrote {path}: {len(out)} rows")
    print(out.dominant.value_counts().to_string())
    print(f"calibrated pessimism: {int(out.calibrated_pessimism.sum())}; by labeller: "
          f"{out.labelled_by.value_counts().to_dict()}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S03 labelling tooling for data/supplementary/")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheets", help="build the two labelling workbooks")
    s.add_argument("--force", action="store_true", help="overwrite existing workbooks")
    sub.add_parser("export", help="filled workbooks -> labels_<name>.csv")
    sub.add_parser("agreement", help="kappa + raw agreement on the shared rows")
    sub.add_parser("merge", help="one final label per row -> supplementary_labelled.csv")
    args = ap.parse_args(argv)
    return {"sheets": cmd_sheets, "export": cmd_export, "agreement": cmd_agreement, "merge": cmd_merge}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
