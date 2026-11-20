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
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from sklearn.metrics import cohen_kappa_score

from src.data import LABEL_CANON

SEED = 42
N_SHARED = 30
ANNOTATORS = ("lumia", "izza")
NO_DIST = "No Distortion"
CP_LABEL = "No Distortion (calibrated pessimism)"  # same string as the 27 Sep relabelling workbooks
SECONDARY_CHOICES = sorted((k for k in LABEL_CANON if k != NO_DIST), key=str.lower)
# Dropdown order: the two no-distortion options first, then the ten distortions A-Z.
DOMINANT_CHOICES = [NO_DIST, CP_LABEL] + SECONDARY_CHOICES
LABEL_COLS = ["dominant", "secondary", "distorted_part", "unsure", "notes"]

# Workbook layout: internal name -> (header the annotator sees, column width).
# read_workbook maps headers back to internal names, so the export never depends on
# column letters.
SHEET_LAYOUT = {
    "row_no": ("#", 6),
    "item_id": ("ID", 9),
    "block": ("Set", 9),
    "text": ("Text to read", 80),
    "dominant": ("Main distortion\n(required)", 30),
    "secondary": ("Second distortion\n(optional)", 24),
    "distorted_part": ("Distorted words\n(copy from the text)", 40),
    "unsure": ("Unsure?\n(Yes / blank)", 11),
    "notes": ("Notes\n(optional)", 26),
    "status": ("Status", 22),
}
SHEET_COLS = list(SHEET_LAYOUT)
SHARED_SET, OWN_SET = "Shared", "Yours"

SUPP = Path("data/supplementary")
SUPP_CSV = SUPP / "supplementary.csv"
LAB_DIR = SUPP / "labelling"
CODEBOOK = "reannotation_2026-09-27/EmpowerLens_Relabelling_Codebook.pdf"
SHARED_FILL, OWN_FILL = "DDEBF7", "FFFFFF"
HEADER_FILL, GRID = "1F3864", "BFBFBF"

# Label guide, condensed from the relabelling codebook (27 Sep 2026).
GUIDE = [  # label, what the thought does, typical cues, example
    ("All-or-nothing thinking", "Sees things in two extremes, with no middle ground",
     "always/never about outcomes, perfect/ruined, total failure", "If I don't get an A, the whole year is wasted."),
    ("Emotional Reasoning", "Treats a feeling as proof of a fact", "I feel ___, so it must be ___",
     "I feel stupid, so I must be stupid."),
    ("Fortune-telling", "Predicts a negative future as if it were certain", "will, going to, never going to",
     "I'll never find a job."),
    ("Labeling", "Attaches a fixed global label to self or others", "I'm a/an..., he's such a...", "I'm a loser."),
    ("Magnification", "Blows up how bad something is, or shrinks the good; catastrophizing",
     "disaster, unbearable, the worst, can't cope", "If I stutter in the meeting, it will be a catastrophe."),
    ("Mental filter", "Dwells on one negative detail and ignores the positives (includes discounting good things)",
     "but, only, doesn't count, yes but", "She praised my work, but she noticed one typo, so it went badly."),
    ("Mind Reading", "Assumes what others think, without evidence", "they think, she must feel, he probably believes",
     "My boss thinks I'm incompetent."),
    ("Overgeneralization", "Draws a sweeping rule from one or a few events", "every time, nothing ever, everyone",
     "My friend cancelled once, so nobody wants to spend time with me."),
    ("Personalization", "Takes responsibility or blame for things outside one's control", "my fault, because of me",
     "My parents argue because I'm a bad son."),
    ("Should statements", "Rigid rules for self or others; guilt or anger when broken", "should, must, have to, ought to",
     "I should be able to handle this without help."),
    (NO_DIST, "Negative or worried, but the reasoning holds up. Facts, feelings, questions, or someone else's "
     "distortion the writer doesn't adopt", "-", "My mother died last month and I can't stop crying."),
    (CP_LABEL, "A negative expectation or self-assessment that FITS the evidence in the text: grounded, hedged, "
     "proportionate. Looks like Fortune-telling / Magnification / Overgeneralization but the reasoning holds",
     "might, probably, because (stated evidence)", "I failed the last two driving tests, so I'm nervous about the third."),
]
PAIRS = [  # pair, deciding question, pick the first if, pick the second if
    ("Fortune-telling vs Mind Reading", "Future, or someone's thoughts now?", "A future event is predicted",
     "Another person's current opinion is assumed"),
    ("Fortune-telling vs Magnification", "Predicting, or inflating size?", "'It will happen' is the core",
     "'It would be terrible' is the core"),
    ("Overgeneralization vs All-or-nothing", "Rule from past events, or a two-option view?",
     "Built from one or few past events", "Only two outcomes exist, no middle"),
    ("Overgeneralization vs Labeling", "Conclusion about events or identity?", "A pattern of events ('this always happens')",
     "A fixed identity ('I'm a failure')"),
    ("Mental filter vs Magnification", "Good ignored, or bad inflated?", "Positives are present but dismissed",
     "One bad thing is blown up, no positives mentioned"),
    ("Personalization vs Labeling", "Blame for an event, or a global identity?", "'It happened because of me'", "'I am a ___'"),
    ("Emotional Reasoning vs Labeling", "Is the feeling the evidence?", "'I feel X, so X is true'",
     "The label stands without a feeling as proof"),
    ("Should statements vs All-or-nothing", "A rule, or an extreme outcome?", "must/should rule about behaviour",
     "Outcome judged as total success or total failure"),
]


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
_THIN = Side(style="thin", color=GRID)
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_WRAP_TOP = Alignment(wrap_text=True, vertical="top")
OK_STATUS = "Done"


def _row_height(texts_and_widths) -> float:
    """Rough row height so wrapped text is visible without resizing (Calibri 11)."""
    lines = 1
    for text, width in texts_and_widths:
        text = str(text or "")
        per_line = max(int(width * 1.15), 1)
        n = sum(max(1, -(-len(part) // per_line)) for part in text.split("\n"))
        lines = max(lines, n)
    return min(409, 15 * lines + 6)


def _header(ws, labels, widths, row=None):
    if row is None:
        ws.append(labels)
        row = ws.max_row
    else:
        for c, v in enumerate(labels, start=1):
            ws.cell(row=row, column=c, value=v)
    for c, w in enumerate(widths, start=1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor=HEADER_FILL)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = _BORDER
        if w:
            ws.column_dimensions[get_column_letter(c)].width = w
    return row


def _status_formula(r: int, col: dict) -> str:
    """Excel formula for the Status column: Done, or what is wrong with the row."""
    d, s, p = (f"{col[n]}{r}" for n in ("dominant", "secondary", "distorted_part"))
    nd = f'LEFT({d},{len(NO_DIST)})="{NO_DIST}"'  # true for both no-distortion options
    return (f'=IF({d}="","",IF(AND({s}<>"",{s}={d}),"Fix: second = main",'
            f'IF(AND({s}<>"",{nd}),"Fix: no second with No Distortion",'
            f'IF(AND(NOT({nd}),{p}=""),"Add the distorted words","{OK_STATUS}"))))')


def _label_sheet(ws, rows, shared_ids):
    col = {n: get_column_letter(i + 1) for i, n in enumerate(SHEET_COLS)}
    _header(ws, [h for h, _ in SHEET_LAYOUT.values()], [w for _, w in SHEET_LAYOUT.values()])
    ws.row_dimensions[1].height = 36
    text_w = SHEET_LAYOUT["text"][1]
    for k, (_, r) in enumerate(rows.iterrows(), start=1):
        i = k + 1
        shared = r.item_id in shared_ids
        ws.append([k, r.item_id, SHARED_SET if shared else OWN_SET, r.text,
                   None, None, None, None, None, _status_formula(i, col)])
        fill = PatternFill("solid", fgColor=SHARED_FILL if shared else OWN_FILL)
        for c, name in enumerate(SHEET_COLS, start=1):
            cell = ws.cell(row=i, column=c)
            cell.fill, cell.border, cell.alignment = fill, _BORDER, _WRAP_TOP
            if name in ("row_no", "item_id"):
                cell.font = Font(color="7F7F7F")
            elif name == "block":
                cell.font = Font(bold=shared, color="1F3864" if shared else "7F7F7F")
            elif name == "status":
                cell.font = Font(bold=True)
        ws.row_dimensions[i].height = _row_height([(r.text, text_w)])
    last = max(ws.max_row, 2)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{col['status']}{last}"
    ws.sheet_properties.tabColor = "2F5597"

    d_list, s_list = ",".join(DOMINANT_CHOICES), ",".join(SECONDARY_CHOICES)
    for lst in (d_list, s_list):
        if len(lst) > 255:
            raise SystemExit(f"dropdown list is {len(lst)} chars; Excel allows 255")
    dvs = {
        "dominant": DataValidation(
            type="list", formula1=f'"{d_list}"', allow_blank=True, showErrorMessage=True,
            errorTitle="Pick from the list", error="Choose one of the labels in the dropdown.",
            showInputMessage=True, promptTitle="Main distortion",
            prompt="Required. The distortion the writer's distress hangs on. Negative but reasonable? "
                   "Pick a No Distortion option. Definitions: 'Label guide' tab."),
        "secondary": DataValidation(
            type="list", formula1=f'"{s_list}"', allow_blank=True, showErrorMessage=True,
            errorTitle="Pick from the list", error="Choose one of the ten distortions, or leave blank.",
            showInputMessage=True, promptTitle="Second distortion",
            prompt="Optional. Only a clearly separate second distortion, shown by different words. "
                   "Most rows: leave blank."),
        "distorted_part": DataValidation(
            showInputMessage=True, promptTitle="Distorted words",
            prompt="Copy the exact words from the Text that show the main distortion (double-click the "
                   "Text cell, select, Ctrl+C). Blank for No Distortion."),
        "unsure": DataValidation(
            type="list", formula1='"Yes"', allow_blank=True, showErrorMessage=True,
            errorTitle="Yes or blank", error="Pick Yes, or leave the cell empty.",
            showInputMessage=True, promptTitle="Unsure?",
            prompt="Yes if you're torn or it took over a minute. Still give your best guess."),
        "notes": DataValidation(showInputMessage=True, promptTitle="Notes",
                                prompt="Optional: a few words on why you were torn, or a third distortion."),
    }
    for n, dv in dvs.items():
        ws.add_data_validation(dv)
        dv.add(f"{col[n]}2:{col[n]}{last}")

    st = f"{col['status']}2:{col['status']}{last}"
    ws.conditional_formatting.add(st, FormulaRule(
        formula=[f'${col["status"]}2="{OK_STATUS}"'],
        fill=PatternFill("solid", fgColor="C6EFCE"), font=Font(bold=True, color="006100")))
    ws.conditional_formatting.add(st, FormulaRule(
        formula=[f'AND(${col["status"]}2<>"",${col["status"]}2<>"{OK_STATUS}")'],
        fill=PatternFill("solid", fgColor="FFC7CE"), font=Font(bold=True, color="9C0006")))
    return col


def _start_sheet(ws, annotator, other, n_shared, n_own, col):
    ws.sheet_properties.tabColor = "548235"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 30
    ws.column_dimensions["B"].width = 110
    lab, stc, setc = "'Label'", col["status"], col["block"]
    hours = (n_shared + n_own) * 45 / 3600

    def line(a="", b="", bold=False, size=11, color="000000"):
        ws.append([a, b])
        r = ws.max_row
        for c in (1, 2):
            ws.cell(row=r, column=c).alignment = _WRAP_TOP
        ws.cell(row=r, column=1).font = Font(bold=True, size=size, color=color)
        ws.cell(row=r, column=2).font = Font(bold=bold, size=size, color=color)
        ws.row_dimensions[r].height = _row_height([(b, 110)]) if b else 18
        return r

    r = line("EmpowerLens labelling", "", size=16, color="1F3864")
    ws.row_dimensions[r].height = 26
    line("Workbook for", annotator.capitalize(), bold=True, size=13)
    line("Rows", f"{n_shared} shared (blue, do these first) + {n_own} yours = {n_shared + n_own} rows, "
                 f"about 45 seconds each (~{hours:.1f} hours in total; breaks are fine)")
    line()
    line("YOUR PROGRESS", "", color="1F3864")
    for a, f in (
        ("Shared rows done", f'=COUNTIFS({lab}!{setc}:{setc},"{SHARED_SET}",{lab}!{stc}:{stc},"{OK_STATUS}")'
                             f'&" of {n_shared}"'),
        ("Your own rows done", f'=COUNTIFS({lab}!{setc}:{setc},"{OWN_SET}",{lab}!{stc}:{stc},"{OK_STATUS}")'
                               f'&" of {n_own}"'),
        ("All rows done", f'=COUNTIF({lab}!{stc}:{stc},"{OK_STATUS}")&" of {n_shared + n_own}"'),
        ("Rows that need fixing", f'=COUNTIF({lab}!{stc}:{stc},"Fix*")+COUNTIF({lab}!{stc}:{stc},"Add*")'),
    ):
        r = line(a, None)
        ws.cell(row=r, column=2, value=f).font = Font(bold=True, size=12)
        ws.cell(row=r, column=2).alignment = Alignment(horizontal="left")
    line()
    line("HOW TO LABEL", "", color="1F3864")
    steps = [
        f"Open the 'Label' tab. The blue rows (Set = {SHARED_SET}) come first: label all {n_shared} of them "
        f"before anything else.",
        "Read the whole text. Then pick the Main distortion from the dropdown (required). "
        "Not sure what a label means? See the 'Label guide' tab.",
        "Negative or worried, but the reasoning holds up? Pick 'No Distortion'. If it makes a negative "
        "prediction or judgement that FITS the evidence in the text, pick 'No Distortion (calibrated pessimism)'.",
        "Second distortion: only if a clearly separate second one is shown by different words. Most rows have none.",
        "Distorted words: copy the exact words from the text that show the main distortion "
        "(double-click the Text cell, select the words, Ctrl+C, then paste). Leave blank for No Distortion.",
        "Unsure: pick Yes if you're torn or it took over a minute, and still give your best guess. "
        "Notes: a few words on why.",
        "Check the Status column. It turns green ('Done') when the row is complete, or red and says what to fix.",
        "Save often (Ctrl+S). Keep the file name and folder. When you have finished, tell the team.",
    ]
    for k, s in enumerate(steps, start=1):
        line(f"Step {k}", s)
    line()
    line("RULES", "", color="1F3864")
    rules = [
        f"Label on your own. Do NOT discuss the shared rows with {other.capitalize()} until you have both "
        "finished them: they measure how well you agree (Cohen's kappa). Don't change them afterwards.",
        "Judge the text, not the person: label what the words show; don't guess at backstory that isn't written.",
        "No AI tools. These labels must come from people.",
        "Don't edit or delete the #, ID, Set or Text columns. Sorting and filtering with the header arrows is fine.",
        f"Full codebook: {CODEBOOK}. The 'Label guide' tab has the short version.",
    ]
    for s in rules:
        line("-", s)


def _guide_sheet(ws):
    ws.sheet_properties.tabColor = "C55A11"
    ws.sheet_view.showGridLines = False
    widths = [30, 46, 34, 46]

    def table(title, header, rows):
        ws.append([title])
        t = ws.max_row
        ws.cell(row=t, column=1).font = Font(bold=True, size=13, color="1F3864")
        ws.row_dimensions[t].height = 22
        _header(ws, header, widths)
        for k, row in enumerate(rows):
            ws.append(list(row))
            r = ws.max_row
            band = PatternFill("solid", fgColor="F2F2F2" if k % 2 else "FFFFFF")
            for c in range(1, len(row) + 1):
                cell = ws.cell(row=r, column=c)
                cell.alignment, cell.border, cell.fill = _WRAP_TOP, _BORDER, band
            ws.cell(row=r, column=1).font = Font(bold=True)
            ws.row_dimensions[r].height = _row_height(zip(row, widths))
        ws.append([])

    table("The labels (from the relabelling codebook)",
          ["Label", "The thought does this", "Typical cues", "Example"], GUIDE)
    table("Confusable pairs: ask the deciding question",
          ["Pair", "Deciding question", "Pick the first if...", "Pick the second if..."], PAIRS)
    ws.append(["Main vs second distortion"])
    ws.cell(row=ws.max_row, column=1).font = Font(bold=True, size=13, color="1F3864")
    for s in [
        "Main = the one the writer's distress hangs on. If you removed it, would the text lose its main problem?",
        "Tie-break: if two seem equal, pick the one in the distorted words as main.",
        "Second only if clearly present: you should be able to point to different words for it.",
        "No third label. If you see three, keep the two strongest and put the third in Notes.",
        "Don't double-count one phrase. \"I'm a failure and always will be\" is Labeling + Fortune-telling "
        "only because it makes two separate claims.",
        "Test for the No Distortion options: could a reasonable, well-informed person think this, given "
        "what's written? If yes, pick one of them.",
    ]:
        ws.append(["-", s])
        r = ws.max_row
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
        ws.cell(row=r, column=2).alignment = _WRAP_TOP
        ws.row_dimensions[r].height = _row_height([(s, sum(widths[1:]))])
    ws.freeze_panes = "A3"


def write_workbook(path, rows: pd.DataFrame, shared_ids, annotator: str) -> None:
    """One labelling workbook: a Start here tab, the Label tab (label columns blank), a Label guide tab."""
    wb = Workbook()
    start = wb.active
    start.title = "Start here"
    label = wb.create_sheet("Label")
    guide = wb.create_sheet("Label guide")
    shared_ids = set(shared_ids)
    n_shared = int(rows.item_id.isin(shared_ids).sum())
    col = _label_sheet(label, rows, shared_ids)
    other = next((a for a in ANNOTATORS if a != annotator), "the other annotator")
    _start_sheet(start, annotator, other, n_shared, len(rows) - n_shared, col)
    _guide_sheet(guide)
    wb.active = 0
    wb.save(path)


def _clean(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    return str(v).strip()


def read_workbook(path) -> pd.DataFrame:
    """Read the Label tab back by header text, so column order or sorting doesn't matter."""
    ws = load_workbook(path, data_only=True)["Label"]
    rows = list(ws.iter_rows(values_only=True))
    to_name = {h: n for n, (h, _) in SHEET_LAYOUT.items()} | {n: n for n in SHEET_LAYOUT}
    idx = {to_name[h]: i for i, h in enumerate(rows[0]) if h in to_name}
    missing = [n for n in ["item_id"] + LABEL_COLS if n not in idx]
    if missing:
        raise ValueError(f"{path}: Label tab is missing columns {missing}; header is {rows[0]}")
    recs = []
    for r in rows[1:]:
        iid = _clean(r[idx["item_id"]])
        if not iid:
            continue
        u = r[idx["unsure"]]
        recs.append({"item_id": iid, "dominant": _clean(r[idx["dominant"]]),
                     "secondary": _clean(r[idx["secondary"]]), "distorted_part": _clean(r[idx["distorted_part"]]),
                     "unsure": u is True or _clean(u).upper() in ("YES", "TRUE"), "notes": _clean(r[idx["notes"]])})
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
