"""S09: flag suspect synthetic labels three ways, and build the blind S10 review sheet.

Checks (each yes/no per synthetic row):
  neighbours   most of the row's 10 nearest neighbours (TF-IDF + SVD, cosine) carry a different
               label: `src.cluster_label_errors --only neighbours --input ...` (run it first).
               The literal ">0.5" rule flags over half the rows (topic, not error: the brief's ARI
               warning), so the cut used is disagreement >= 0.8 (8 of 10 neighbours); the doc
               reports the count at every cut.
  model        the D12 model (MentalRoBERTa, seed 42; trained on the benchmark only) confidently
               disagrees with the panel label. Same prediction path as src/evaluate.py
               (src/app_parity.run -> predict_logits):
                 No Distortion row: P(distorted) >= the D13 binary threshold (0.977);
                 distorted row:     P(distorted) <= 0.10, or the model's top type has prob >= 0.5,
                                    is neither the panel's dominant nor secondary, and the panel's
                                    dominant is below its D13 per-type threshold.
  close_votes  the panel accepted the row 2-to-1 (critic_votes[dominant] == 2, or
               no_distortion_votes == 2 for No Distortion rows).

Outputs:
  data/synthetic/flags.csv          one row per synthetic row, yes/no per check + the scores
  data/synthetic/review_sheet.xlsx  flagged rows still in the S07 clean file + 30 random unflagged
                                    clean rows, shuffled; review_id + text + empty label columns only
  data/synthetic/review_key.csv     review_id -> synthetic id, panel label, which checks flagged it.
                                    Not to be opened by the S10 labeller until S10 is finished.

    venv\\Scripts\\python.exe -m src.synth_label_flags --models ../empowerlens-demo/models/d12
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from src.data import DISTORTIONS, LABEL_CANON
from src.supplementary_labels import (_BORDER, _WRAP_TOP, DOMINANT_CHOICES, SECONDARY_CHOICES, _guide_sheet,
                                      _header, _row_height, _status_formula)

SYN = Path("data/synthetic/synthetic_train_en.jsonl")
CLEAN = Path("data/synthetic/synthetic_train_en_clean.jsonl")
NEIGHBOURS = Path("results/results_synth_neighbours/unsupervised_queue.csv")
OUT_DIR = Path("data/synthetic")
RES = Path("results/results_synth_label_flags")
NEIGHBOUR_CUT = 0.8
LOW_P = 0.10
TYPE_P = 0.5
N_RANDOM = 30
SEED = 42

NO_DIST = "No Distortion"
CANON_TO_RAW = {v: k for k, v in LABEL_CANON.items()}


# ---------------------------------------------------------------- the three checks
def neighbour_flag(disagreement: float, cut: float = NEIGHBOUR_CUT) -> bool:
    return disagreement >= cut


def close_vote_flag(row: dict) -> bool:
    if row["dominant_distortion"] == NO_DIST:
        return row.get("no_distortion_votes") == 2
    return (row.get("critic_votes") or {}).get(row["dominant_distortion"]) == 2


def model_flag(row: dict, p_dist: float, probs: dict, thresholds: dict, bin_thr: float) -> tuple[bool, str]:
    """Confident disagreement between the D12 model and the panel label. Returns (flag, reason)."""
    dom = row["dominant_distortion"]
    if dom == NO_DIST:
        if p_dist >= bin_thr:
            return True, f"P(distorted) {p_dist:.3f} >= {bin_thr:.3f}"
        return False, ""
    if p_dist <= LOW_P:
        return True, f"P(distorted) {p_dist:.3f} <= {LOW_P}"
    d, s = LABEL_CANON[dom], LABEL_CANON.get(row.get("secondary_distortion") or "", None)
    top = max(probs, key=probs.get)
    if probs[top] >= TYPE_P and top not in (d, s) and probs[d] < thresholds[d]:
        return True, f"model {CANON_TO_RAW[top]} {probs[top]:.2f}; {dom} {probs[d]:.2f} < {thresholds[d]}"
    return False, ""


# ---------------------------------------------------------------- model scores
def model_scores(models: Path, texts: list[str], device: str = "cpu") -> pd.DataFrame:
    import os
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from src.app_parity import run
    from src.train_transformer import resolve_device
    card = json.loads((models / "MODEL_CARD.json").read_text(encoding="utf-8"))
    dev = resolve_device(device)
    s = pd.Series(texts)
    bmeta, blog = run(models / card["binary"]["checkpoint"], s, dev)
    z = blog - blog.max(axis=1, keepdims=True)
    p = np.exp(z[:, 1]) / np.exp(z).sum(axis=1)
    mmeta, mlog = run(models / card["multilabel"]["checkpoint"], s, dev)
    probs = 1 / (1 + np.exp(-mlog))
    df = pd.DataFrame(probs, columns=[f"p_{d}" for d in DISTORTIONS])
    df.insert(0, "p_distorted", p)
    meta = {"binary_threshold": float(bmeta["binary_threshold"]),
            "thresholds": dict(zip(DISTORTIONS, map(float, mmeta["thresholds"]))),
            "model_version": card["model_version"]}
    return df, meta


# ---------------------------------------------------------------- review sheet
SHEET = {  # internal -> (header, width); no route, no panel label
    "review_id": ("Review ID", 11),
    "text": ("Text to read", 80),
    "dominant": ("Main distortion\n(required)", 30),
    "secondary": ("Second distortion\n(optional)", 24),
    "distorted_part": ("Distorted words\n(copy from the text)", 40),
    "unsure": ("Unsure?\n(Yes / blank)", 11),
    "notes": ("Notes\n(optional)", 26),
    "status": ("Status", 22),
}


def write_review_sheet(path: Path, items: pd.DataFrame) -> None:
    wb = Workbook()
    start = wb.active
    start.title = "Start here"
    ws = wb.create_sheet("Label")
    _guide_sheet(wb.create_sheet("Label guide"))
    names = list(SHEET)
    col = {n: get_column_letter(i + 1) for i, n in enumerate(names)}
    _header(ws, [h for h, _ in SHEET.values()], [w for _, w in SHEET.values()])
    ws.row_dimensions[1].height = 36
    for k, r in enumerate(items.itertuples(), start=2):
        ws.append([r.review_id, r.text, None, None, None, None, None, _status_formula(k, col)])
        for c in range(1, len(names) + 1):
            cell = ws.cell(row=k, column=c)
            cell.border, cell.alignment = _BORDER, _WRAP_TOP
        ws.cell(row=k, column=1).font = Font(color="7F7F7F")
        ws.cell(row=k, column=len(names)).font = Font(bold=True)
        ws.row_dimensions[k].height = _row_height([(r.text, SHEET["text"][1])])
    last = max(ws.max_row, 2)
    ws.freeze_panes = "C2"
    for n, choices in (("dominant", DOMINANT_CHOICES), ("secondary", SECONDARY_CHOICES)):
        dv = DataValidation(type="list", formula1='"' + ",".join(choices) + '"', allow_blank=True,
                            showErrorMessage=True, error="Choose from the dropdown.")
        ws.add_data_validation(dv)
        dv.add(f"{col[n]}2:{col[n]}{last}")
    dv = DataValidation(type="list", formula1='"Yes"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{col['unsure']}2:{col['unsure']}{last}")

    start.column_dimensions["A"].width = 110
    lines = [
        ("S10 — blind label check of synthetic rows", True),
        (f"{len(items)} short texts. Label each one on the 'Label' tab exactly as in S03, with the codebook.", False),
        ("Main distortion is required; second only if clearly separate; copy the distorted words from the text.", False),
        ("You are not told why a row is here or what label it was generated for. Some were flagged by checks, "
         "some were picked at random; that's deliberate.", False),
        ("Do NOT open data/synthetic/review_key.csv until every row is labelled (Status = Done).", True),
        ("Definitions and confusable pairs: 'Label guide' tab.", False),
    ]
    for text, bold in lines:
        start.append([text])
        c = start.cell(row=start.max_row, column=1)
        c.font, c.alignment = Font(bold=bold, size=13 if bold else 11), Alignment(wrap_text=True)
    start.sheet_view.showGridLines = False
    ws.sheet_properties.tabColor = "2F5597"
    start.cell(row=1, column=1).fill = PatternFill("solid", fgColor="DDEBF7")
    wb.save(path)


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="S09: flag suspect synthetic labels; build the S10 review sheet")
    ap.add_argument("--models", default="../empowerlens-demo/models/d12")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--neighbour-cut", type=float, default=NEIGHBOUR_CUT)
    ap.add_argument("--reuse-scores", action="store_true", help="reuse saved model scores instead of re-predicting")
    args = ap.parse_args(argv)

    rows = [json.loads(line) for line in SYN.read_text(encoding="utf-8").splitlines() if line.strip()]
    clean_ids = {json.loads(line)["id"] for line in CLEAN.read_text(encoding="utf-8").splitlines() if line.strip()}
    nb = pd.read_csv(NEIGHBOURS).set_index("Id_Number")
    RES.mkdir(parents=True, exist_ok=True)

    scores_csv, meta_json = RES / "model_scores.csv", RES / "model_meta.json"
    if args.reuse_scores and scores_csv.exists():
        sc, meta = pd.read_csv(scores_csv), json.loads(meta_json.read_text(encoding="utf-8"))
    else:
        sc, meta = model_scores(Path(args.models), [r["text"] for r in rows], args.device)
        sc.insert(0, "id", [r["id"] for r in rows])
        sc.to_csv(scores_csv, index=False)
        meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    sc = sc.set_index("id")

    out = []
    for r in rows:
        s = sc.loc[r["id"]]
        probs = {d: float(s[f"p_{d}"]) for d in DISTORTIONS}
        mflag, why = model_flag(r, float(s.p_distorted), probs, meta["thresholds"], meta["binary_threshold"])
        dis = float(nb.loc[r["id"], "neighbour_disagreement"])
        out.append({
            "id": r["id"], "in_clean": "yes" if r["id"] in clean_ids else "no",
            "flag_neighbours": "yes" if neighbour_flag(dis, args.neighbour_cut) else "no",
            "flag_model": "yes" if mflag else "no",
            "flag_close_votes": "yes" if close_vote_flag(r) else "no",
            "neighbour_disagreement": dis,
            "neighbour_majority_label": nb.loc[r["id"], "neighbour_majority_label"],
            "model_p_distorted": round(float(s.p_distorted), 4),
            "model_top_type": CANON_TO_RAW[max(probs, key=probs.get)],
            "model_top_prob": round(max(probs.values()), 4),
            "model_reason": why,
            "panel_votes_for_label": (r["no_distortion_votes"] if r["dominant_distortion"] == NO_DIST
                                      else r["critic_votes"].get(r["dominant_distortion"])),
        })
    flags = pd.DataFrame(out)
    checks = ["flag_neighbours", "flag_model", "flag_close_votes"]
    flags["n_checks"] = (flags[checks] == "yes").sum(axis=1)
    flags.to_csv(OUT_DIR / "flags.csv", index=False, encoding="utf-8")

    # review sheet: flagged clean rows + 30 random unflagged clean rows, shuffled
    text_of = {r["id"]: r["text"] for r in rows}
    label_of = {r["id"]: r["dominant_distortion"] for r in rows}
    clean = flags[flags.in_clean == "yes"]
    flagged = clean[clean.n_checks > 0]
    unflagged = clean[clean.n_checks == 0]
    rand = unflagged.sample(n=min(N_RANDOM, len(unflagged)), random_state=SEED)
    pick = pd.concat([flagged.assign(random="no"), rand.assign(random="yes")])
    pick = pick.sample(frac=1, random_state=SEED).reset_index(drop=True)
    pick["review_id"] = [f"R{k:03d}" for k in range(1, len(pick) + 1)]
    pick["text"] = pick.id.map(text_of)
    write_review_sheet(OUT_DIR / "review_sheet.xlsx", pick[["review_id", "text"]])
    key = pick[["review_id", "id", "random", *checks]].copy()
    key.insert(2, "panel_label", key.id.map(label_of))
    key_path = OUT_DIR / "review_key.csv"
    with key_path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("# S10: do not open until every row of review_sheet.xlsx is labelled.\n")
        key.to_csv(fh, index=False)

    cuts = {c: int((flags.neighbour_disagreement > c).sum()) for c in (0.5, 0.6, 0.7)}
    cuts.update({f">={c}": int((flags.neighbour_disagreement >= c).sum()) for c in (0.8, 0.9)})
    summary = {
        "n_rows": len(flags), "n_clean": int((flags.in_clean == "yes").sum()),
        "flagged_per_check": {c: int((flags[c] == "yes").sum()) for c in checks},
        "rows_flagged_by_n_checks": flags.n_checks.value_counts().sort_index().to_dict(),
        "neighbour_counts_by_cut": cuts, "neighbour_cut_used": args.neighbour_cut,
        "flagged_per_check_by_label": {c: flags[flags[c] == "yes"].id.map(label_of).value_counts().to_dict()
                                       for c in checks},
        "pairwise_overlap": {f"{a}&{b}": int(((flags[a] == "yes") & (flags[b] == "yes")).sum())
                             for i, a in enumerate(checks) for b in checks[i + 1:]},
        "review_sheet": {"flagged": len(flagged), "random": len(rand), "total": len(pick)},
        "model_version": meta["model_version"],
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=int))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
