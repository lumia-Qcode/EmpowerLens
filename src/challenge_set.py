"""
D19 - verify the justified-concern pilot pairs, check them for wording leaks, and score the model.

RUN (from repo root, in this order):
    venv\\Scripts\\python.exe -m src.challenge_set sheet     # blind annotator sheet + key + items.csv
    venv\\Scripts\\python.exe -m src.challenge_set leak      # official leak check -> leak_check.json
    # Kaggle (notebooks/kaggle_runner_d19_challenge.ipynb), per seed:
    python -m src.challenge_set predict --checkpoint <binary ckpt> --out data/challenge_set/predictions_<seed>.csv
    venv\\Scripts\\python.exe -m src.challenge_set score     # -> data/challenge_set/results.md

Inputs: data/challenge_set/pairs_lumia.csv + pairs_izza.csv (25 pairs each, 100 texts).
Item IDs (c001..c100) are assigned after a seeded shuffle, so the sheet the annotator sees
carries no pair, type or intended label. The key (annotator_key.csv) is opened only for scoring.

The challenge set is evaluation only (standing rule 8): nothing here trains on it, and the
pairs are never edited. The binary model is scored at its D13 threshold (stored in meta.json as
``binary_threshold``), with specificity / FPR first, never accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.check_challenge_pairs import COLS, LEAK_LIMIT, leak_f1

DIR = Path("data/challenge_set")
PAIR_FILES = [DIR / "pairs_lumia.csv", DIR / "pairs_izza.csv"]
SEEDS = (42, 1337, 2024)
SHUFFLE_SEED = 42
LABELS = ("distorted", "not distorted")


# ---------------------------------------------------------------------------- items

def load_pairs(paths=PAIR_FILES) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, encoding="utf-8")
        missing = [c for c in COLS if c not in df.columns]
        if missing:
            raise SystemExit(f"{p}: missing columns {missing}")
        df["source_file"] = Path(p).name
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if out["pair_id"].duplicated().any():
        raise SystemExit(f"duplicate pair_id across files: {out.loc[out['pair_id'].duplicated(), 'pair_id'].tolist()}")
    return out


def build_items(pairs: pd.DataFrame, seed: int = SHUFFLE_SEED) -> pd.DataFrame:
    """Two items per pair, shuffled with a fixed seed, then numbered c001..cNNN."""
    rows = []
    for _, r in pairs.iterrows():
        for intended, col in (("justified", "justified_text"), ("distorted", "distorted_text")):
            rows.append({"pair_id": r["pair_id"], "distortion_type": r["distortion_type"],
                         "intended": intended, "text": r[col], "source_file": r["source_file"]})
    items = pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    items.insert(0, "item_id", [f"c{i + 1:03d}" for i in range(len(items))])
    return items


# ---------------------------------------------------------------------------- sheet

def write_sheet(items: pd.DataFrame, out_dir: Path = DIR) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    out_dir.mkdir(parents=True, exist_ok=True)
    items[["item_id", "pair_id", "distortion_type", "intended", "source_file"]].to_csv(
        out_dir / "annotator_key.csv", index=False, encoding="utf-8")
    items[["item_id", "text"]].to_csv(out_dir / "items.csv", index=False, encoding="utf-8")

    wb = Workbook()
    ws = wb.active
    ws.title = "Label"
    ws.append(["item_id", "text", "label", "notes"])
    for _, r in items.iterrows():
        ws.append([r["item_id"], r["text"], None, None])
    for c in ws[1]:
        c.font = Font(bold=True)
    ws.column_dimensions["A"].width = 9
    ws.column_dimensions["B"].width = 100
    ws.column_dimensions["C"].width = 16
    ws.column_dimensions["D"].width = 40
    yellow = PatternFill("solid", fgColor="FFF2CC")
    for row in ws.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
        row[2].fill = yellow
        row[3].fill = yellow
    dv = DataValidation(type="list", formula1=f'"{",".join(LABELS)}"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"C2:C{len(items) + 1}")
    ws.freeze_panes = "A2"

    info = wb.create_sheet("Instructions")
    for line in [
        "Read each text and decide: is the writer's conclusion a cognitive distortion?",
        "label = 'distorted' if the conclusion goes beyond what the facts in the text support.",
        "label = 'not distorted' if the negative conclusion is supported by the facts given "
        "(calibrated pessimism counts as not distorted).",
        "Use the EmpowerLens relabelling codebook for definitions. Label every row; use notes if unsure.",
        "Work alone and do not ask the authors about any item until you have finished all of them.",
    ]:
        info.append([line])
    info.column_dimensions["A"].width = 120
    wb.save(out_dir / "annotator_sheet.xlsx")


# ---------------------------------------------------------------------------- leak

def run_leak(pairs: pd.DataFrame, out: Path = DIR / "leak_check.json") -> dict:
    f1 = leak_f1(pairs)
    per_file = {f: round(leak_f1(g), 4) for f, g in pairs.groupby("source_file")}
    res = {
        "check": "TF-IDF (1-2 grams, sublinear tf) + logistic regression, 5-fold stratified CV, seed 42",
        "target": "F1 of 'distorted' (justified vs distorted)",
        "n_items": int(2 * len(pairs)),
        "n_pairs": int(len(pairs)),
        "f1_distorted": round(f1, 4),
        "limit": LEAK_LIMIT,
        "verdict": "OK" if f1 <= LEAK_LIMIT else "TOO HIGH - the halves differ in wording, not reasoning",
        "per_file_f1": per_file,
    }
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


# ---------------------------------------------------------------------------- predict (Kaggle)

def predict(checkpoint: Path, items_path: Path, out: Path, device: str = "auto") -> pd.DataFrame:
    """Binary model -> P(distorted) and the decision at the checkpoint's D13 threshold."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.evaluate import predict_logits
    from src.fit_binary_threshold import prob_distorted
    from src.train_transformer import encode_texts, resolve_device

    meta = json.loads((checkpoint / "meta.json").read_text(encoding="utf-8"))
    if meta["task"] != "binary":
        raise SystemExit(f"{checkpoint} is a {meta['task']} checkpoint; D19 scores the binary model")
    thr = meta.get("binary_threshold")
    if thr is None:
        raise SystemExit("no binary_threshold in meta.json - run src.fit_binary_threshold first (D13)")
    dev = resolve_device(device)
    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint)).to(dev)
    items = pd.read_csv(items_path, encoding="utf-8")
    enc, _ = encode_texts(items["text"], tok, meta["max_length"], meta["truncation"],
                          meta.get("head_keep", 128))
    p = prob_distorted(predict_logits(model, enc, tok.pad_token_id, dev))
    df = pd.DataFrame({"item_id": items["item_id"], "prob_distorted": np.round(p, 6),
                       "pred": (p >= thr).astype(int)})
    df["seed"] = meta["seed"]
    df["threshold"] = thr
    df["model"] = meta["model"]
    df.to_csv(out, index=False, encoding="utf-8")
    return df


# ---------------------------------------------------------------------------- score

def _norm_label(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().lower().replace("-", " ").replace("_", " ")
    if s in ("distorted", "d", "yes", "1"):
        return "distorted"
    if s in ("not distorted", "no distortion", "justified", "n", "no", "0"):
        return "justified"
    raise ValueError(f"unrecognised annotator label {v!r}")


def load_annotations(path: Path) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    df = pd.read_excel(path, sheet_name="Label") if str(path).endswith(".xlsx") else pd.read_csv(path)
    df["annot"] = df["label"].map(_norm_label)
    if df["annot"].isna().all():
        return None
    return df[["item_id", "annot"]]


def annotator_agreement(key: pd.DataFrame, ann: pd.DataFrame) -> dict:
    m = key.merge(ann, on="item_id", how="left")
    labelled = m[m["annot"].notna()]
    agree = labelled["annot"] == labelled["intended"]
    return {
        "n_labelled": int(len(labelled)),
        "n_total": int(len(m)),
        "n_agree": int(agree.sum()),
        "disagree_pairs": sorted(labelled.loc[~agree, "pair_id"].unique().tolist()),
        "blank_items": sorted(m.loc[m["annot"].isna(), "item_id"].tolist()),
    }


def seed_counts(key: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Per distortion type: justified texts left unflagged, distorted texts flagged."""
    m = key.merge(preds[["item_id", "pred"]], on="item_id", how="inner")
    if len(m) != len(key):
        raise SystemExit(f"predictions cover {len(m)} of {len(key)} items")
    rows = []
    for t, g in m.groupby("distortion_type"):
        j, d = g[g["intended"] == "justified"], g[g["intended"] == "distorted"]
        rows.append({"distortion_type": t, "spec_k": int((j["pred"] == 0).sum()), "spec_n": len(j),
                     "sens_k": int((d["pred"] == 1).sum()), "sens_n": len(d)})
    return pd.DataFrame(rows)


def _ms(vals):
    vals = np.asarray(vals, dtype=float)
    return f"{vals.mean():.3f} ± {vals.std(ddof=0):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"


def render_results(key, leak, agreement, per_seed: dict) -> str:
    L = ["# D19 — Justified-concern challenge set: verification and model scores", ""]
    L += ["Pilot set: 50 matched pairs (100 texts), 5 pairs per distortion type, from "
          "`pairs_lumia.csv` and `pairs_izza.csv`. Evaluation only; the pairs are not edited.", ""]

    L += ["## 1. Annotator check (blind)", ""]
    if agreement is None:
        L += ["**Pending.** `annotator_sheet.xlsx` (100 shuffled texts, no pair, type or intended label) "
              "has not been returned by the psychology annotator yet.", ""]
    else:
        a = agreement
        L += [f"Agreement with the intended labels: **{a['n_agree']} of {a['n_labelled']}** labelled texts "
              f"({a['n_total'] - a['n_labelled']} left blank).", ""]
        L += ["Pairs with at least one disagreement: "
              + (", ".join(a["disagree_pairs"]) if a["disagree_pairs"] else "none") + ".", ""]

    L += ["## 2. Leak check", ""]
    L += [f"{leak['check']}. {leak['target']} on all {leak['n_items']} texts: "
          f"**{leak['f1_distorted']:.3f}** (limit {leak['limit']}): **{leak['verdict']}**.", ""]
    L += ["Per file: " + ", ".join(f"`{k}` {v:.3f}" for k, v in leak["per_file_f1"].items()) + ".", ""]
    L += ["Above 0.75 would mean the two halves of each pair differ in wording, not reasoning; "
          "those pairs would be rewritten and not scored.", ""]

    L += ["## 3. Model scores (D12 winner, binary, D13 threshold)", ""]
    if not per_seed:
        L += ["**Pending Kaggle run** (`notebooks/kaggle_runner_d19_challenge.ipynb`): one "
              "`predictions_<seed>.csv` per seed 42 / 1337 / 2024 in `data/challenge_set/`.", ""]
        return "\n".join(L) + "\n"

    seeds = sorted(per_seed)
    overall = []
    for s in seeds:
        c = per_seed[s]["counts"]
        sk, sn, ek, en = c["spec_k"].sum(), c["spec_n"].sum(), c["sens_k"].sum(), c["sens_n"].sum()
        overall.append({"seed": s, "spec": sk / sn, "fpr": 1 - sk / sn, "sens": ek / en,
                        "spec_str": f"{sk}/{sn}", "sens_str": f"{ek}/{en}",
                        "thr": per_seed[s]["threshold"]})
    L += [f"Model: `{per_seed[seeds[0]]['model']}`; seeds {', '.join(map(str, seeds))}. "
          "Specificity = justified texts left unflagged; sensitivity = distorted texts flagged.", ""]
    L += ["| | " + " | ".join(f"seed {o['seed']}" for o in overall) + " | mean ± std |",
          "|---|" + "---|" * (len(overall) + 1)]
    L += ["| Specificity (justified) | " + " | ".join(o["spec_str"] for o in overall)
          + f" | {_ms([o['spec'] for o in overall])} |"]
    L += ["| FPR | " + " | ".join(f"{o['fpr']:.3f}" for o in overall)
          + f" | {_ms([o['fpr'] for o in overall])} |"]
    L += ["| Sensitivity (distorted) | " + " | ".join(o["sens_str"] for o in overall)
          + f" | {_ms([o['sens'] for o in overall])} |"]
    L += ["| Threshold on P(distorted) | " + " | ".join(f"{o['thr']:.4f}" for o in overall) + " | |", ""]

    L += ["### Per distortion type (5 justified + 5 distorted texts each)", ""]
    L += ["| Type | " + " | ".join(f"spec s{s}" for s in seeds) + " | spec mean ± std | "
          + " | ".join(f"sens s{s}" for s in seeds) + " | sens mean ± std |",
          "|---|" + "---|" * (2 * len(seeds) + 2)]
    types = sorted(key["distortion_type"].unique())
    for t in types:
        rs = [per_seed[s]["counts"].set_index("distortion_type").loc[t] for s in seeds]
        L += [f"| {t} | " + " | ".join(f"{r.spec_k}/{r.spec_n}" for r in rs)
              + f" | {_ms([r.spec_k / r.spec_n for r in rs])} | "
              + " | ".join(f"{r.sens_k}/{r.sens_n}" for r in rs)
              + f" | {_ms([r.sens_k / r.sens_n for r in rs])} |"]
    L += ["", "With 5 texts per cell, one text moves a rate by 0.2; read per-type differences as "
          "indicative only.", ""]
    if agreement is not None and agreement["disagree_pairs"]:
        L += ["Note: pairs the annotator disagreed on are still included above; see section 1.", ""]
    return "\n".join(L) + "\n"


def score(out: Path = DIR / "results.md", annotations: Path | None = None) -> str:
    key = pd.read_csv(DIR / "annotator_key.csv", encoding="utf-8")
    leak_path = DIR / "leak_check.json"
    if not leak_path.exists():
        raise SystemExit("run `python -m src.challenge_set leak` first")
    leak = json.loads(leak_path.read_text(encoding="utf-8"))
    ann = load_annotations(annotations or DIR / "annotator_sheet_returned.xlsx")
    agreement = annotator_agreement(key, ann) if ann is not None else None
    per_seed = {}
    for s in SEEDS:
        p = DIR / f"predictions_{s}.csv"
        if p.exists():
            preds = pd.read_csv(p, encoding="utf-8")
            per_seed[s] = {"counts": seed_counts(key, preds), "threshold": float(preds["threshold"].iloc[0]),
                           "model": preds["model"].iloc[0]}
    text = render_results(key, leak, agreement, per_seed)
    out.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="D19 challenge-set tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sheet")
    sub.add_parser("leak")
    p = sub.add_parser("predict")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--items", default=str(DIR / "items.csv"))
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="auto")
    s = sub.add_parser("score")
    s.add_argument("--annotations", default=None,
                   help="returned sheet (xlsx with a 'Label' tab, or CSV with item_id,label); "
                        "default data/challenge_set/annotator_sheet_returned.xlsx")
    args = ap.parse_args(argv)

    if args.cmd == "sheet":
        items = build_items(load_pairs())
        write_sheet(items)
        print(f"wrote annotator_sheet.xlsx, annotator_key.csv, items.csv ({len(items)} items)")
    elif args.cmd == "leak":
        res = run_leak(load_pairs())
        print(json.dumps(res, indent=2))
    elif args.cmd == "predict":
        df = predict(Path(args.checkpoint), Path(args.items), Path(args.out), args.device)
        print(f"wrote {args.out}: {int(df['pred'].sum())} of {len(df)} flagged")
    elif args.cmd == "score":
        print(score(annotations=Path(args.annotations) if args.annotations else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
