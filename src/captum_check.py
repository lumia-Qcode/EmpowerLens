"""
D22 - Captum cross-check of the D12 multilabel classifier against the D03 sentence selector.

Two internal checks on VAL rows that contain a distortion (test.csv is never read):

1. Agreement: in what share of rows is the selector's top-scoring sentence also the
   sentence with the highest total Captum (LayerIntegratedGradients) score?
2. Cue or topic: 20 val distorted rows containing absolute words, each shown with its
   top 5 words by Captum score, for a human verdict (cue / topic / mixed).

INTERNAL CHECK ONLY. Captum scores are never shown to a trainer.

Run (attribution needs a GPU in practice; use notebooks/kaggle_runner_d22_captum.ipynb):

    python -m src.captum_check attribute \
        --checkpoint checkpoints/multilabel_mental-roberta-base_42 \
        --out results/results_explain/attributions_val.csv
    python -m src.captum_check report

Local (Windows):
    venv\\Scripts\\python.exe -m src.captum_check report

`report` without attributions writes a scaffold md ("pending Kaggle run") that already
lists the 20 selected rows. Re-running `report` keeps the human's verdicts (from
captum_check_rows.csv) and summary paragraph (between the summary markers in the md).

Design choices
--------------
- Attributed label: the top-scoring distortion (argmax of the 10 multilabel logits) for
  each row. Every distorted row gets a target this way, even when no label clears its
  val threshold; whether it cleared is recorded (`above_threshold`).
- Encoding mirrors src/train_transformer.encode_texts: tokenise without special tokens,
  truncate (head, or head_tail if the checkpoint's meta.json says so) to max_length minus
  the special tokens, then add <s> ... </s> by hand (transformers 5.x).
- LayerIntegratedGradients over model.roberta.embeddings (as src/explain.py), baseline =
  same length, <s>/</s> kept, every other position <pad> (Captum BERT tutorial), 50 steps.
  Token score = attribution summed over the embedding dimension (signed).
- Alignment: character offsets. Each token's (start, end) from the fast tokenizer is
  assigned to the sentence whose [sent_start, sent_end) holds its start; sentence
  offsets come from results/results_sentence_label/sentence_labels_val.csv, the exact
  sentences (same `sent_idx`, same text, checked) in the selector's predictions file.
  Sentence total = signed sum of its tokens' scores.
- Words: regex word spans in the original text; word score = sum of its tokens' scores.
  "Top 5 words" = the 5 highest signed word scores (pushing towards the label).
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS_DIR = Path("data/splits")
SENT_LABELS = Path("results/results_sentence_label/sentence_labels_val.csv")
SELECTOR_PRED = Path("results/results_span_selector/predictions_val_42.csv")
OUT_DIR = Path("results/results_explain")
ATTR_CSV = OUT_DIR / "attributions_val.csv"
REPORT_MD = OUT_DIR / "captum_check.md"
ROWS_CSV = OUT_DIR / "captum_check_rows.csv"

ID_COL = "Id_Number"
TEXT_COL = "Patient Question"

# Absolute words for check 2 (whole-word, case-insensitive match on the full text).
ABSOLUTE_WORDS = ["never", "always", "everyone", "nobody", "everything",
                  "nothing", "completely", "all", "every", "totally"]
N_CHECK_ROWS = 20
SELECTION_SEED = 42
VERDICTS = ("cue", "topic", "mixed")

WORD_RE = re.compile(r"\w+(?:['’]\w+)*")
SUMMARY_START = "<!-- summary:start -->"
SUMMARY_END = "<!-- summary:end -->"
SUMMARY_PLACEHOLDER = "_To be written by the reviewer after filling the verdicts._"


# ---------------------------------------------------------------- pure helpers
def sibling(path: Path, suffix: str) -> Path:
    """attributions_val.csv -> attributions_val_<suffix>.csv"""
    path = Path(path)
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def word_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group(0)) for m in WORD_RE.finditer(text)]


def absolute_words_in(text: str) -> list[str]:
    found = {w.lower() for _, _, w in word_spans(text)}
    return [w for w in ABSOLUTE_WORDS if w in found]


def assign_sentence(tok_start: int, sentences: pd.DataFrame) -> int:
    """sent_idx whose [sent_start, sent_end) holds tok_start; a token in an inter-sentence
    gap goes to the preceding sentence (or the first one). -1 if there are no sentences."""
    if sentences.empty:
        return -1
    starts = sentences["sent_start"].to_numpy()
    ends = sentences["sent_end"].to_numpy()
    idx = sentences["sent_idx"].to_numpy()
    inside = np.where((starts <= tok_start) & (tok_start < ends))[0]
    if len(inside):
        return int(idx[inside[0]])
    before = np.where(starts <= tok_start)[0]
    return int(idx[before[-1]]) if len(before) else int(idx[0])


def aggregate_row(row_id, text: str, offsets, scores, sentences: pd.DataFrame):
    """Map token scores of one row to words and sentences.

    offsets: (start, end) per token or None for special tokens; scores: same length.
    sentences: this row's sentences (sent_idx, sent_start, sent_end, sentence).
    Returns (word_df, sent_df).
    """
    sentences = sentences.sort_values("sent_idx").reset_index(drop=True)
    toks = [(s, e, float(sc)) for off, sc in zip(offsets, scores) if off is not None
            for s, e in [off] if e > s]

    words = word_spans(text)
    w_scores = np.zeros(len(words))
    w_starts = np.array([w[0] for w in words]) if words else np.array([], dtype=int)
    w_ends = np.array([w[1] for w in words]) if words else np.array([], dtype=int)
    for s, e, sc in toks:
        hit = np.where((w_starts < e) & (s < w_ends))[0]
        if len(hit):                      # a token spanning 2 words (rare) counts for the first
            w_scores[hit[0]] += sc

    word_rows = [{ID_COL: row_id, "word_idx": i, "word": w, "char_start": s, "char_end": e,
                  "sent_idx": assign_sentence(s, sentences), "attr": w_scores[i]}
                 for i, (s, e, w) in enumerate(words)]
    word_df = pd.DataFrame(word_rows, columns=[ID_COL, "word_idx", "word", "char_start",
                                               "char_end", "sent_idx", "attr"])

    tot = {int(i): 0.0 for i in sentences["sent_idx"]}
    ab = dict(tot)
    ntok = {k: 0 for k in tot}
    for s, _, sc in toks:
        k = assign_sentence(s, sentences)
        if k < 0:
            continue
        tot[k] += sc
        ab[k] += abs(sc)
        ntok[k] += 1
    sent_df = pd.DataFrame({
        ID_COL: row_id,
        "sent_idx": sentences["sent_idx"].astype(int),
        "sentence": sentences["sentence"],
        "attr_total": [tot[int(i)] for i in sentences["sent_idx"]],
        "attr_abs_total": [ab[int(i)] for i in sentences["sent_idx"]],
        "n_tokens": [ntok[int(i)] for i in sentences["sent_idx"]],
    })
    return word_df, sent_df


def top_words(word_df: pd.DataFrame, k: int = 5) -> list[str]:
    """k highest signed word scores (stable order on ties: earlier word first)."""
    if word_df.empty:
        return []
    w = word_df.sort_values(["attr", "word_idx"], ascending=[False, True], kind="mergesort")
    return w["word"].head(k).tolist()


def top_sentence(df: pd.DataFrame, score_col: str) -> pd.Series:
    """Per Id_Number, sent_idx with the highest score (ties -> lowest sent_idx)."""
    d = df.sort_values([ID_COL, score_col, "sent_idx"], ascending=[True, False, True], kind="mergesort")
    return d.groupby(ID_COL)["sent_idx"].first()


def agreement(selector: pd.DataFrame, attributions: pd.DataFrame) -> dict:
    """Share of rows (present in both) where the selector's top-scoring sentence is also the
    sentence with the highest total Captum score. Also the chance rate (mean 1/n_sentences)."""
    sel_top = top_sentence(selector, "score")
    cap_top = top_sentence(attributions, "attr_total")
    ids = sorted(set(sel_top.index) & set(cap_top.index))
    agree = sum(int(sel_top[i] == cap_top[i]) for i in ids)
    n_sent = selector.groupby(ID_COL)["sent_idx"].nunique()
    chance = float(np.mean([1.0 / n_sent[i] for i in ids])) if ids else float("nan")
    return {"agree": agree, "n": len(ids), "share": agree / len(ids) if ids else float("nan"),
            "chance": chance, "ids": ids,
            "per_row": pd.DataFrame({ID_COL: ids, "selector_top": [int(sel_top[i]) for i in ids],
                                     "captum_top": [int(cap_top[i]) for i in ids]})}


def fmt_agreement(a: dict) -> str:
    return f"{a['agree']} of {a['n']} rows ({100 * a['share']:.1f}%)"


def select_check_rows(val: pd.DataFrame, n: int = N_CHECK_ROWS, seed: int = SELECTION_SEED) -> pd.DataFrame:
    """Rule: val rows with y_bin == 1 whose text contains >= 1 ABSOLUTE_WORDS (whole word,
    any case); sort eligible rows by Id_Number; draw n without replacement with
    numpy default_rng(seed); return them sorted by Id_Number. All eligible if fewer than n."""
    d = val[val["y_bin"] == 1].copy()
    d["absolute_words"] = d[TEXT_COL].astype(str).map(absolute_words_in)
    d = d[d["absolute_words"].map(len) > 0].sort_values(ID_COL, kind="mergesort").reset_index(drop=True)
    if len(d) > n:
        pick = np.random.default_rng(seed).choice(len(d), size=n, replace=False)
        d = d.iloc[np.sort(pick)]
    return d.sort_values(ID_COL, kind="mergesort").reset_index(drop=True)


# ------------------------------------------------------------- model-dependent
def encode_with_offsets(text: str, tokenizer, max_length: int, truncation: str = "head",
                        head_keep: int = 128):
    """Same token ids as train_transformer.encode_texts, plus char offsets (None for specials)."""
    from src.train_transformer import _special_token_ids
    prefix, suffix = _special_token_ids(tokenizer)
    budget = max_length - len(prefix) - len(suffix)
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    truncated = len(ids) > budget
    if truncated:
        if truncation == "head_tail" and budget > head_keep:
            keep = list(range(head_keep)) + list(range(len(ids) - (budget - head_keep), len(ids)))
        else:
            keep = list(range(budget))
        ids, offs = [ids[i] for i in keep], [offs[i] for i in keep]
    return (prefix + ids + suffix, [None] * len(prefix) + offs + [None] * len(suffix),
            truncated, len(prefix), len(suffix))


def cmd_attribute(args) -> int:
    import torch
    from captum.attr import LayerIntegratedGradients
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.data import DISTORTIONS

    ck = Path(args.checkpoint)
    meta = json.loads((ck / "meta.json").read_text(encoding="utf-8")) if (ck / "meta.json").exists() else {}
    if meta.get("task", "multilabel") != "multilabel":
        raise SystemExit(f"{ck} is a {meta['task']} checkpoint; D22 needs the multilabel one.")
    max_length = args.max_length or meta.get("max_length", 512)
    truncation = meta.get("truncation", "head")
    head_keep = meta.get("head_keep", 128)
    thresholds = meta.get("thresholds") or [0.5] * len(DISTORTIONS)

    val = pd.read_csv(SPLITS_DIR / "val.csv", encoding="utf-8-sig")
    val = val[val["y_bin"] == 1].sort_values(ID_COL).reset_index(drop=True)
    if args.limit:
        val = val.head(args.limit)
    sents = pd.read_csv(SENT_LABELS, encoding="utf-8").rename(columns={"id": ID_COL})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(ck))
    if not getattr(tok, "is_fast", False):
        raise SystemExit("Need a fast tokenizer for character offsets.")
    model = AutoModelForSequenceClassification.from_pretrained(str(ck)).to(device).eval()
    if model.config.num_labels != len(DISTORTIONS):
        raise SystemExit(f"Expected {len(DISTORTIONS)} labels, checkpoint has {model.config.num_labels}.")

    def forward(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    lig = LayerIntegratedGradients(forward, model.roberta.embeddings)
    pad_id = tok.pad_token_id

    words_all, sents_all, rows = [], [], []
    for n, (_, r) in enumerate(val.iterrows()):
        rid, text = int(r[ID_COL]), str(r[TEXT_COL])
        ids, offs, truncated, n_pre, n_suf = encode_with_offsets(text, tok, max_length, truncation, head_keep)
        x = torch.tensor([ids], device=device)
        mask = torch.ones_like(x)
        base = torch.full_like(x, pad_id)
        base[0, :n_pre] = x[0, :n_pre]
        if n_suf:
            base[0, -n_suf:] = x[0, -n_suf:]
        with torch.no_grad():
            probs = torch.sigmoid(forward(x, mask))[0].float().cpu().numpy()
        target = int(np.argmax(probs))
        attr, delta = lig.attribute(inputs=x, baselines=base, target=target,
                                    additional_forward_args=(mask,), n_steps=args.n_steps,
                                    internal_batch_size=args.internal_batch_size,
                                    return_convergence_delta=True)
        scores = attr.sum(dim=-1)[0].detach().float().cpu().numpy()

        w_df, s_df = aggregate_row(rid, text, offs, scores, sents[sents[ID_COL] == rid])
        s_df["target_label"] = DISTORTIONS[target]
        words_all.append(w_df)
        sents_all.append(s_df)
        gold = [d for d in DISTORTIONS if r.get(f"ml_{d}", 0) == 1]
        rows.append({ID_COL: rid, "target_label": DISTORTIONS[target], "target_prob": float(probs[target]),
                     "above_threshold": bool(probs[target] >= thresholds[target]),
                     "target_is_gold": DISTORTIONS[target] in gold, "gold_labels": "|".join(gold),
                     "n_tokens": len(ids), "truncated": truncated, "n_sentences": len(s_df),
                     "convergence_delta": float(delta.abs().max()),
                     "top_words": " | ".join(top_words(w_df, 5))})
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(val)} rows", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(sents_all, ignore_index=True).to_csv(out, index=False, encoding="utf-8")
    pd.concat(words_all, ignore_index=True).to_csv(sibling(out, "words"), index=False, encoding="utf-8")
    pd.DataFrame(rows).to_csv(sibling(out, "rows"), index=False, encoding="utf-8")
    cfg = {"checkpoint": str(ck), "model": meta.get("model"), "seed": meta.get("seed"),
           "max_length": max_length, "truncation": truncation, "n_steps": args.n_steps,
           "layer": "model.roberta.embeddings", "baseline": "<pad> with <s>/</s> kept",
           "target": "argmax of the 10 multilabel logits", "split": "val, y_bin == 1",
           "n_rows": len(rows), "limit": args.limit}
    sibling(out, "config").with_suffix(".json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Wrote {out}, {sibling(out, 'words')}, {sibling(out, 'rows')} ({len(rows)} rows)")
    return 0


# ---------------------------------------------------------------------- report
def _existing_summary(md_path: Path) -> str:
    if not md_path.exists():
        return SUMMARY_PLACEHOLDER
    t = md_path.read_text(encoding="utf-8")
    if SUMMARY_START in t and SUMMARY_END in t:
        s = t.split(SUMMARY_START, 1)[1].split(SUMMARY_END, 1)[0].strip()
        return s or SUMMARY_PLACEHOLDER
    return SUMMARY_PLACEHOLDER


def _existing_verdicts(rows_csv: Path) -> dict:
    if not rows_csv.exists():
        return {}
    d = pd.read_csv(rows_csv, encoding="utf-8", dtype={"verdict": str}).fillna({"verdict": ""})
    return {int(i): str(v).strip() for i, v in zip(d[ID_COL], d["verdict"]) if str(v).strip()}


def _md_verdicts(md_path: Path) -> dict:
    """Verdicts typed straight into the md table (last cell of rows starting with an Id)."""
    if not md_path.exists():
        return {}
    out = {}
    for line in md_path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) >= 5 and cells[0].isdigit() and cells[-1]:
            out[int(cells[0])] = cells[-1]
    return out


def build_check_table(val: pd.DataFrame, row_attr: pd.DataFrame | None, verdicts: dict) -> pd.DataFrame:
    sel = select_check_rows(val)
    ra = row_attr.set_index(ID_COL) if row_attr is not None else None
    out = []
    for r in sel.itertuples(index=False):
        rid = int(getattr(r, ID_COL))
        has = ra is not None and rid in ra.index
        out.append({ID_COL: rid,
                    "absolute_words": ", ".join(r.absolute_words),
                    "attributed_label": ra.loc[rid, "target_label"] if has else "",
                    "top5_words": ra.loc[rid, "top_words"] if has else "",
                    "verdict": verdicts.get(rid, "")})
    return pd.DataFrame(out)


def render_md(table: pd.DataFrame, agr: dict | None, summary: str, cfg: dict | None) -> str:
    pending = agr is None
    L = ["# D22 — Captum cross-check (val)", "",
         "> **Internal check only.** Captum scores are never shown to a trainer.", ""]
    if pending:
        L += ["**Status: pending Kaggle run.** Run `notebooks/kaggle_runner_d22_captum.ipynb`, copy its "
              "`results_explain/` files into `results/results_explain/`, then "
              "`venv\\Scripts\\python.exe -m src.captum_check report` to fill this page.", ""]
    L += ["## Setup", "",
          "- Model: D12 winner, MentalRoBERTa multilabel, seed 42 (D12 recipe: lr 3e-5, 5 epochs, weight decay 0.1, "
          "max length 512, batch 16)" + (f"; checkpoint `{cfg.get('checkpoint')}`." if cfg else "."),
          "- Method: LayerIntegratedGradients over `model.roberta.embeddings` (as `src/explain.py`), 50 steps, "
          "baseline = `<pad>` tokens with `<s>`/`</s>` kept. Token score = attribution summed over the embedding.",
          "- Label explained: the top-scoring distortion of each row (argmax of the 10 logits).",
          "- Rows: val rows with a distortion (`y_bin == 1`). Test is not used.",
          "- Sentences: the selector's own sentences (`results/results_span_selector/predictions_val_42.csv`), "
          "aligned by `Id_Number` + `sent_idx` using character offsets from "
          "`results/results_sentence_label/sentence_labels_val.csv`. Sentence score = signed sum of its tokens.", "",
          "## 1. Agreement with the D03 selector (seed 42)", ""]
    if pending:
        L += ["Agreement: **pending** (X of N rows (Y%)).", ""]
    else:
        L += [f"Agreement: **{fmt_agreement(agr)}** — the selector's top-scoring sentence is also the sentence "
              f"with the highest total Captum score.", "",
              f"For reference, picking a sentence at random would agree in {100 * agr['chance']:.1f}% of these rows "
              f"(mean of 1 / number of sentences). Rows are those in both files; the selector file excludes the "
              f"T12 rows whose quote could not be located.", ""]
    L += ["## 2. Cue or topic? (20 rows with absolute words)", "",
          f"Selection rule: val rows with `y_bin == 1` whose text contains at least one of "
          f"{', '.join('`' + w + '`' for w in ABSOLUTE_WORDS)} (whole word, any case); eligible rows sorted by "
          f"`Id_Number`, {N_CHECK_ROWS} drawn without replacement with `numpy.random.default_rng({SELECTION_SEED})`, "
          f"shown sorted by `Id_Number`. Top 5 words = highest signed Captum score towards the attributed label.", "",
          "Verdict: **cue** = top words are the distortion's wording (absolute words, \"should\", labels, ...); "
          "**topic** = top words are the subject (people, jobs, bodies, ...); **mixed** = both.", "",
          "| Id_Number | Absolute words | Attributed label | Top 5 words (Captum) | Verdict |",
          "|---|---|---|---|---|"]
    for r in table.itertuples(index=False):
        L.append(f"| {r.Id_Number} | {r.absolute_words} | {r.attributed_label or 'pending'} | "
                 f"{(r.top5_words or 'pending').replace('|', '/')} | {r.verdict} |")
    counts = table["verdict"].str.lower().value_counts()
    L += ["", "Verdict counts: " + ", ".join(f"{v} {int(counts.get(v, 0))}" for v in VERDICTS)
          + f", blank {int((table['verdict'] == '').sum())}.", "",
          "## Summary", "", SUMMARY_START, summary, SUMMARY_END, "",
          "## Files", "",
          "`attributions_val.csv` (per sentence), `attributions_val_words.csv` (per word), "
          "`attributions_val_rows.csv` (per row: label, probability, top words), `attributions_val_config.json`, "
          "`captum_check_rows.csv` (the 20-row table). Fill the verdicts in that CSV or in the table above, and the "
          "summary between the markers; re-running `report` keeps both (the CSV wins if they differ).", ""]
    return "\n".join(L)


def cmd_report(args) -> int:
    attr_path = Path(args.attributions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, rows_path = out_dir / REPORT_MD.name, out_dir / ROWS_CSV.name

    val = pd.read_csv(SPLITS_DIR / "val.csv", encoding="utf-8-sig")
    verdicts = {**_md_verdicts(md_path), **_existing_verdicts(rows_path)}   # CSV wins on conflict
    summary = _existing_summary(md_path)

    agr, row_attr, cfg = None, None, None
    if attr_path.exists():
        attributions = pd.read_csv(attr_path, encoding="utf-8")
        selector = pd.read_csv(args.selector, encoding="utf-8")
        row_attr = pd.read_csv(sibling(attr_path, "rows"), encoding="utf-8")
        cfg_path = sibling(attr_path, "config").with_suffix(".json")
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None
        agr = agreement(selector, attributions)
        agr["per_row"].to_csv(out_dir / "agreement_per_row.csv", index=False, encoding="utf-8")
        print("Agreement:", fmt_agreement(agr))
    else:
        print(f"{attr_path} not found -> writing the pending scaffold.")

    table = build_check_table(val, row_attr, verdicts)
    table.to_csv(rows_path, index=False, encoding="utf-8")
    md_path.write_text(render_md(table, agr, summary, cfg), encoding="utf-8")
    print(f"Wrote {md_path} and {rows_path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("attribute", help="Captum attributions for distorted val rows")
    a.add_argument("--checkpoint", required=True, help="multilabel checkpoint dir (with meta.json)")
    a.add_argument("--out", default=str(ATTR_CSV))
    a.add_argument("--max-length", type=int, default=None, help="default: the checkpoint's meta.json (512)")
    a.add_argument("--n-steps", type=int, default=50)
    a.add_argument("--internal-batch-size", type=int, default=16)
    a.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    r = sub.add_parser("report", help="agreement + 20-row table -> captum_check.md")
    r.add_argument("--attributions", default=str(ATTR_CSV))
    r.add_argument("--selector", default=str(SELECTOR_PRED))
    r.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    return cmd_attribute(args) if args.cmd == "attribute" else cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
