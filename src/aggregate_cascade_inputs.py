"""
D17 - roll the cascade-input eval files into one summary table.

RUN (from repo root, after copying the Kaggle output into results/results_cascade_inputs/):
    venv\\Scripts\\python.exe -m src.aggregate_cascade_inputs

Reads   results/results_cascade_inputs/<row>/eval_*_multilabel_<seed>.json   (written by src.evaluate)
        rows: full, crop_sel, crop_gold, marked_sel, marked_gold; seeds 42, 1337, 2024
Writes  results/results_cascade_inputs/summary.md and summary.csv
Never opens test.csv: every number comes from the eval JSONs. Fails if any of the 15 files is missing.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

RES = Path("results/results_cascade_inputs")
SEEDS = (42, 1337, 2024)
ROWS = [  # (folder, label in the table)
    ("full", "full text"),
    ("crop_sel", "selected sentences only - trained selector"),
    ("crop_gold", "selected sentences only - gold span (oracle)"),
    ("marked_sel", "full text, sentences marked - trained selector"),
    ("marked_gold", "full text, sentences marked - gold span (oracle)"),
]
COLS = {"val_macro_f1": ("val", "macro_f1"), "test_macro_f1": ("test", "macro_f1"),
        "test_specificity_per_label": ("test", "macro_specificity_10")}


def load(res: Path):
    recs = []
    for folder, _ in ROWS:
        for seed in SEEDS:
            hits = [p for p in (res / folder).glob(f"eval_*_multilabel_{seed}.json")
                    if re.search(rf"_multilabel_{seed}\.json$", p.name)]
            if len(hits) != 1:
                raise SystemExit(f"expected one eval file for {folder} seed {seed}, found {len(hits)} in {res / folder}")
            ev = json.loads(hits[0].read_text(encoding="utf-8"))
            rec = {"row": folder, "seed": seed, "file": f"{folder}/{hits[0].name}"}
            for col, (split, key) in COLS.items():
                s = ev["splits"][split]
                val = s.get(key, s["metrics"].get(key))
                if val is None:
                    raise SystemExit(f"{hits[0]} has no {split} {key} (was it evaluated with the D17 evaluate.py?)")
                rec[col] = float(val)
            recs.append(rec)
    return pd.DataFrame(recs)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for folder, label in ROWS:
        sub = df[df["row"] == folder]
        row = {"row": folder, "input": label, "n_seeds": len(sub)}
        for col in COLS:
            row[col] = sub[col].mean()
            row[f"{col}_std"] = sub[col].std(ddof=1)
        out.append(row)
    return pd.DataFrame(out)


def cost(df: pd.DataFrame, kind: str, col: str):
    """gold - selector, paired by seed; mean and std over the 3 seeds."""
    g = df[df["row"] == f"{kind}_gold"].set_index("seed")[col]
    s = df[df["row"] == f"{kind}_sel"].set_index("seed")[col]
    d = (g - s).loc[list(SEEDS)]
    return d.mean(), d.std(ddof=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RES))
    res = Path(ap.parse_args().results)
    per_seed = load(res)
    tab = summarise(per_seed)
    tab.round(4).to_csv(res / "summary.csv", index=False, encoding="utf-8")
    per_seed.round(4).to_csv(res / "summary_per_seed.csv", index=False, encoding="utf-8")

    def ms(r, c):
        return f"{r[c]:.3f} ± {r[c + '_std']:.3f}"

    lines = [
        "# D17 - Cascade inputs and the gap to the human answer", "",
        "**Question.** Does pointing the classifier at the distorted part help, and how much of that gain "
        "survives when a trained selector, not a human, chooses the sentences?", "",
        "**Setup.** MentalRoBERTa (`mental/mental-roberta-base`), multilabel over the 10 distortions, on the "
        "**distorted rows** of the frozen split, minus the 13 rows whose quoted span could not be located "
        "(train 1,268 / val 157 / test 159; the same rows in every version). Recipe: max length 512, batch 16, "
        "up to 12 epochs with early stopping (patience 3), `--deterministic`, seeds 42 / 1337 / 2024; per-label "
        "thresholds tuned on val at training time. Inputs are built by `src/make_splits_cascade_inputs.py` "
        "(`data/splits_cascade_inputs/`).", "",
        "- **Selected sentences only**: the chosen sentences, in text order. **Marked**: the full text with each "
        "run of chosen sentences wrapped in `@ ... @`.",
        "- **Gold span** = the sentences the human annotator quoted (T12 sentence labels). **Trained selector** = "
        "the D03 selector's picks for the same seed (rows where it picked nothing fall back to its top-scoring "
        "sentence; counts in `data/splits_cascade_inputs/manifest.json`).",
        "- Each span classifier is **trained once on gold sentences** and evaluated twice: with gold and with "
        "selector sentences (the real cascade at use time). The selector rows reuse the thresholds tuned on "
        "gold-input val.", "",
        "## Results (mean ± std over 3 seeds, std with n-1)", "",
        "| Input | Val macro-F1 | Test macro-F1 | Test specificity (per-label) |",
        "| --- | --- | --- | --- |",
        *[f"| {r['input']} | {ms(r, 'val_macro_f1')} | {ms(r, 'test_macro_f1')} | "
          f"{ms(r, 'test_specificity_per_label')} |" for _, r in tab.iterrows()],
        "",
        "Macro-F1 is over the 10 distortion labels. **Specificity here is per-label**: for each label, the share "
        "of rows without that label that the model leaves unflagged, averaged over the 10 labels. It is *not* the "
        "binary headline specificity: this set has no No-Distortion rows, because the span versions only exist "
        "for distorted texts.", "",
        "## Cost of automatic selection (gold span - trained selector, paired by seed)", "",
        "| Version | Val macro-F1 | Test macro-F1 |", "| --- | --- | --- |",
    ]
    for kind, name in (("crop", "selected sentences only"), ("marked", "marked full text")):
        v, t = cost(per_seed, kind, "val_macro_f1"), cost(per_seed, kind, "test_macro_f1")
        lines.append(f"| {name} | {v[0]:+.3f} ± {v[1]:.3f} | {t[0]:+.3f} ± {t[1]:.3f} |")
    full_t = tab.set_index("row").loc["full", "test_macro_f1"]
    best = tab.set_index("row").drop(index=["crop_gold", "marked_gold"])["test_macro_f1"].idxmax()
    lines += [
        "",
        f"Best non-oracle input on test: **{dict(ROWS)[best]}** "
        f"({tab.set_index('row').loc[best, 'test_macro_f1']:.3f} vs full text {full_t:.3f}).", "",
        "**Reference (different model).** The earlier oracle run with TF-IDF + logistic regression gave val "
        "macro-F1 0.402 on the gold span vs 0.322 on full text (`docs/FYP_PLAN.md` section 3.1, distorted-only, "
        "10 labels). It is a ceiling from a different model, not a row of this table.", "",
        "## Files", "",
        "Every number above comes from the eval files in this folder (written by `src/evaluate.py`):", "",
        *[f"- `{f}`" for f in per_seed["file"]], "",
        "`summary.csv` (table), `summary_per_seed.csv` (one row per input x seed). "
        "Script: `python -m src.aggregate_cascade_inputs`.", "",
    ]
    (res / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(tab[["row", "val_macro_f1", "test_macro_f1", "test_specificity_per_label"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
