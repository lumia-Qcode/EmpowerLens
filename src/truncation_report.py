"""D15 — truncation test: head vs head_tail, MentalRoBERTa multiclass, max length 256.

Rolls the six Kaggle runs (2 truncation options x 3 seeds) into one table of
**val** macro-F1, mean +/- std. Test is never read: this is a design ablation,
chosen on val like every other setting.

Expected layout (written by notebooks/kaggle_runner_truncation.ipynb):

    <results>/head/seed_42/meta.json         (+ epoch_history.csv)
    <results>/head_tail/seed_42/meta.json
    ...
    <results>/token_lengths.json             (share of texts cut at 256 / 512)

Run:
    venv\\Scripts\\python.exe -m src.truncation_report --results results/results_truncation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ARMS = ("head", "head_tail")
SEEDS = (42, 1337, 2024)
MAX_LENGTH = 256
EXPECTED = {"task": "multiclass", "splits": "data/splits", "max_length": MAX_LENGTH}


def token_lengths(texts, tokenizer) -> list[int]:
    """Token count of each text without special tokens (same count encode_texts uses)."""
    return [len(ids) for ids in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]


def cut_share(lengths, max_length: int, n_special: int = 2) -> float:
    """Share of texts longer than the text budget (max_length minus <s> and </s>).

    Mirrors encode_texts in src/train_transformer.py: a text is cut when its
    token count exceeds max_length - n_special.
    """
    lengths = list(lengths)
    if not lengths:
        raise ValueError("no lengths given")
    budget = max_length - n_special
    return sum(n > budget for n in lengths) / len(lengths)


def _load_meta(results: Path, arm: str, seed: int) -> dict:
    path = results / arm / f"seed_{seed}" / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"missing run: {path}")
    meta = json.loads(path.read_text(encoding="utf-8"))
    want = dict(EXPECTED, truncation=arm, seed=seed)
    bad = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
    if bad:
        raise ValueError(f"{path}: settings do not match D15 (got, expected): {bad}")
    return meta


def summarize(results) -> pd.DataFrame:
    """Per-seed val metrics for both arms, plus one mean row (with std) per arm."""
    results = Path(results)
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            m = _load_meta(results, arm, seed)
            vm = m["val_metrics"]
            rows.append({
                "arm": arm, "seed": str(seed),
                "macro_f1": vm["eval_macro_f1"], "macro_f1_10": vm["eval_macro_f1_10"],
                "best_epoch": m.get("best_epoch"), "val_truncation_rate": m.get("val_truncation_rate"),
            })
    df = pd.DataFrame(rows)
    means = []
    for arm in ARMS:
        g = df[df.arm == arm]
        means.append({
            "arm": arm, "seed": "mean",
            "macro_f1": g.macro_f1.mean(), "macro_f1_std": g.macro_f1.std(ddof=1),
            "macro_f1_10": g.macro_f1_10.mean(), "macro_f1_10_std": g.macro_f1_10.std(ddof=1),
            "best_epoch": g.best_epoch.astype(float).mean(),
            "val_truncation_rate": g.val_truncation_rate.astype(float).mean(),
        })
    return pd.concat([df, pd.DataFrame(means)], ignore_index=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", default="results/results_truncation")
    args = ap.parse_args(argv)
    results = Path(args.results)

    df = summarize(results)
    out = results / "summary.csv"
    df.round(4).to_csv(out, index=False)
    print(df.round(3).to_string(index=False))
    print(f"\nwrote {out}")

    lengths_file = results / "token_lengths.json"
    if lengths_file.exists():
        tl = json.loads(lengths_file.read_text(encoding="utf-8"))
        print("\nshare of texts cut:")
        for part, d in tl.items():
            print(f"  {part:8s} n={d['n']:5d}  median={d['median']:.0f}  "
                  f"cut@256={d['cut_256']:.1%}  cut@512={d['cut_512']:.1%}")
    else:
        print(f"\n[warn] {lengths_file} not found - cut shares not reported")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
