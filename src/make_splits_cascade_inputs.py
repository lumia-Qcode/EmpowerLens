"""
D17 - input versions for the cascade-input experiment (no re-split, no training).

RUN (from repo root):
    venv\\Scripts\\python.exe -m src.make_splits_cascade_inputs [--force]

Builds data/splits_cascade_inputs/<variant>/{train,val,test}.csv + manifest.json. Every variant
holds the SAME rows: the distorted rows of the frozen data/splits, minus the 13 rows whose quoted
span could not be located (they have no sentence labels; train 1,268 / val 157 / test 159). The
prepared model input is written into `Patient Question`, so src.train_transformer and
src.evaluate run unchanged (default --input-repr document).

Variants
    full                 the whole text
    crop_gold            only the gold sentences (T12 label = 1), in text order
    marked_gold          the whole text, each gold sentence wrapped as "@ ... @"
    crop_sel_<seed>      val/test: only the sentences the D03 selector (that seed) picked
    marked_sel_<seed>    val/test: the whole text with those sentences marked
For the selector variants train.csv is a copy of the gold variant's train (never trained on:
the classifier is trained on gold and evaluated with the selector's picks). A row where the
selector picked nothing falls back to its single highest-scoring sentence (counted in manifest).
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import CSV_ENCODING, DISTORTIONS, ID_COL, TEXT_COL

SRC = Path("data/splits")
OUT = Path("data/splits_cascade_inputs")
LABELS = Path("results/results_sentence_label")
SELECTOR = Path("results/results_span_selector")
SPLITS = ("train", "val", "test")
SEEDS = (42, 1337, 2024)
MARK = "@"   # same marker as src.train_transformer.SPAN_MARKER
KEEP = [ID_COL, TEXT_COL, "y_bin", "y_mc"] + [f"ml_{d}" for d in DISTORTIONS]


def crop(sents: pd.DataFrame, pick: np.ndarray) -> str:
    """Picked sentences of one row, in text order, joined by a space."""
    return " ".join(sents.loc[pick, "sentence"].astype(str).str.strip())


def mark(doc: str, sents: pd.DataFrame, pick: np.ndarray) -> str:
    """Wrap each run of consecutive picked sentences of `doc` in one pair of markers
    (like the single gold span src.train_transformer's span_marked wraps), using offsets."""
    runs, prev = [], None
    for i in np.flatnonzero(pick):
        s, e = int(sents.at[i, "sent_start"]), int(sents.at[i, "sent_end"])
        if prev is not None and i == prev + 1:
            runs[-1][1] = e
        else:
            runs.append([s, e])
        prev = i
    out, pos = [], 0
    for s, e in runs:
        out += [doc[pos:s], f"{MARK} {doc[s:e]} {MARK}"]
        pos = e
    return "".join(out + [doc[pos:]])


def selector_pick(sents: pd.DataFrame) -> tuple[np.ndarray, bool]:
    """sents has `selected` (yes/no) and `score`. Returns (mask, fell_back_to_top1)."""
    pick = (sents["selected"] == "yes").to_numpy()
    if pick.any():
        return pick, False
    top = np.zeros(len(sents), bool)
    top[int(np.argmax(sents["score"].to_numpy()))] = True
    return top, True


def build_texts(base: pd.DataFrame, sents_all: pd.DataFrame, pick_col: str | None, kind: str):
    """kind = crop | marked; pick_col = 'label' (gold) or None (selector: needs selected/score)."""
    texts, fallbacks = [], 0
    for rid, doc in zip(base[ID_COL], base[TEXT_COL]):
        sents = sents_all[sents_all[ID_COL] == rid].sort_values("sent_idx").reset_index(drop=True)
        assert all(doc[s:e] == t for s, e, t in sents[["sent_start", "sent_end", "sentence"]].to_numpy()), rid
        if pick_col:
            pick = (sents[pick_col] == 1).to_numpy()
        else:
            pick, fb = selector_pick(sents)
            fallbacks += fb
        assert pick.any(), rid
        texts.append(crop(sents, pick) if kind == "crop" else mark(doc, sents, pick))
    return texts, fallbacks


def _write(df, variant, split):
    d = OUT / variant
    d.mkdir(parents=True, exist_ok=True)
    df[KEEP].to_csv(d / f"{split}.csv", index=False, encoding="utf-8")


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help=f"overwrite {OUT}")
    args = ap.parse_args()
    if OUT.exists() and any(OUT.iterdir()):
        if not args.force:
            raise SystemExit(f"{OUT} already exists; pass --force to rebuild it.")
        shutil.rmtree(OUT)

    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "task": "D17 cascade inputs",
                "marker": MARK, "rows": {}, "dropped_unlocated": {}, "selector_top1_fallbacks": {},
                "sources": {}}
    bases, labels = {}, {}
    for split in SPLITS:
        src = pd.read_csv(SRC / f"{split}.csv", encoding=CSV_ENCODING)
        lab = pd.read_csv(LABELS / f"sentence_labels_{split}.csv", encoding="utf-8").rename(columns={"id": ID_COL})
        dist = src[src["y_bin"] == 1]
        base = dist[dist[ID_COL].isin(set(lab[ID_COL]))].reset_index(drop=True)
        manifest["dropped_unlocated"][split] = sorted(int(i) for i in set(dist[ID_COL]) - set(lab[ID_COL]))
        manifest["rows"][split] = int(len(base))
        manifest["sources"][str(SRC / f"{split}.csv")] = _sha(SRC / f"{split}.csv")
        manifest["sources"][str(LABELS / f"sentence_labels_{split}.csv")] = _sha(LABELS / f"sentence_labels_{split}.csv")
        bases[split], labels[split] = base, lab

        _write(base, "full", split)
        for kind in ("crop", "marked"):
            texts, _ = build_texts(base, lab, "label", kind)
            _write(base.assign(**{TEXT_COL: texts}), f"{kind}_gold", split)

    for seed in SEEDS:
        for kind in ("crop", "marked"):
            variant = f"{kind}_sel_{seed}"
            (OUT / variant).mkdir(parents=True, exist_ok=True)
            shutil.copy(OUT / f"{kind}_gold" / "train.csv", OUT / variant / "train.csv")
            for split in ("val", "test"):
                pred_path = SELECTOR / f"predictions_{split}_{seed}.csv"
                pred = pd.read_csv(pred_path, encoding="utf-8")
                sents = labels[split].merge(pred[[ID_COL, "sent_idx", "selected", "score"]],
                                            on=[ID_COL, "sent_idx"], how="left", validate="1:1")
                assert sents["selected"].notna().all(), f"{pred_path} misses sentences"
                texts, fb = build_texts(bases[split], sents, None, kind)
                _write(bases[split].assign(**{TEXT_COL: texts}), variant, split)
                manifest["selector_top1_fallbacks"][f"{split}_{seed}"] = int(fb)
                manifest["sources"][str(pred_path)] = _sha(pred_path)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "selector_top1_fallbacks")}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
