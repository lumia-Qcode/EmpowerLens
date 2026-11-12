"""
D18 - sanity checks for the challenge-set pilot pairs (pairs_lumia.csv / pairs_izza.csv).

RUN (from repo root):
    venv\\Scripts\\python.exe -m src.check_challenge_pairs data/challenge_set/pairs_lumia.csv [more.csv]

Checks, per file: required columns, unique pair_id, 5 pairs per distortion type, no empty
texts, and the two texts of each pair within 20% of each other's length (characters, relative
to the longer one). Also runs a PRE-CHECK of D19's leak check: TF-IDF + logistic regression,
5-fold stratified CV, separating justified from distorted texts (F1 of "distorted"). D19 runs
the official one on all 100 items; above 0.75 means the halves differ in wording, not reasoning.
Exit code 1 if any structural check fails. Writes nothing.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline

COLS = ["pair_id", "distortion_type", "situation", "justified_text", "distorted_text"]
MAX_LEN_GAP = 0.20
LEAK_LIMIT = 0.75


def length_gap(a: str, b: str) -> float:
    la, lb = len(a), len(b)
    return abs(la - lb) / max(la, lb)


def leak_f1(df: pd.DataFrame, seed: int = 42) -> float:
    texts = list(df["justified_text"]) + list(df["distorted_text"])
    y = np.array([0] * len(df) + [1] * len(df))
    model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                          LogisticRegression(max_iter=1000))
    pred = cross_val_predict(model, texts, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed))
    return float(f1_score(y, pred))


def check(path: str) -> bool:
    df = pd.read_csv(path, encoding="utf-8")
    ok = True
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        print(f"[FAIL] {path}: missing columns {missing}")
        return False
    if df["pair_id"].duplicated().any():
        print(f"[FAIL] duplicate pair_id: {df.loc[df['pair_id'].duplicated(), 'pair_id'].tolist()}")
        ok = False
    counts = df["distortion_type"].value_counts()
    if (counts != 5).any():
        print(f"[FAIL] expected 5 pairs per type, got {counts.to_dict()}")
        ok = False
    if df[COLS].isna().any().any() or (df[COLS[2:]].apply(lambda s: s.str.strip().str.len()) == 0).any().any():
        print("[FAIL] empty cells")
        ok = False
    gaps = df.apply(lambda r: length_gap(r["justified_text"], r["distorted_text"]), axis=1)
    bad = df.loc[gaps > MAX_LEN_GAP, "pair_id"].tolist()
    if bad:
        print(f"[FAIL] length gap > {MAX_LEN_GAP:.0%}: {bad}")
        ok = False
    f1 = leak_f1(df)
    print(f"{path}: {len(df)} pairs, types {counts.to_dict()}")
    print(f"  length gap: max {gaps.max():.1%}, mean {gaps.mean():.1%}")
    print(f"  leak pre-check (TF-IDF+LR, 5-fold) F1 = {f1:.3f} "
          f"({'OK' if f1 <= LEAK_LIMIT else 'TOO HIGH - wording gives the label away'})")
    print(f"  structure: {'OK' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    paths = sys.argv[1:] or ["data/challenge_set/pairs_lumia.csv"]
    sys.exit(0 if all([check(p) for p in paths]) else 1)
