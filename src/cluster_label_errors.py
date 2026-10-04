"""Unsupervised label-error detection, and a retrospective comparison of triage routes.

This script does three separable jobs. Run them together or pick one with --only.

STAGE 1  --only cluster
    Re-runs the K-Means experiment recorded in docs/FYP_PLAN.md section 3.4
    (ARI 0.009, "clusters track topic, not reasoning pattern") under three
    vectoriser settings, on spans and on full documents:

        stopwords   : stop_words="english"  -- the suspected original setting
        nostopwords : stop_words=None       -- keeps should/always/never/they
        markers     : binary presence over a CBT marker lexicon only

    16 of the 18 top learned features listed in FYP_PLAN section 3.5 are in
    sklearn's English stop list, so the "stopwords" run deletes most of the
    distortion signal before clustering while leaving every topic word intact.
    If that run reproduces ARI ~0.009 and the others do not, the original
    conclusion was a preprocessing artefact.

STAGE 2  --only neighbours
    The method the proposal actually specifies: unsupervised identification of
    probable LABEL ERRORS, which is a different task from recovering distortion
    type. No cluster has to align with any class. For each row we embed the
    text, take its k nearest neighbours, and score it by the fraction of those
    neighbours carrying a different label. A row sitting in a neighbourhood that
    disagrees with it is a label-error candidate. Output is a ranked queue.

STAGE 3  --only compare
    The retrospective three-way comparison, which needs NO new annotation.
    The re-annotation session already produced human verdicts; a row whose
    human label differs from its original label is a confirmed label error.
    That gives ground truth against which each route's ranking can be scored:

        model-driven : the confident-learning triage already run
        unsupervised : stage 2's neighbour disagreement
        random       : the floor

    Reported as average precision and as precision/recall at k, so the routes
    are compared on how many confirmed errors each surfaces per row reviewed.

Usage (Windows):
    venv\\Scripts\\python.exe -m src.cluster_label_errors
    venv\\Scripts\\python.exe -m src.cluster_label_errors --only cluster
    venv\\Scripts\\python.exe -m src.cluster_label_errors --k 15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

SEED = 42
N_CLUSTERS = 10
N_NEIGHBOURS = 10
SVD_DIMS = 100

TRAIN_SPLIT = Path("data/splits/train.csv")
TRIAGE = Path("reannotation_2026-09-27/inputs/triage_full.csv")
MERGED = Path("reannotation_2026-09-27/outputs/relabel/merged_labels.csv")
OUT = Path("results/results_cluster_label_errors")

TEXT_COL = "Patient Question"
SPAN_COL = "Distorted part"
LABEL_COL = "Dominant Distortion"
ID_COL = "Id_Number"

# Markers taken from FYP_PLAN section 3.5 ("the task is lexically cued"), which
# lists the top learned features per class on spans.
MARKER_LEXICON = [
    "should", "shouldnt", "supposed", "must", "have to", "ought",
    "always", "never", "everything", "nothing", "everyone", "nobody",
    "all", "every", "completely", "totally", "constantly",
    "they", "think", "thinks", "thinking", "probably", "going to", "will",
    "my fault", "guilty", "blame", "because of me",
    "stupid", "failure", "worthless", "loser", "idiot", "useless",
    "feel like", "i feel", "makes me feel",
    "cant", "cannot", "impossible", "no point", "hopeless",
    "terrible", "awful", "horrible", "disaster", "ruined",
]


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def load_distorted_train() -> pd.DataFrame:
    """Distorted rows of the frozen train split. Never re-splits the data."""
    if not TRAIN_SPLIT.exists():
        raise SystemExit(f"missing {TRAIN_SPLIT} -- run src.make_splits first")
    df = pd.read_csv(TRAIN_SPLIT, encoding="utf-8-sig")
    mask = df[LABEL_COL].astype(str).str.strip().str.lower() != "no distortion"
    out = df[mask].copy()
    out[SPAN_COL] = out[SPAN_COL].fillna("").astype(str)
    return out.reset_index(drop=True)


def purity(labels_true, labels_pred) -> float:
    ct = pd.crosstab(labels_pred, labels_true)
    return ct.max(axis=1).sum() / ct.values.sum()


def majority_floor(labels_true) -> float:
    counts = pd.Series(labels_true).value_counts()
    return counts.iloc[0] / counts.sum()


def build_matrix(texts, setting: str):
    """Return a document-term matrix under one of the three vectoriser settings."""
    if setting == "stopwords":
        vec = TfidfVectorizer(stop_words="english", min_df=2)
    elif setting == "nostopwords":
        vec = TfidfVectorizer(stop_words=None, min_df=2)
    elif setting == "markers":
        # Binary presence over the marker lexicon only. Topic words cannot
        # enter the space at all, so any structure found is reasoning-related.
        vec = TfidfVectorizer(vocabulary=MARKER_LEXICON, ngram_range=(1, 2),
                              binary=True, use_idf=False, norm="l2")
    else:
        raise ValueError(setting)
    return vec.fit_transform(texts), vec


# --------------------------------------------------------------------------- #
# stage 1: does the clustering result survive a change of preprocessing?
# --------------------------------------------------------------------------- #
def stage_cluster(df: pd.DataFrame) -> pd.DataFrame:
    y = df[LABEL_COL].tolist()
    floor = majority_floor(y)

    # Guard against the bug the identical spans/documents numbers hint at:
    # if these two are the same text, the comparison is vacuous.
    docs = df[TEXT_COL].astype(str).tolist()
    spans = df[SPAN_COL].astype(str).tolist()
    identical = sum(a.strip() == b.strip() for a, b in zip(docs, spans))
    print(f"[check] rows where span text == document text: "
          f"{identical}/{len(df)} ({identical / len(df):.1%})")
    empty_spans = sum(not s.strip() for s in spans)
    print(f"[check] rows with an empty span: {empty_spans}")

    rows = []
    for source, texts in (("spans", spans), ("full documents", docs)):
        for setting in ("stopwords", "nostopwords", "markers"):
            usable = [t if t.strip() else " " for t in texts]
            X, vec = build_matrix(usable, setting)
            n_feat = X.shape[1]
            if n_feat < N_CLUSTERS:
                print(f"[skip] {source}/{setting}: only {n_feat} features")
                continue
            km = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
            pred = km.fit_predict(X)
            rows.append({
                "clustered on": source,
                "vectoriser": setting,
                "features": n_feat,
                "ARI": round(adjusted_rand_score(y, pred), 4),
                "NMI": round(normalized_mutual_info_score(y, pred), 4),
                "purity": round(purity(y, pred), 4),
                "majority floor": round(floor, 4),
            })
            print(f"  {source:15} {setting:12} ARI={rows[-1]['ARI']:.4f} "
                  f"purity={rows[-1]['purity']:.4f}")

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "cluster_recovery.csv", index=False, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# stage 2: unsupervised label-error candidates by neighbourhood disagreement
# --------------------------------------------------------------------------- #
def stage_neighbours(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Score each row by how much its nearest neighbours disagree with its label.

    This asks only for LOCAL consistency, so it does not require any cluster to
    line up with a distortion class -- which is why it can work on data where
    stage 1's global partition fails.
    """
    texts = df[TEXT_COL].astype(str).tolist()
    X, _ = build_matrix(texts, "nostopwords")
    dims = min(SVD_DIMS, X.shape[1] - 1)
    Z = normalize(TruncatedSVD(n_components=dims, random_state=SEED).fit_transform(X))

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Z)
    _, idx = nn.kneighbors(Z)
    idx = idx[:, 1:]  # drop self

    labels = df[LABEL_COL].to_numpy()
    disagree = np.array([(labels[nb] != labels[i]).mean() for i, nb in enumerate(idx)])

    # The label the neighbourhood would assign instead.
    suggested = []
    for i, nb in enumerate(idx):
        vals, counts = np.unique(labels[nb], return_counts=True)
        suggested.append(vals[counts.argmax()])

    out = pd.DataFrame({
        ID_COL: df[ID_COL],
        "original_label": labels,
        "neighbour_disagreement": disagree.round(4),
        "neighbour_majority_label": suggested,
    }).sort_values("neighbour_disagreement", ascending=False)

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "unsupervised_queue.csv", index=False, encoding="utf-8")
    print(f"  scored {len(out)} rows; "
          f"{(disagree >= 0.9).sum()} rows disagree with >=90% of neighbours")
    return out


# --------------------------------------------------------------------------- #
# stage 3: score every route against the re-annotation that already happened
# --------------------------------------------------------------------------- #
def human_verdicts() -> pd.DataFrame | None:
    """Majority human label per row from the completed re-annotation session."""
    if not MERGED.exists():
        print(f"[skip] {MERGED} not found -- stage 3 needs the completed "
              f"workbooks merged by relabel_agreement.py")
        return None
    m = pd.read_csv(MERGED, encoding="utf-8-sig")
    m = m[m["Primary"].notna()]
    if m.empty:
        print("[skip] merged_labels.csv has no filled labels yet")
        return None

    rows = []
    for rid, grp in m.groupby("Row ID"):
        vote = grp["primary_collapsed"].value_counts()
        rows.append({
            ID_COL: rid,
            "human_label": vote.index[0],
            "n_annotators": len(grp),
            "unanimous": bool(len(vote) == 1),
            "original_dominant": grp["original_dominant"].iloc[0],
        })
    v = pd.DataFrame(rows)
    v["changed"] = (v["human_label"].str.strip().str.lower()
                    != v["original_dominant"].astype(str).str.strip().str.lower())
    return v


def average_precision(scores: np.ndarray, truth: np.ndarray) -> float:
    order = np.argsort(-scores)
    hits = truth[order]
    if hits.sum() == 0:
        return float("nan")
    cum = np.cumsum(hits)
    prec = cum / np.arange(1, len(hits) + 1)
    return float((prec * hits).sum() / hits.sum())


def stage_compare(unsup: pd.DataFrame) -> pd.DataFrame | None:
    verdicts = human_verdicts()
    if verdicts is None:
        return None
    if not TRIAGE.exists():
        print(f"[skip] missing {TRIAGE}")
        return None

    tri = pd.read_csv(TRIAGE, encoding="utf-8-sig")
    d = verdicts.merge(tri, on=ID_COL, how="inner").merge(
        unsup[[ID_COL, "neighbour_disagreement"]], on=ID_COL, how="left")
    d["neighbour_disagreement"] = d["neighbour_disagreement"].fillna(0.0)

    truth = d["changed"].to_numpy()
    base = truth.mean()
    print(f"  rows with a human verdict and triage score: {len(d)}")
    print(f"  confirmed label changes: {truth.sum()} ({base:.1%} base rate)")

    rng = np.random.default_rng(SEED)
    routes = {
        # Low self-confidence means the model disputes the recorded label.
        "model-driven (self-confidence)": -d["self_confidence"].fillna(0).to_numpy(),
        "model-driven (cleanlab)": -d["cleanlab_quality"].fillna(1).to_numpy(),
        "model uncertainty (entropy)": d["entropy"].fillna(0).to_numpy(),
        "unsupervised (neighbours)": d["neighbour_disagreement"].to_numpy(),
        "random": rng.random(len(d)),
    }

    rows = []
    for name, score in routes.items():
        order = np.argsort(-score)
        rec = {"route": name, "AP": round(average_precision(score, truth), 4)}
        for k in (100, 250, 500):
            if k > len(d):
                continue
            top = truth[order[:k]]
            rec[f"precision@{k}"] = round(top.mean(), 4)
            rec[f"recall@{k}"] = round(top.sum() / truth.sum(), 4)
            rec[f"lift@{k}"] = round(top.mean() / base, 2) if base else float("nan")
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values("AP", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "route_comparison.csv", index=False, encoding="utf-8")
    (OUT / "route_comparison_meta.json").write_text(json.dumps({
        "rows_scored": int(len(d)),
        "confirmed_changes": int(truth.sum()),
        "base_rate": round(float(base), 4),
        "seed": SEED,
        "n_neighbours": N_NEIGHBOURS,
    }, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cluster", "neighbours", "compare"])
    ap.add_argument("--k", type=int, default=N_NEIGHBOURS)
    args = ap.parse_args()

    df = load_distorted_train()
    print(f"distorted train rows: {len(df)}\n")

    unsup = None
    if args.only in (None, "cluster"):
        print("STAGE 1  cluster recovery of distortion type")
        print(stage_cluster(df).to_string(index=False), "\n")

    if args.only in (None, "neighbours", "compare"):
        print("STAGE 2  unsupervised label-error queue")
        unsup = stage_neighbours(df, args.k)
        print()

    if args.only in (None, "compare"):
        print("STAGE 3  route comparison against human verdicts")
        res = stage_compare(unsup)
        if res is not None:
            print(res.to_string(index=False))

    print(f"\nwrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
