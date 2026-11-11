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

    Verdicts follow D08 exactly (reannotation_2026-09-27/scripts/label_changes.py):
    a vote counts only if Primary is filled, it was not filled in from the
    original label, and Unsure is not ticked; a row needs one 11-way label with
    more than half the valid votes, and "changed" means that label differs from
    the original dominant label. Two tables (D11):
        route_comparison.csv       all lists on the rows every list can score
                                   (neighbour queue = distorted train rows only)
        route_comparison_test.csv  model triage vs random on the test rows, the
                                   only re-annotated rows that include unflagged ones
    Confirmed errors per hour assume 45 s per row (no times were recorded).

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
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

SEED = 42
N_CLUSTERS = 10
N_NEIGHBOURS = 10
SVD_DIMS = 100
MIN_CLUSTER_SIZE = 5      # HDBSCAN: smallest group it will call a cluster
DUP_THRESHOLD = 0.80      # cosine above which two texts are "near-duplicates"

RAW = Path("Annotated_data.csv")
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

            # HDBSCAN needs dense input and benefits from the same reduction
            # the neighbour stage uses; K-Means runs on the sparse matrix.
            dims = min(SVD_DIMS, n_feat - 1)
            Z = normalize(TruncatedSVD(n_components=dims,
                                       random_state=SEED).fit_transform(X))

            for algo in ("kmeans", "hdbscan"):
                if algo == "kmeans":
                    pred = KMeans(n_clusters=N_CLUSTERS, random_state=SEED,
                                  n_init=10).fit_predict(X)
                    keep = np.ones(len(pred), dtype=bool)
                else:
                    # Density-based: picks its own cluster count and leaves
                    # rows in sparse regions unassigned (label -1 = noise).
                    pred = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE,
                                   metric="euclidean").fit_predict(Z)
                    keep = pred != -1
                    if keep.sum() < N_CLUSTERS:
                        print(f"  {source:15} {setting:12} hdbscan  "
                              f"-- everything called noise, skipped")
                        continue

                # Scored on clustered rows only; "covered" says how many that is,
                # so a high score on 3% of the data cannot look like a win.
                yk = np.asarray(y)[keep]
                pk = pred[keep]
                rows.append({
                    "clustered on": source,
                    "vectoriser": setting,
                    "algorithm": algo,
                    "features": n_feat,
                    "clusters": int(len(set(pk))),
                    "covered": round(keep.mean(), 4),
                    "ARI": round(adjusted_rand_score(yk, pk), 4),
                    "NMI": round(normalized_mutual_info_score(yk, pk), 4),
                    "purity": round(purity(yk, pk), 4),
                    "majority floor": round(floor, 4),
                })
                r = rows[-1]
                print(f"  {source:15} {setting:12} {algo:8} "
                      f"k={r['clusters']:3} covered={r['covered']:.0%} "
                      f"ARI={r['ARI']:.4f} purity={r['purity']:.4f}")

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
# stage 2b: near-duplicate pairs carrying conflicting labels
# --------------------------------------------------------------------------- #
def load_with_splits() -> pd.DataFrame:
    """Whole corpus with a split column, so duplicates can be checked across them."""
    df = pd.read_csv(RAW, encoding="utf-8-sig")
    df["split"] = "unassigned"
    for name in ("train", "val", "test"):
        p = Path(f"data/splits/{name}.csv")
        if p.exists():
            ids = set(pd.read_csv(p, encoding="utf-8-sig")[ID_COL])
            df.loc[df[ID_COL].isin(ids), "split"] = name
    return df


def stage_duplicates(threshold: float) -> pd.DataFrame:
    """Find text pairs that are near-copies, then report where their labels differ.

    This needs no cluster structure at all. Two texts that are near-identical
    should carry the same label; where they do not, one of the two is very
    likely mislabelled. The measured agreement at cosine >= 0.9 is 52%, against
    a 10.9% chance rate, so a conflict in this band is a strong signal.

    The same scan also catches near-duplicates that straddle a split boundary,
    which would be leakage rather than a label error.
    """
    df = load_with_splits()
    texts = df[TEXT_COL].astype(str).tolist()

    # Raw TF-IDF, not the SVD projection: near-duplicate detection wants exact
    # wording overlap, and the reduction blurs precisely that.
    X = normalize(TfidfVectorizer(stop_words=None, min_df=1).fit_transform(texts))
    S = (X @ X.T).toarray()
    np.fill_diagonal(S, 0.0)

    ia, ib = np.where(np.triu(S, k=1) >= threshold)
    if len(ia) == 0:
        print(f"  no pairs at cosine >= {threshold}")
        return pd.DataFrame()

    lab = df[LABEL_COL].astype(str).str.strip().to_numpy()
    out = pd.DataFrame({
        "id_a": df[ID_COL].to_numpy()[ia],
        "id_b": df[ID_COL].to_numpy()[ib],
        "similarity": S[ia, ib].round(4),
        "label_a": lab[ia],
        "label_b": lab[ib],
        "split_a": df["split"].to_numpy()[ia],
        "split_b": df["split"].to_numpy()[ib],
        "text_a": [texts[i][:160].replace("\n", " ") for i in ia],
        "text_b": [texts[i][:160].replace("\n", " ") for i in ib],
    })
    out["label_conflict"] = out["label_a"] != out["label_b"]
    out["cross_split"] = out["split_a"] != out["split_b"]
    out = out.sort_values(["label_conflict", "similarity"], ascending=[False, False])

    n, c, x = len(out), int(out["label_conflict"].sum()), int(out["cross_split"].sum())
    print(f"  pairs at cosine >= {threshold}: {n}")
    print(f"  of those, labels conflict      : {c} ({c / n:.0%}) <- label-error candidates")
    print(f"  of those, cross split boundary : {x} <- leakage candidates")

    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "near_duplicates.csv", index=False, encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# stage 3: score every route against the re-annotation that already happened
# --------------------------------------------------------------------------- #
LABEL_CHANGES = Path("reannotation_2026-09-27/scripts/label_changes.py")
SECONDS_PER_ROW = 45          # no per-row times were recorded in the workbooks
TIME_BASIS = "45 s/row estimate (no times recorded)"
KS = (50, 100, 200)
BUCKET_RANK = {"A": 0, "B": 1}  # anything else (C, "agree") ranks after the flagged buckets


def _d08_classify():
    """D08's grouping rule, loaded from its script so both tasks share one definition."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("label_changes", LABEL_CHANGES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.classify


def human_verdicts() -> pd.DataFrame | None:
    """Majority human label per row, with D08's vote rules (see module docstring)."""
    if not MERGED.exists():
        print(f"[skip] {MERGED} not found -- stage 3 needs the completed "
              f"workbooks merged by relabel_agreement.py")
        return None
    m = pd.read_csv(MERGED, encoding="utf-8-sig")
    filled = m["filled_from_original"].astype(str).str.lower() == "true"
    valid = m[m["Primary"].notna() & ~filled & m["Unsure"].isna()]
    if valid.empty:
        print("[skip] merged_labels.csv has no usable labels yet")
        return None

    classify = _d08_classify()
    info = m.drop_duplicates("Row ID").set_index("Row ID")
    rows = []
    for rid, votes in valid.groupby("Row ID")["Primary"]:
        new, group = classify(str(info.at[rid, "original_dominant"]).strip(),
                              info.at[rid, "original_secondary"], votes.tolist())
        if new is None:          # split vote: no verdict
            continue
        rows.append({ID_COL: int(rid), "human_label": new, "changed": group is not None,
                     "group": group, "n_annotators": len(votes), "split": info.at[rid, "split"]})
    return pd.DataFrame(rows)


def rank_order(scores, tiebreak=None) -> np.ndarray:
    """Indices from highest to lowest score; ties broken by `tiebreak` (higher first)."""
    scores = np.asarray(scores, float)
    tb = np.zeros(len(scores)) if tiebreak is None else np.asarray(tiebreak, float)
    return np.lexsort((-tb, -scores))


def average_precision(scores: np.ndarray, truth: np.ndarray, tiebreak: np.ndarray | None = None) -> float:
    """AP of a ranking (highest score first)."""
    hits = np.asarray(truth, float)[rank_order(scores, tiebreak)]
    if hits.sum() == 0:
        return float("nan")
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((prec * hits).sum() / hits.sum())


def model_triage_score(bucket, self_confidence) -> np.ndarray:
    """Model triage list as one sortable score: bucket A, then B, then the rest; inside a
    bucket, lowest self-confidence (the model disputes the label most) first."""
    rank = np.array([BUCKET_RANK.get(str(b), 2) for b in bucket], float)
    return -(rank * 2 + np.nan_to_num(np.asarray(self_confidence, float), nan=1.0))


def score_lists(d: pd.DataFrame, lists: dict) -> pd.DataFrame:
    truth = d["changed"].to_numpy(bool)
    base = truth.mean()
    tiebreak = np.random.default_rng(SEED + 1).random(len(d))  # fixed, fair order for tied scores
    rows = []
    for name, score in lists.items():
        hits = truth[rank_order(score, tiebreak)]
        rec = {"list": name, "rows_scored": len(d), "confirmed_errors": int(truth.sum()),
               "base_rate": round(float(base), 4),
               "AP": round(average_precision(score, truth, tiebreak), 4)}
        for k in KS:
            rec[f"precision@{k}"] = round(float(hits[:k].mean()), 4) if k <= len(d) else float("nan")
        p100 = rec["precision@100"]
        rec["lift@100"] = round(p100 / base, 2) if base else float("nan")
        rec["errors_per_hour@100"] = round(p100 * 3600 / SECONDS_PER_ROW, 1)
        rec["time_basis"] = TIME_BASIS
        rows.append(rec)
    return pd.DataFrame(rows)


def stage_compare(unsup: pd.DataFrame) -> pd.DataFrame | None:
    verdicts = human_verdicts()
    if verdicts is None:
        return None
    if not TRIAGE.exists():
        print(f"[skip] missing {TRIAGE}")
        return None
    tri = pd.read_csv(TRIAGE, encoding="utf-8-sig").drop(columns=["Patient Question"], errors="ignore")
    allv = verdicts.merge(tri, on=ID_COL, how="inner")

    def model_lists(d):
        return {
            "model triage (A+B buckets)": model_triage_score(d["bucket"], d["self_confidence"]),
            "random (seed 42)": np.random.default_rng(SEED).random(len(d)),
            "extra: model self-confidence only": -d["self_confidence"].fillna(1).to_numpy(),
            "extra: model cleanlab quality": -d["cleanlab_quality"].fillna(1).to_numpy(),
            "extra: model entropy": d["entropy"].fillna(0).to_numpy(),
        }

    # Table 1: rows every list can score (the neighbour queue holds distorted train rows only).
    common = allv.merge(unsup[[ID_COL, "neighbour_disagreement"]], on=ID_COL, how="inner")
    lists1 = {"neighbours (clustering)": common["neighbour_disagreement"].to_numpy(), **model_lists(common)}
    t1 = score_lists(common, lists1)
    # Table 2: test rows, the only re-annotated rows that include unflagged (bucket C) ones.
    test = allv[allv["split"] == "test"].reset_index(drop=True)
    t2 = score_lists(test, model_lists(test))

    OUT.mkdir(parents=True, exist_ok=True)
    t1.to_csv(OUT / "route_comparison.csv", index=False, encoding="utf-8")
    t2.to_csv(OUT / "route_comparison_test.csv", index=False, encoding="utf-8")
    flag = test["bucket"].isin(BUCKET_RANK)
    meta = {
        "verdict_rows": int(len(allv)), "verdict_rows_changed": int(allv["changed"].sum()),
        "table1_rows": int(len(common)),
        "table1_buckets": {str(k): int(v) for k, v in common["bucket"].value_counts().items()},
        "table2_rows": int(len(test)),
        "table2_flagged_changed": f"{int(test.loc[flag, 'changed'].sum())} of {int(flag.sum())}",
        "table2_unflagged_changed": f"{int(test.loc[~flag, 'changed'].sum())} of {int((~flag).sum())}",
        "seconds_per_row": SECONDS_PER_ROW, "time_basis": TIME_BASIS,
        "seed": SEED, "n_neighbours": N_NEIGHBOURS,
    }
    (OUT / "route_comparison_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_compare_report(t1, t2, meta)
    return t1


def write_compare_report(t1: pd.DataFrame, t2: pd.DataFrame, meta: dict) -> None:
    show = ["list", "AP", "precision@50", "precision@100", "precision@200", "lift@100",
            "errors_per_hour@100", "rows_scored"]

    def get(t, name, col):
        return float(t.loc[t["list"] == name, col].iloc[0])

    def table(t):  # precision@k is undefined when the list is shorter than k
        return t[show].astype(object).where(t[show].notna(), "n/a").to_markdown(index=False)

    nb, mt, rd = "neighbours (clustering)", "model triage (A+B buckets)", "random (seed 42)"
    winner = max((nb, mt, rd), key=lambda n: get(t1, n, "AP"))
    ari = ""
    rec_path = OUT / "cluster_recovery.csv"
    if rec_path.exists():
        a = pd.read_csv(rec_path)["ARI"]
        ari = (f" Stage 1 shows why clustering struggles: clusters follow topic, not distortion type "
               f"(ARI {a.min():.3f} to {a.max():.3f} across settings, `cluster_recovery.csv`).")
    md = [
        "# D11 - Which list finds wrong labels best?", "",
        "Three ranked lists are scored against the re-annotation verdicts: **neighbours** (stage 2: a row whose "
        "10 nearest texts mostly carry another label ranks high), **model triage** (the A and B buckets of the "
        "re-annotation session: A before B, and inside a bucket the row whose label the model trusts least first), "
        "and **random** order (seed 42), the floor. Rows marked *extra* are the model's raw scores, for reference.", "",
        "**Verdicts** use the same rule as D08 (`label_changes.py`): a vote counts only if Primary is filled, it was "
        "not filled in from the original label (81 CS-majors blanks), and Unsure is not ticked (never adjudicated); "
        "a row needs one 11-way label (calibrated pessimism = No Distortion) with more than half its valid votes. "
        "A row is a **confirmed error** when that label differs from the original dominant label. "
        f"{meta['verdict_rows']} rows have a verdict; {meta['verdict_rows_changed']} of them changed.", "",
        "**Scores.** Average precision (AP) is one number for the whole list, high when real errors sit near the top. "
        "Precision@k = share of the first k rows that were real errors. Lift@100 = precision@100 / base rate "
        "(what random order gets on average). **Confirmed errors per hour** = precision@100 x 80 rows per hour, "
        f"using the **{SECONDS_PER_ROW} s per row estimate** (no times were recorded). Tied scores are ordered by "
        "a fixed random draw.", "",
        "## Table 1 - all three lists, on the rows each list can score", "",
        f"{meta['table1_rows']} rows (buckets: {meta['table1_buckets']}). The neighbour queue covers distorted "
        "**train** rows only, and the only train rows that were re-annotated are ones the model had flagged, so "
        "**every row here is already in the model's A/B buckets** and the base rate is high "
        f"({t1['base_rate'].iloc[0]:.1%}). Precision near the base rate means a list adds nothing; read lift "
        "and AP against random.", "",
        table(t1), "",
        "## Table 2 - model triage vs random on the test rows", "",
        f"{meta['table2_rows']} test rows with a verdict (base rate {t2['base_rate'].iloc[0]:.1%}): flagged "
        f"changed {meta['table2_flagged_changed']}, unflagged changed {meta['table2_unflagged_changed']}. This is "
        "the only part of the re-annotation that included rows the model did **not** flag, so it is the only fair "
        "test of the A/B selection itself. Neighbours cannot be scored here (its queue has no test rows).", "",
        table(t2), "",
        "precision@200 is n/a on this table: it has fewer than 200 rows.", "",
        "## Answer", "",
        f"On the shared rows the best of the three lists by AP is **{winner}**: neighbours AP "
        f"{get(t1, nb, 'AP'):.3f}, model triage {get(t1, mt, 'AP'):.3f}, random {get(t1, rd, 'AP'):.3f}. "
        f"On the test rows model triage scores AP {get(t2, mt, 'AP'):.3f} vs random {get(t2, rd, 'AP'):.3f} "
        f"(precision@50 {get(t2, mt, 'precision@50'):.2f} vs {get(t2, rd, 'precision@50'):.2f}).{ari}", "",
        "**Caveat.** Ground truth exists only for re-annotated rows, and the model chose most of them, so Table 1 "
        "compares the lists *inside the model's own selection*. Table 2 is small and has one annotator per row. "
        "Neither table is an unbiased estimate over all 2,530 rows.", "",
        "Files: `route_comparison.csv` (Table 1), `route_comparison_test.csv` (Table 2), "
        "`route_comparison_meta.json`. Script: `python -m src.cluster_label_errors --only compare`.", "",
    ]
    (OUT / "route_comparison.md").write_text("\n".join(md), encoding="utf-8")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cluster", "neighbours", "duplicates",
                                       "compare"])
    ap.add_argument("--k", type=int, default=N_NEIGHBOURS)
    ap.add_argument("--dup-threshold", type=float, default=DUP_THRESHOLD)
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

    if args.only in (None, "duplicates"):
        print("STAGE 2b near-duplicate pairs with conflicting labels")
        stage_duplicates(args.dup_threshold)
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
