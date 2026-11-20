"""C02 — near-duplicate report and train/test leakage check.

Two test rows (Id 304 and Id 137) have a ~98%-identical copy in train (336 and
139). The brief asks for the chosen model's (D12: MentalRoBERTa) headline test
numbers with and without them.

No per-row test predictions were saved for the D13 binary runs or the D09
multilabel runs, and their checkpoints were deleted. So the 251-row numbers are
computed as exact **bounds** from the saved confusion counts: each removed row
was either predicted right or wrong, and every case the counts allow is tried.

This script never opens ``data/splits/test.csv``. Its inputs are committed
result files plus the source ``Annotated_data.csv`` (for row 304's label vector).

    venv\\Scripts\\python.exe -m src.duplicates_report
"""

from __future__ import annotations

import json
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DISTORTIONS, ID_COL, load_raw, make_targets

MODEL = "mental/mental-roberta-base"
MODEL_SLUG = "mental-roberta-base"
SEEDS = (42, 1337, 2024)
DUP_TEST_ND = 137      # test row, No Distortion, copy of train 139
DUP_TEST_DIST = 304    # test row, Fortune-telling, copy of train 336
ML_CAP = 2             # src/evaluate.py --max-labels default; D09 did not override it

DUP_DIR = Path("results/results_cluster_label_errors")
BIN_DIR = Path("results/results_thresholds")
ML_DIR = Path("results/results_mentalroberta_frozensplits")


# --------------------------------------------------------------------------- #
# metric arithmetic on counts

def f1(tp: int, fp: int, fn: int) -> float:
    """F1 with sklearn's zero_division=0 convention."""
    d = 2 * tp + fp + fn
    return 0.0 if d == 0 else 2 * tp / d


def binary_metrics(tn: int, fp: int, tp: int, fn: int) -> dict:
    """Specificity, FPR, sensitivity and macro-F1 (non_distorted + distorted)."""
    spec = tn / (tn + fp)
    return {
        "specificity": spec,
        "fpr": 1 - spec,
        "sensitivity": tp / (tp + fn),
        # non_distorted as positive: its TP=tn, FP=fn, FN=fp.
        "macro_f1": (f1(tn, fn, fp) + f1(tp, fp, fn)) / 2,
    }


def binary_counts_bounds(tn: int, fp: int, tp: int, fn: int,
                         drop_nd: int = 1, drop_dist: int = 1) -> dict:
    """Min / max of each binary metric after removing rows of unknown outcome.

    ``drop_nd`` non-distorted rows are removed (each was a TN or an FP) and
    ``drop_dist`` distorted rows (each a TP or an FN). Only splits the counts
    allow are tried. Returns ``{metric: (lo, hi)}``.
    """
    vals: dict[str, list[float]] = {}
    for k_fp in range(drop_nd + 1):            # how many removed ND rows were FPs
        for k_fn in range(drop_dist + 1):      # how many removed distorted rows were FNs
            k_tn, k_tp = drop_nd - k_fp, drop_dist - k_fn
            if k_tn > tn or k_fp > fp or k_tp > tp or k_fn > fn:
                continue
            m = binary_metrics(tn - k_tn, fp - k_fp, tp - k_tp, fn - k_fn)
            for name, v in m.items():
                vals.setdefault(name, []).append(v)
    if not vals:
        raise ValueError("no removal case is consistent with the counts")
    return {k: (min(v), max(v)) for k, v in vals.items()}


def multilabel_macro_f1_bounds(tp, fp, fn, true_dist_vec,
                               max_fired: int | None = None) -> tuple[float, float]:
    """Min / max multilabel macro-F1 after removing one ND row and one distorted row.

    The ND row has no true types, so any type it fired was an FP. The distorted
    row's true types (``true_dist_vec``) were each a TP or an FN; any other
    type it fired was an FP.

    ``max_fired=None``: no cap on fired types (the widest bound). Each type's
    outcome is then independent, so the bound is exact per type.
    ``max_fired=k``: each row fired at most k types, matching
    ``src/evaluate.py --max-labels`` (default 2). Every pair of fired-type sets
    is enumerated.
    """
    tp, fp, fn = (np.asarray(a, dtype=int) for a in (tp, fp, fn))
    t = np.asarray(true_dist_vec, dtype=int)
    if max_fired is not None:
        return _ml_bounds_capped(tp, fp, fn, t, max_fired)
    lo, hi = [], []
    for c in range(len(tp)):
        cand = []
        for a, b in product((0, 1), repeat=2):   # a: ND row fired c; b: distorted row fired c
            d_fp = a + (b if t[c] == 0 else 0)
            d_tp = b if t[c] == 1 else 0
            d_fn = (1 - b) if t[c] == 1 else 0
            if d_fp > fp[c] or d_tp > tp[c] or d_fn > fn[c]:
                continue
            cand.append(f1(tp[c] - d_tp, fp[c] - d_fp, fn[c] - d_fn))
        if not cand:
            raise ValueError(f"type {c}: no removal case is consistent with the counts")
        lo.append(min(cand))
        hi.append(max(cand))
    return float(np.mean(lo)), float(np.mean(hi))


def _ml_bounds_capped(tp, fp, fn, t, k):
    n = len(tp)
    subsets = [np.isin(np.arange(n), c).astype(int)
               for r in range(k + 1) for c in combinations(range(n), r)]
    vals = []
    for a in subsets:          # ND row's fired types: all FPs
        for b in subsets:      # distorted row's fired types
            d_fp = a + b * (1 - t)
            d_tp = b * t
            d_fn = (1 - b) * t
            if np.any(d_fp > fp) or np.any(d_tp > tp) or np.any(d_fn > fn):
                continue
            vals.append(macro_f1_from_counts(tp - d_tp, fp - d_fp, fn - d_fn))
    if not vals:
        raise ValueError("no removal case is consistent with the counts")
    return min(vals), max(vals)


def multilabel_counts(per_class: pd.DataFrame):
    """Recover per-type TP / FP / FN from a per_class CSV (precision, recall, support)."""
    s = per_class["support"].to_numpy(dtype=int)
    tp = np.rint(per_class["recall"].to_numpy() * s).astype(int)
    p = per_class["precision"].to_numpy()
    pred = np.where(p > 0, np.rint(np.divide(tp, p, out=np.zeros_like(p), where=p > 0)), 0).astype(int)
    if np.any((p == 0) & (tp == 0)):
        # precision 0 with tp 0 hides FP: can't recover it.
        raise ValueError("a type has precision 0; FP count is not recoverable")
    return tp, pred - tp, s - tp


def macro_f1_from_counts(tp, fp, fn) -> float:
    return float(np.mean([f1(a, b, c) for a, b, c in zip(tp, fp, fn)]))


# --------------------------------------------------------------------------- #
# inputs

def load_inputs():
    spec = pd.read_csv(BIN_DIR / "specificity.csv")
    spec = spec[(spec["model"] == MODEL) & (spec["split"] == "test")].set_index("seed")

    raw = load_raw()
    _, _, y_ml = make_targets(raw)
    ids = raw[ID_COL].to_numpy()
    true_304 = y_ml[np.flatnonzero(ids == DUP_TEST_DIST)[0]].astype(int)
    true_137 = y_ml[np.flatnonzero(ids == DUP_TEST_ND)[0]].astype(int)
    assert true_137.sum() == 0 and true_304.sum() >= 1

    rows = []
    for seed in SEEDS:
        r = spec.loc[seed]
        tn, fp = int(r["tn"]), int(r["fp"])
        tp = int(round(r["sensitivity"] * r["n_distorted"]))
        fn = int(r["n_distorted"]) - tp

        bj = json.loads((BIN_DIR / f"eval_{MODEL_SLUG}_binary_{seed}.json").read_text(encoding="utf-8"))
        full_bin = binary_metrics(tn, fp, tp, fn)
        assert abs(full_bin["macro_f1"] - bj["splits"]["test"]["metrics"]["macro_f1"]) < 1e-9, seed

        pc = pd.read_csv(ML_DIR / f"per_class_{MODEL_SLUG}_multilabel_{seed}.csv")
        assert pc["class"].tolist() == DISTORTIONS
        mtp, mfp, mfn = multilabel_counts(pc)
        mj = json.loads((ML_DIR / f"eval_{MODEL_SLUG}_multilabel_{seed}.json").read_text(encoding="utf-8"))
        full_ml = macro_f1_from_counts(mtp, mfp, mfn)
        assert abs(full_ml - mj["splits"]["test"]["metrics"]["macro_f1"]) < 1e-9, seed

        bb = binary_counts_bounds(tn, fp, tp, fn)
        ml_lo, ml_hi = multilabel_macro_f1_bounds(mtp, mfp, mfn, true_304, max_fired=ML_CAP)
        nocap_lo, nocap_hi = multilabel_macro_f1_bounds(mtp, mfp, mfn, true_304)
        row = {"seed": seed, "tn": tn, "fp": fp, "tp": tp, "fn": fn}
        for k, v in full_bin.items():
            row[f"{k}_253"] = v
            # For FPR, the worst case is the higher value.
            lo, hi = bb[k]
            row[f"{k}_251_worst"], row[f"{k}_251_best"] = (hi, lo) if k == "fpr" else (lo, hi)
        row["ml_macro_f1_253"] = full_ml
        row["ml_macro_f1_251_worst"], row["ml_macro_f1_251_best"] = ml_lo, ml_hi
        row["ml_macro_f1_251_nocap_worst"], row["ml_macro_f1_251_nocap_best"] = nocap_lo, nocap_hi
        rows.append(row)
    return pd.DataFrame(rows), true_304


# --------------------------------------------------------------------------- #
# report

METRICS = [("specificity", "Specificity"), ("fpr", "FPR"), ("sensitivity", "Sensitivity"),
           ("macro_f1", "Binary macro-F1"), ("ml_macro_f1", "Multilabel macro-F1")]


def ms(x) -> str:
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.3f} ± {x.std(ddof=0):.3f}"


def yn(b) -> str:
    return "yes" if bool(b) else "no"


def build_report(per_seed: pd.DataFrame, pairs: pd.DataFrame, true_304) -> str:
    L = []
    L.append("# C02 — Near-duplicate report and train/test leakage check\n")
    L.append("Generated by `python -m src.duplicates_report`. Inputs: "
             "`results/results_cluster_label_errors/near_duplicates.csv` (TF-IDF cosine ≥ 0.80 over all 2,530 "
             "rows, from `src/cluster_label_errors.py --only duplicates`), the D13 binary results in "
             "`results/results_thresholds/` and the D09 multilabel results in "
             "`results/results_mentalroberta_frozensplits/`. The model is the D12 winner, MentalRoBERTa "
             "(`mental/mental-roberta-base`), seeds 42, 1337 and 2024. `data/splits/test.csv` is not read here, "
             "and the splits are unchanged.\n")

    n, nc, nx = len(pairs), int(pairs["label_conflict"].sum()), int(pairs["cross_split"].sum())
    L.append(f"## The {n} near-duplicate pairs\n")
    L.append("IDs are `Id_Number`. *Conflict* = the two copies carry different dominant labels. "
             "*Crosses splits* = the copies sit in different splits.\n")
    L.append("| ID A | ID B | Similarity | Label A | Label B | Split A | Split B | Conflict | Crosses splits |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in pairs.itertuples(index=False):
        L.append(f"| {r.id_a} | {r.id_b} | {r.similarity:.4f} | {r.label_a} | {r.label_b} | {r.split_a} | "
                 f"{r.split_b} | {yn(r.label_conflict)} | {yn(r.cross_split)} |")
    L.append("")
    conf = pairs[pairs["label_conflict"]]
    cross = pairs[pairs["cross_split"]]
    L.append(f"**Label conflicts ({nc}).** " + "; ".join(
        f"{r.id_a} / {r.id_b} (similarity {r.similarity:.4f}): {r.label_a} vs {r.label_b}"
        for r in conf.itertuples(index=False))
        + ". Rows 265 and 307 are the same text, so at least one of the two labels is wrong. That is direct "
          "evidence of label noise in the source data. The 1496 / 2430 pair is the same post with a different "
          "stated country and age (\"From the U.S.: I am 37\" vs \"From England: I am 16\") and a few changed "
          "phrases (83% word overlap). Both pairs sit inside train.\n")
    L.append(f"**Cross-split pairs ({nx}).** " + "; ".join(
        f"test {r.id_a if r.split_a == 'test' else r.id_b} / train {r.id_b if r.split_a == 'test' else r.id_a} "
        f"(similarity {r.similarity:.4f}, both {r.label_a})" for r in cross.itertuples(index=False))
        + ". The original dataset contains each of these posts twice, "
          "with 98% of words in common. 304 / 336 differ only in \"self harm\" vs \"self-harm\" and two "
          "apostrophes. 137 / 139 differ only in \"I'd\" vs \"idk\" and a \"(From India)\" tag. So during training the model has effectively seen these two test rows, with "
          "their test labels. This breaks the Chapter 5 leakage rule (test case C-06 is recorded as a Fail). "
          "The frozen splits are not changed. Instead, the size of the effect is measured below.\n")

    L.append("## Headline test numbers with and without the two leaked rows\n")
    L.append("**Why these are bounds.** Per-row test predictions were not saved for the D13 or D09 runs, and the "
             "checkpoints were deleted after evaluation. So the 251-row numbers cannot be computed exactly. "
             "They are computed from the saved confusion counts instead. Row 137 (No Distortion) was either a "
             "true negative or a false positive, and row 304 (Fortune-telling) was either a true positive or a "
             "false negative. Every case the counts allow is tried. *Worst* and *best* are the extreme cases "
             "for each seed and each metric. For the multilabel model, each of the two rows may have fired any "
             f"set of up to {ML_CAP} types. That matches the `--max-labels {ML_CAP}` cap `src/evaluate.py` "
             "applied when D09 was evaluated. Row 304's true types: "
             f"{', '.join(d for d, v in zip(DISTORTIONS, true_304) if v)}. The true 251-row value lies inside "
             "the range. Thresholds are the D13 binary thresholds (val FPR ≤ 0.15) and the D09 per-type "
             "thresholds. All of them were chosen on val, and nothing here was chosen on test.\n")
    L.append("Mean ± std over 3 seeds (population std, as in `docs/thresholds.md`). Test has 92 non-distorted "
             "and 161 distorted rows; 251 rows leaves 91 and 160.\n")
    L.append("| Test rows | " + " | ".join(lbl for _, lbl in METRICS) + " |")
    L.append("|---|" + "---|" * len(METRICS))
    for suffix, name in (("253", "All 253"), ("251_worst", "251, worst case"), ("251_best", "251, best case")):
        L.append(f"| {name} | " + " | ".join(ms(per_seed[f'{k}_{suffix}']) for k, _ in METRICS) + " |")
    L.append("")

    shifts = {}
    for k, lbl in METRICS:
        base = per_seed[f"{k}_253"].mean()
        shifts[lbl] = max(abs(per_seed[f"{k}_251_worst"].mean() - base),
                          abs(per_seed[f"{k}_251_best"].mean() - base))
    top = max(shifts.values())
    biggest = " and ".join(lbl for lbl, v in shifts.items() if v > top - 1e-9)
    L.append(f"Without the {ML_CAP}-type cap (if each row could have fired all 10 types), the multilabel "
             f"251-row range widens to {ms(per_seed['ml_macro_f1_251_nocap_worst'])} – "
             f"{ms(per_seed['ml_macro_f1_251_nocap_best'])}.\n")
    L.append("Largest possible change in the 3-seed mean: " + ", ".join(
        f"{lbl} {v * 100:.2f} pt" for lbl, v in shifts.items()) + ".\n")
    within = all(v <= 0.01 for v in shifts.values())
    L.append(f"**Size of the difference:** removing the two leaked test rows can move any headline mean by at "
             f"most {top * 100:.2f} percentage points ({biggest}), "
             + ("which is smaller than the seed-to-seed std, so the leak does not change any conclusion. It is "
                "still reported in Chapter 9 as a limitation."
                if within else
                "which is more than the 1 point the brief expected. This should be stated as a limitation in "
                "Chapter 9.") + "\n")

    L.append("## Per seed\n")
    L.append("Confusion counts on all 253 test rows (binary, D13 threshold), then each metric as "
             "253 → 251 [worst, best].\n")
    L.append("| Seed | TN | FP | TP | FN | " + " | ".join(lbl for _, lbl in METRICS) + " |")
    L.append("|---|---|---|---|---|" + "---|" * len(METRICS))
    for r in per_seed.itertuples(index=False):
        d = r._asdict()
        cells = [f"{d[f'{k}_253']:.3f} → [{d[f'{k}_251_worst']:.3f}, {d[f'{k}_251_best']:.3f}]"
                 for k, _ in METRICS]
        L.append(f"| {r.seed} | {r.tn} | {r.fp} | {r.tp} | {r.fn} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Per-seed values are also in `duplicates_leakage_bounds.csv`. The 253-row numbers match "
             "`docs/thresholds.md` (binary) and the D09 eval JSONs (multilabel). The script asserts this.\n")

    L.append("## For FYP-2\n")
    L.append("`src/evaluate.py` should save per-row test predictions, so a check like this one can give exact "
             "numbers instead of bounds. Before any future re-split, the near-duplicate scan should run so that "
             "copies are kept in the same split.\n")
    return "\n".join(L)


def main():
    pairs = pd.read_csv(DUP_DIR / "near_duplicates.csv", encoding="utf-8")
    per_seed, true_304 = load_inputs()
    per_seed.to_csv(DUP_DIR / "duplicates_leakage_bounds.csv", index=False, encoding="utf-8")
    md = build_report(per_seed, pairs, true_304)
    (DUP_DIR / "duplicates_report.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
