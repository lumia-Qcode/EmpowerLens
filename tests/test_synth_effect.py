"""Tests for the S08 code: make_splits_synth, eval_supplementary (label mapping), synth_effect (aggregate)."""

import json

import numpy as np
import pandas as pd

from src import eval_supplementary as E
from src import make_splits_synth as M
from src import synth_effect as A


def test_synthetic_rows_map_to_benchmark_columns():
    cols = list(pd.read_csv("data/splits/train.csv", nrows=1, encoding="utf-8-sig").columns)
    df = M.synthetic_rows([
        {"id": "syn_1", "text": "t1", "dominant_distortion": "Labeling", "secondary_distortion": "Mind Reading"},
        {"id": "syn_2", "text": "t2", "dominant_distortion": "No Distortion", "secondary_distortion": None},
    ], cols)
    assert list(df.columns) == cols
    assert list(df.y_bin) == [1, 0] and list(df.y_mc) == [10, 0]
    assert df.loc[0, "ml_labeling"] == 1 and df.loc[0, "ml_mind_reading"] == 1
    assert df.loc[0, [c for c in cols if c.startswith("ml_")]].sum() == 2
    assert df.loc[1, [c for c in cols if c.startswith("ml_")]].sum() == 0


def test_build_train_only_adds_synthetic_and_keeps_val(tmp_path):
    m = M.build(out=tmp_path / "s")
    assert m["n_train"] == m["n_train_benchmark"] + m["n_train_synthetic"]
    assert not (tmp_path / "s" / "test.csv").exists()
    assert (tmp_path / "s" / "val.csv").read_bytes() == open("data/splits/val.csv", "rb").read()


def test_supplementary_gold_maps_calibrated_pessimism():
    df = pd.DataFrame({"dominant": ["No Distortion (calibrated pessimism)", "Labeling", "No Distortion"],
                       "secondary": [np.nan, "Fortune-telling", np.nan]})
    y_bin, y_ml = E.gold(df)
    assert list(y_bin) == [0, 1, 0]
    assert y_ml[0].sum() == 0 and y_ml[1].sum() == 2


def test_binary_rates():
    r = E.binary_rates([0, 0, 1, 1], [0, 1, 1, 0])
    assert r["specificity"] == 0.5 and r["fpr"] == 0.5 and r["sensitivity"] == 0.5


def _write(raw, cond, seed, spec, sens, mf1):
    pcb = [{"class": "non_distorted", "recall": spec}, {"class": "distorted", "recall": sens}]
    for task, splits in (("binary", {s: {"per_class": pcb, "metrics": {}} for s in ("val", "test")}),
                         ("multilabel", {s: {"metrics": {"macro_f1": mf1}} for s in ("val", "test")})):
        p = raw / cond / "eval" / f"eval_mental-roberta-base_{task}_{seed}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"splits": splits}))
    s = raw / cond / "supp" / f"supp_eval_{cond}_{seed}.json"
    s.parent.mkdir(parents=True, exist_ok=True)
    s.write_text(json.dumps({"binary": {"specificity": spec, "fpr": 1 - spec, "sensitivity": sens},
                             "multilabel": {"macro_f1": mf1},
                             "per_class": [{"class": "labeling", "f1": mf1, "support": 3}]}))


def test_aggregate_mean_std_and_paired(tmp_path):
    for seed, (a, b) in zip((42, 1337, 2024), ((0.2, 0.3), (0.3, 0.4), (0.4, 0.5))):
        _write(tmp_path, "train_only", seed, 0.8, 0.6, a)
        _write(tmp_path, "train_synth", seed, 0.8, 0.6, b)
    per_seed, pcs = A.collect(tmp_path)
    assert len(per_seed) == 2 * 3 * 3
    s = A.summarise(per_seed).set_index(["condition", "split"])
    assert abs(s.loc[("train_synth", "test"), "multilabel_macro_f1_mean"] - 0.4) < 1e-9
    assert abs(s.loc[("train_only", "val"), "multilabel_macro_f1_std"] - 0.1) < 1e-9
    d = A.paired(per_seed)
    assert np.allclose(d.multilabel_macro_f1, 0.1) and np.allclose(d.specificity, 0)
    pc = A.per_class(pcs)
    assert pc.loc[0, "support"] == 3
