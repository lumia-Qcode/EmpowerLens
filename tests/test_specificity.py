"""Tests for src/specificity.py.

Specificity is read straight out of the eval JSON as the recall of the
``non_distorted`` class; these tests pin that mapping and the counts derived
from it, on hand-built files so no real results are needed.
"""

import json

import pytest

from src.specificity import build_meanstd, load_binary_rows


def _eval(task, seed, nd_recall, nd_support=92, d_recall=0.75, d_support=161, model="roberta-base"):
    return {
        "meta": {"task": task, "model": model, "seed": seed},
        "splits": {"test": {"per_class": [
            {"class": "non_distorted", "recall": nd_recall, "support": nd_support},
            {"class": "distorted", "recall": d_recall, "support": d_support},
        ]}},
    }


def _write(tmp_path, name, data):
    (tmp_path / name).write_text(json.dumps(data), encoding="utf-8")


def test_specificity_is_non_distorted_recall(tmp_path):
    _write(tmp_path, "eval_roberta-base_binary_42.json", _eval("binary", 42, 54 / 92))
    df = load_binary_rows(str(tmp_path))
    r = df.iloc[0]
    assert r["specificity"] == pytest.approx(54 / 92)
    assert r["fpr"] == pytest.approx(38 / 92)
    assert (r["tn"], r["fp"], r["n_non_distorted"]) == (54, 38, 92)
    assert r["sensitivity"] == pytest.approx(0.75)


def test_non_binary_runs_are_skipped(tmp_path):
    _write(tmp_path, "eval_roberta-base_binary_42.json", _eval("binary", 42, 0.6))
    _write(tmp_path, "eval_roberta-base_multiclass_42.json", {"meta": {"task": "multiclass"}, "splits": {}})
    assert len(load_binary_rows(str(tmp_path))) == 1


def test_missing_non_distorted_row_fails_loudly(tmp_path):
    bad = _eval("binary", 42, 0.6)
    bad["splits"]["test"]["per_class"] = bad["splits"]["test"]["per_class"][1:]
    _write(tmp_path, "eval_roberta-base_binary_42.json", bad)
    with pytest.raises(ValueError, match="non_distorted"):
        load_binary_rows(str(tmp_path))


def test_meanstd_uses_population_std(tmp_path):
    for seed, tn in [(42, 54), (1337, 65), (2024, 62)]:
        _write(tmp_path, f"eval_roberta-base_binary_{seed}.json", _eval("binary", seed, tn / 92))
    ms = build_meanstd(load_binary_rows(str(tmp_path)))
    r = ms.iloc[0]
    assert r["n_seeds"] == 3
    assert r["specificity"] == "0.656 ± 0.050"
    assert r["mean_tn"] == "60.3 of 92"
