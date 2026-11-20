"""Tests for src/specificity_audit.py (D27 row D-04), on hand-built eval files."""

import json

import pytest

from src.specificity_audit import audit


def _write(path, task, per_class, extra=None, seed=42):
    split = {"metrics": {"task": task, "macro_f1": 0.5}, "per_class": per_class}
    split.update(extra or {})
    data = {"meta": {"task": task, "model": "m", "seed": seed}, "splits": {"val": split, "test": split}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_binary_multiclass_multilabel(tmp_path):
    _write(tmp_path / "a" / "eval_m_binary_42.json", "binary",
           [{"class": "non_distorted", "recall": 0.8}, {"class": "distorted", "recall": 0.6}])
    _write(tmp_path / "a" / "eval_m_multiclass_42.json", "multiclass",
           [{"class": "no_distortion", "recall": 0.7}, {"class": "labeling", "recall": 0.2}])
    _write(tmp_path / "b" / "eval_m_multilabel_42.json", "multilabel",
           [{"class": "labeling", "recall": 0.3}], extra={"macro_specificity_10": 0.9})
    _write(tmp_path / "b" / "eval_m_multilabel_1337.json", "multilabel",
           [{"class": "labeling", "recall": 0.3}], seed=1337)
    df = audit([tmp_path]).set_index(["task", "seed", "split"])
    assert df.loc[("binary", 42, "test"), "specificity"] == pytest.approx(0.8)
    assert df.loc[("binary", 42, "test"), "fpr"] == pytest.approx(0.2)
    assert df.loc[("multiclass", 42, "test"), "specificity"] == pytest.approx(0.7)
    assert df.loc[("multilabel", 42, "test"), "specificity"] == pytest.approx(0.9)
    assert df.loc[("multilabel", 42, "test"), "basis"] == "macro_specificity_10 (per label)"
    assert df.loc[("multilabel", 1337, "test"), "basis"] == "not recorded"
    assert df.loc[("multilabel", 1337, "test"), "specificity"] != df.loc[("multilabel", 1337, "test"), "specificity"]  # NaN


def test_skips_non_classifier_files(tmp_path):
    p = tmp_path / "eval_val_42.json"
    p.write_text(json.dumps({"precision": 0.4, "threshold": 0.7}), encoding="utf-8")
    assert audit([tmp_path]).empty
