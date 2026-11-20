"""Tests for src/truncation_report.py (D15: head vs head_tail at max length 256).

Hand-built meta.json files stand in for the Kaggle runs, so no real results or
tokenizer download are needed.
"""

import json

import pytest

from src.truncation_report import ARMS, SEEDS, cut_share, summarize, token_lengths


def test_cut_share_budget_boundary():
    # max_length 10 with 2 special tokens leaves a budget of 8 text tokens.
    assert cut_share([8], 10) == 0.0       # exactly at budget: not cut
    assert cut_share([9], 10) == 1.0       # one over: cut
    assert cut_share([1, 8, 9, 20], 10) == 0.5


def test_cut_share_empty_raises():
    with pytest.raises(ValueError):
        cut_share([], 256)


class _FakeTok:
    def __call__(self, texts, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [t.split() for t in texts]}


def test_token_lengths_excludes_special_tokens():
    assert token_lengths(["a b c", "d"], _FakeTok()) == [3, 1]


def _meta(trunc, seed, mf1, mf1_10, **over):
    m = {
        "task": "multiclass", "model": "mental/mental-roberta-base", "splits": "data/splits",
        "seed": seed, "max_length": 256, "truncation": trunc, "best_epoch": 5.0,
        "val_truncation_rate": 0.2,
        "val_metrics": {"eval_macro_f1": mf1, "eval_macro_f1_10": mf1_10},
    }
    m.update(over)
    return m


def _write_all(root, overrides=None):
    vals = {"head": (0.20, 0.30, 0.40), "head_tail": (0.25, 0.25, 0.25)}
    for arm in ARMS:
        for seed, v in zip(SEEDS, vals[arm]):
            d = root / arm / f"seed_{seed}"
            d.mkdir(parents=True)
            meta = _meta(arm, seed, v, v - 0.05)
            if overrides and (arm, seed) in overrides:
                meta.update(overrides[(arm, seed)])
            (d / "meta.json").write_text(json.dumps(meta))


def test_summarize_mean_std(tmp_path):
    _write_all(tmp_path)
    df = summarize(tmp_path)
    assert len(df) == 8  # 6 seed rows + 2 mean rows
    head = df[(df.arm == "head") & (df.seed == "mean")].iloc[0]
    assert head.macro_f1 == pytest.approx(0.30)
    assert head.macro_f1_std == pytest.approx(0.10)  # sample std (ddof=1)
    ht = df[(df.arm == "head_tail") & (df.seed == "mean")].iloc[0]
    assert ht.macro_f1_std == pytest.approx(0.0)
    assert ht.macro_f1_10 == pytest.approx(0.20)


def test_summarize_missing_seed_raises(tmp_path):
    _write_all(tmp_path)
    (tmp_path / "head_tail" / "seed_2024" / "meta.json").unlink()
    with pytest.raises(FileNotFoundError):
        summarize(tmp_path)


@pytest.mark.parametrize("bad", [
    {"max_length": 512},
    {"truncation": "head_tail"},
    {"task": "multilabel"},
    {"splits": "data/splits_relabelled"},
])
def test_summarize_rejects_mismatched_run(tmp_path, bad):
    _write_all(tmp_path, overrides={("head", 1337): bad})
    with pytest.raises(ValueError):
        summarize(tmp_path)
