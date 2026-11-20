"""Tests for src/cbt_transfer_report.py (D21: CBT-Bench transfer on the D12 winner).

Fake per-seed CSVs in tmp_path stand in for the Kaggle run; no model or download needed.
"""

import numpy as np
import pandas as pd
import pytest

from src.cbt_transfer_report import (
    CBT_LABEL_CANON,
    OCT4,
    PENDING,
    PER_CLASS_COLS,
    load_seed_tables,
    main,
    per_class_rows,
    strict_canon_map,
    summarize,
    to_matrix,
)
from src.data import DISTORTIONS


def test_label_map_covers_all_ten_distortions():
    assert sorted(CBT_LABEL_CANON.values()) == sorted(DISTORTIONS)


def test_strict_canon_map_maps_and_rejects_unknown():
    assert strict_canon_map([" mind reading", "fortune-telling"]) == ["mind_reading", "fortune_telling"]
    with pytest.raises(ValueError):
        strict_canon_map(["Mind Reading"])   # case differs: no fuzzy matching


def test_per_class_rows_values():
    y_true = to_matrix([["mind_reading"], ["mind_reading", "labeling"], ["labeling"]])
    y_pred = to_matrix([["mind_reading"], ["labeling"], ["labeling", "mind_reading"]])
    rows = {r["label"]: r for r in per_class_rows(y_true, y_pred)}
    assert rows["mind_reading"]["precision"] == pytest.approx(0.5)
    assert rows["mind_reading"]["recall"] == pytest.approx(0.5)
    assert rows["mind_reading"]["support"] == 2
    assert rows["labeling"]["f1"] == pytest.approx(1.0)
    assert rows["emotional_reasoning"]["support"] == 0
    assert rows["micro"]["support"] == 4
    # micro: tp=3, fp=1, fn=1 -> 0.75
    assert rows["micro"]["f1"] == pytest.approx(0.75)
    # macro over all 10 labels (8 have no gold and no prediction -> 0)
    assert rows["macro"]["f1"] == pytest.approx((0.5 + 1.0) / 10)


def _fake_seed_csv(path, seed, f1_shift):
    rows = []
    for inp in ("ori_text", "situation_thoughts"):
        for sub, n in (("all", 1.0), ("not_in_train", 0.25)):
            for j, lab in enumerate(list(DISTORTIONS) + ["macro", "micro"]):
                f1 = 0.30 + 0.01 * j + f1_shift
                rows.append({"input": inp, "decision": "cap2", "subset": sub, "label": lab,
                             "precision": f1, "recall": f1, "f1": f1,
                             "support": int(round(OCT4[lab][3] * n))})
    pd.DataFrame(rows, columns=PER_CLASS_COLS).to_csv(path / f"cbt_per_class_{seed}.csv",
                                                     index=False, encoding="utf-8")


def test_summarize_mean_and_std(tmp_path):
    for seed, shift in ((42, 0.0), (1337, 0.1), (2024, 0.2)):
        _fake_seed_csv(tmp_path, seed, shift)
    (tmp_path / "cbt_per_class_notes.csv").write_text("ignored", encoding="utf-8")
    df = load_seed_tables(tmp_path)
    assert sorted(df.seed.unique()) == [42, 1337, 2024]
    s = summarize(df)
    r = s[(s.input == "ori_text") & (s.subset == "all") & (s.label == "macro")].iloc[0]
    assert r.n_seeds == 3
    assert r.f1 == pytest.approx(0.40 + 0.1)          # 0.30 + 0.01*10 + mean shift 0.1
    assert r.f1_std == pytest.approx(np.std([0, 0.1, 0.2], ddof=1))


def test_report_with_seeds(tmp_path):
    for seed, shift in ((42, 0.0), (1337, 0.1), (2024, 0.2)):
        _fake_seed_csv(tmp_path, seed, shift)
    out = tmp_path / "final.md"
    assert main(["--dir", str(tmp_path), "--out", str(out)]) == 0
    md = out.read_text(encoding="utf-8")
    assert PENDING not in md
    assert "42, 1337, 2024" in md
    assert "| mind_reading | 47 | 0.450 ± 0.100 |" in md      # j=5: 0.35 + mean shift 0.1
    assert "0.47 / 0.49 / 0.48" in md                          # 4 Oct column
    assert "situation" in md and "thought" in md               # format-difference line
    assert "tie" in md
    assert "| `situation_thoughts` | cap2 | not_in_train |" in md


def test_scaffold_when_no_csvs(tmp_path):
    out = tmp_path / "final.md"
    assert main(["--dir", str(tmp_path / "empty"), "--out", str(out)]) == 0
    md = out.read_text(encoding="utf-8")
    assert PENDING in md
    assert "| all_or_nothing | 65 | pending Kaggle run" in md
    assert "0.73 / 0.42 / 0.53" in md
    assert "**Macro avg**" in md and "0.31 / 0.40 / 0.33" in md


def test_missing_columns_raise(tmp_path):
    pd.DataFrame({"label": ["macro"], "f1": [0.3]}).to_csv(tmp_path / "cbt_per_class_42.csv", index=False)
    with pytest.raises(ValueError):
        load_seed_tables(tmp_path)
