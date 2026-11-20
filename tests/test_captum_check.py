"""D22 captum_check: alignment, agreement and the 20-row selection rule (no model)."""

import pandas as pd
import pytest

from src import captum_check as cc


def _sents(text, row_id=1):
    """Split on '. ' and record offsets, like sentence_labels_val.csv."""
    out, pos = [], 0
    for i, part in enumerate(p for p in text.split(". ") if p):
        s = text.index(part, pos)
        e = s + len(part) + (1 if text[s + len(part):s + len(part) + 1] == "." else 0)
        out.append({"Id_Number": row_id, "sent_idx": i, "sent_start": s, "sent_end": e, "sentence": text[s:e]})
        pos = e
    return pd.DataFrame(out)


def _tokens(text):
    """Fake tokenizer: each word split into 2-char pieces, punctuation as own token."""
    offs = []
    for s, e, _ in cc.word_spans(text):
        for k in range(s, e, 2):
            offs.append((k, min(k + 2, e)))
    offs += [(i, i + 1) for i, ch in enumerate(text) if ch in ".!?,"]
    return sorted(offs)


def test_aggregate_row_words_and_sentences():
    text = "I am fine. Nobody ever likes me. I always fail."
    sents = _sents(text)
    assert list(sents["sentence"]) == ["I am fine.", "Nobody ever likes me.", "I always fail."]
    offs = _tokens(text)
    scores = [0.0] * len(offs)
    # put weight on "Nobody" (3 pieces) and "always" (3 pieces)
    for i, (s, e) in enumerate(offs):
        if text[s:e] in ("No", "bo", "dy"):
            scores[i] = 1.0
        if s >= text.index("always") and e <= text.index("always") + 6:
            scores[i] = 0.5
    # specials at both ends must be ignored
    offsets = [None] + offs + [None]
    scores = [99.0] + scores + [99.0]

    words, sent = cc.aggregate_row(1, text, offsets, scores, sents)
    w = dict(zip(words["word"], words["attr"]))
    assert w["Nobody"] == pytest.approx(3.0)
    assert w["always"] == pytest.approx(1.5)
    assert w["fine"] == 0.0
    assert cc.top_words(words, 2) == ["Nobody", "always"]
    assert list(sent["attr_total"]) == pytest.approx([0.0, 3.0, 1.5])
    assert sent["n_tokens"].sum() == len(offs)            # every non-special token assigned once
    assert words.loc[words["word"] == "always", "sent_idx"].item() == 2


def test_assign_sentence_gap_and_empty():
    sents = pd.DataFrame({"sent_idx": [0, 1], "sent_start": [0, 10], "sent_end": [8, 20]})
    assert cc.assign_sentence(3, sents) == 0
    assert cc.assign_sentence(9, sents) == 0        # gap -> preceding sentence
    assert cc.assign_sentence(15, sents) == 1
    assert cc.assign_sentence(3, sents.iloc[0:0]) == -1


def test_word_spans_keep_curly_apostrophes():
    assert [w for *_, w in cc.word_spans("I don’t can't do it.")] == ["I", "don’t", "can't", "do", "it"]


def test_agreement_counts_and_ties():
    selector = pd.DataFrame({
        "Id_Number": [1, 1, 1, 2, 2, 3, 3, 4],
        "sent_idx":  [0, 1, 2, 0, 1, 0, 1, 0],
        "score":     [.1, .9, .2, .5, .5, .3, .1, .7],
    })
    attributions = pd.DataFrame({
        "Id_Number": [1, 1, 1, 2, 2, 3, 3, 5],
        "sent_idx":  [0, 1, 2, 0, 1, 0, 1, 0],
        "attr_total": [-1, 2, 0.5, 0.2, 0.1, 0.0, 0.4, 1.0],
    })
    a = cc.agreement(selector, attributions)
    # row 1: 1 vs 1 agree; row 2: tie -> sent 0 vs 0 agree; row 3: 0 vs 1 disagree; rows 4/5 not in both
    assert (a["agree"], a["n"]) == (2, 3)
    assert cc.fmt_agreement(a) == "2 of 3 rows (66.7%)"
    assert a["chance"] == pytest.approx((1 / 3 + 1 / 2 + 1 / 2) / 3)


def _val(n=40):
    rows = []
    for i in range(n):
        if i % 4 == 0:
            text = "No distortion here, I always relax."      # absolute but y_bin 0
            yb = 0
        elif i % 4 == 1:
            text = "Things went badly today."                  # no absolute word
            yb = 1
        elif i % 4 == 2:
            text = "Nobody ever calls me. I'm NEVER good enough."
            yb = 1
        else:
            text = "I finally finished, all of it, overall."   # 'all' whole word; 'finally'/'overall' are not
            yb = 1
        rows.append({"Id_Number": 1000 - i, "Patient Question": text, "y_bin": yb})
    return pd.DataFrame(rows)


def test_absolute_words_whole_word_only():
    assert cc.absolute_words_in("I finally finished overall") == []
    assert cc.absolute_words_in("NEVER again, Everyone knows") == ["never", "everyone"]


def test_select_check_rows_rule():
    val = _val(40)
    sel = cc.select_check_rows(val, n=8, seed=42)
    eligible = val[(val["y_bin"] == 1) & val["Patient Question"].map(lambda t: bool(cc.absolute_words_in(t)))]
    assert len(eligible) == 20
    assert len(sel) == 8
    assert set(sel["Id_Number"]) <= set(eligible["Id_Number"])
    assert list(sel["Id_Number"]) == sorted(sel["Id_Number"])
    # deterministic and independent of input row order
    shuffled = val.sample(frac=1, random_state=0)
    assert list(cc.select_check_rows(shuffled, n=8, seed=42)["Id_Number"]) == list(sel["Id_Number"])
    # fewer eligible than n -> all eligible
    assert len(cc.select_check_rows(val, n=50)) == 20


def test_report_scaffold_and_verdicts_kept(tmp_path, monkeypatch):
    val = _val(40)
    splits = tmp_path / "splits"
    splits.mkdir()
    val.to_csv(splits / "val.csv", index=False, encoding="utf-8")
    monkeypatch.setattr(cc, "SPLITS_DIR", splits)
    out = tmp_path / "out"
    args = ["report", "--attributions", str(tmp_path / "missing.csv"), "--out-dir", str(out)]
    assert cc.main(args) == 0
    md = (out / "captum_check.md").read_text(encoding="utf-8")
    assert "pending Kaggle run" in md and "never shown to a trainer" in md
    rows = pd.read_csv(out / "captum_check_rows.csv", dtype={"verdict": str})
    assert len(rows) == 20

    # human fills a verdict and a summary; a re-run keeps both
    rows.loc[0, "verdict"] = "cue"
    rows.to_csv(out / "captum_check_rows.csv", index=False)
    md = md.replace(cc.SUMMARY_PLACEHOLDER, "My summary.")
    # a verdict typed into the md table (not the CSV) is kept too
    second = str(rows["Id_Number"].iloc[1])
    md = "\n".join(l[:-3] + " topic |" if l.startswith(f"| {second} |") else l for l in md.split("\n"))
    (out / "captum_check.md").write_text(md, encoding="utf-8")
    assert cc.main(args) == 0
    md2 = (out / "captum_check.md").read_text(encoding="utf-8")
    assert "My summary." in md2 and "| cue |" in md2
    kept = pd.read_csv(out / "captum_check_rows.csv")["verdict"]
    assert list(kept.iloc[:2]) == ["cue", "topic"]
