"""D19: challenge-set sheet blinding and scoring (no model needed)."""
import pandas as pd

from src import challenge_set as cs

TYPES = ["Labeling", "Mind Reading"]


def _pairs(n_per_type=5):
    rows = []
    for t in TYPES:
        for i in range(n_per_type):
            rows.append({"pair_id": f"{t[:2]}-{i}", "distortion_type": t, "situation": "s",
                         "justified_text": f"justified {t} {i} facts support it",
                         "distorted_text": f"distorted {t} {i} nobody ever will", "source_file": "x.csv"})
    return pd.DataFrame(rows)


def test_items_are_shuffled_unique_and_complete():
    items = cs.build_items(_pairs())
    assert len(items) == 20 and items["item_id"].is_unique
    assert items["item_id"].iloc[0] == "c001"
    assert (items.groupby("pair_id")["intended"].apply(set) == {"justified", "distorted"}).all()
    assert cs.build_items(_pairs())["text"].tolist() == items["text"].tolist()  # fixed seed


def test_sheet_is_blind_and_key_round_trips(tmp_path):
    items = cs.build_items(_pairs())
    cs.write_sheet(items, tmp_path)
    sheet = pd.read_excel(tmp_path / "annotator_sheet.xlsx", sheet_name="Label")
    assert list(sheet.columns) == ["item_id", "text", "label", "notes"]
    assert sheet["label"].isna().all()
    key = pd.read_csv(tmp_path / "annotator_key.csv")
    assert set(key["item_id"]) == set(sheet["item_id"])
    merged = sheet.merge(key, on="item_id").merge(items[["item_id", "text"]], on="item_id", suffixes=("", "_i"))
    assert (merged["text"] == merged["text_i"]).all()


def test_annotator_agreement_and_seed_counts():
    items = cs.build_items(_pairs())
    key = items[["item_id", "pair_id", "distortion_type", "intended"]]
    ann = key[["item_id"]].assign(annot=key["intended"])
    ann.loc[0, "annot"] = "justified" if key.loc[0, "intended"] == "distorted" else "distorted"
    a = cs.annotator_agreement(key, ann)
    assert a["n_agree"] == 19 and a["n_labelled"] == 20 and a["disagree_pairs"] == [key.loc[0, "pair_id"]]

    # model flags every distorted text and one justified Labeling text
    pred = (key["intended"] == "distorted").astype(int)
    first_j = key[(key["intended"] == "justified") & (key["distortion_type"] == "Labeling")].index[0]
    pred.loc[first_j] = 1
    c = cs.seed_counts(key, key[["item_id"]].assign(pred=pred)).set_index("distortion_type")
    assert (c.loc["Labeling", ["spec_k", "spec_n", "sens_k", "sens_n"]].tolist() == [4, 5, 5, 5])
    assert (c.loc["Mind Reading", ["spec_k", "sens_k"]].tolist() == [5, 5])


def test_label_normalisation():
    assert cs._norm_label("Not distorted") == "justified"
    assert cs._norm_label("distorted") == "distorted"
    assert cs._norm_label(float("nan")) is None


def test_render_reports_pending_and_counts():
    items = cs.build_items(_pairs())
    key = items[["item_id", "pair_id", "distortion_type", "intended"]]
    leak = {"check": "c", "target": "t", "n_items": 20, "f1_distorted": 0.2, "limit": 0.75,
            "verdict": "OK", "per_file_f1": {"x.csv": 0.2}}
    assert "Pending Kaggle run" in cs.render_results(key, leak, None, {})
    pred = (key["intended"] == "distorted").astype(int)
    counts = cs.seed_counts(key, key[["item_id"]].assign(pred=pred))
    per_seed = {s: {"counts": counts, "threshold": 0.9, "model": "m"} for s in (42, 1337)}
    out = cs.render_results(key, leak, None, per_seed)
    assert "10/10" in out and "1.000 ± 0.000" in out and "| Labeling | 5/5" in out
