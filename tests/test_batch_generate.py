"""CPU tests for src/synth/batch_generate.py with the fake backend (no GPU, no models)."""

import json

import pytest

from src.synth import attributes as A
from src.synth import batch_generate as B
from src.synth.generate import accepted_row, aggregate, rejected_row

SPEC = {"spec_id": "s0", "primary": "Labeling", "secondary": None, "granularity": "snippet",
        "trigger": "t", "register": "r", "stage": "st", "sector": "se", "cultural_marker": None,
        "seed_ids": ["seed_001"]}


def _args(tmp_path, n=12, **kw):
    a = B.build_parser().parse_args(["--n", str(n), "--backend", "fake", "--out", str(tmp_path / "out.jsonl"),
                                     "--chunk", "5"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _read(p):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_rows_match_sequential_schema_and_carry_seed_ids(tmp_path):
    args = _args(tmp_path)
    summary = B.run_batch(args, B.FakeBackend(), log=lambda m: None)
    rows = _read(tmp_path / "out.jsonl")
    assert summary["accepted"] == len(rows) == 12
    agg = aggregate([], 3)
    want = set(accepted_row(0, SPEC, "x", "Labeling", None, agg, [], revised=False, n_critics=3,
                            gen_model="g", critic_models=[]))
    for r in rows:
        assert set(r) == want
        assert r["seed_ids"], "S06 needs the seeds each row was prompted with"
        assert len(r["critic_models"]) == 3
        assert r["dominant_distortion"] == r["intended_primary"]
        assert r["gen_model"] == "Qwen/Qwen3-32B-AWQ"
    s = json.loads((tmp_path / "out.summary.json").read_text(encoding="utf-8"))
    assert s["n_requested"] == 12 and s["cost_usd"] == 0.0


def test_reject_revise_accept(tmp_path):
    fake = B.FakeBackend(disagree_first={"Mind Reading"})
    args = _args(tmp_path, n=22)
    summary = B.run_batch(args, fake, log=lambda m: None)
    rows = _read(tmp_path / "out.jsonl")
    mr = [r for r in rows if r["intended_primary"] == "Mind Reading"]
    assert mr and all(r["revised"] for r in mr)
    assert all(not r["revised"] for r in rows if r["intended_primary"] != "Mind Reading")
    assert summary["revised"] == len(mr)
    # the revision prompt's seeds are added to the row's seed_ids
    assert all(len(r["seed_ids"]) >= 4 for r in mr)


def test_filter_rejects_never_reach_critics(tmp_path):
    fake = B.FakeBackend(too_short={"Labeling"})
    args = _args(tmp_path, n=22)
    summary = B.run_batch(args, fake, log=lambda m: None)
    rej = _read(tmp_path / "out.rejected.jsonl")
    lab = [r for r in rej if r["intended_primary"] == "Labeling"]
    assert lab and all(r["reason"] == "length" for r in lab)
    assert all("critic_votes" not in r for r in lab)
    critic_models = {m for slot in B.DEFAULT_CRITICS.split(",") for m in slot.split("|")}
    n_critic_prompts = sum(n for m, n in fake.calls if m in critic_models)
    assert n_critic_prompts == 3 * (22 - len(lab))
    assert summary["accepted"] + summary["rejected"] == 22
    assert summary["filtered_length"] == len(lab)


def test_slot_falls_back_to_backup(tmp_path):
    fake = B.FakeBackend(fail_models={"Qwen/Qwen3-32B-AWQ", "stelterlab/phi-4-AWQ"})
    args = _args(tmp_path, n=6)
    B.run_batch(args, fake, log=lambda m: None)
    rows = _read(tmp_path / "out.jsonl")
    assert rows and all(r["gen_model"] == "Qwen/Qwen3-14B-AWQ" for r in rows)
    # phi-4 has no backup: that critic is missing, 2 of 3 still decide
    assert all(len(r["critic_models"]) == 2 for r in rows)


def test_clean_generation_strips_quotes_and_thinking():
    assert B.clean_generation('<think>x</think>\n"I lost the deal."') == "I lost the deal."
    assert B.clean_generation("“Quoted.”") == "Quoted."
    assert B.clean_generation("Plain.") == "Plain."


def test_request_response_roundtrip(tmp_path):
    B.write_requests(tmp_path / "r.jsonl", [("sys", "user ü")])
    assert json.loads((tmp_path / "r.jsonl").read_text(encoding="utf-8"))["user"] == "user ü"
    (tmp_path / "o.jsonl").write_text(json.dumps({"text": "hi"}) + "\n", encoding="utf-8")
    assert B.read_responses(tmp_path / "o.jsonl") == ["hi"]


def test_rejected_row_has_no_verdict_for_filter_rejects():
    row = rejected_row(SPEC, "length", text="x", detail="1 words", gen_model="g")
    assert "critic_votes" not in row and row["seed_ids"] == ["seed_001"]
    assert A.NO_DISTORTION  # attributes import is used
