"""Tests for the NVIDIA NIM helpers in src/synth/llm.py (no network)."""

import os

from src.synth import llm


def test_strip_reasoning_removes_think_blocks():
    assert llm.strip_reasoning("<think>pondering</think>\nThe answer.") == "The answer."
    assert llm.strip_reasoning("half a thought</think> Final.") == "Final."
    assert llm.strip_reasoning("Plain text.") == "Plain text."


def test_extract_json_variants():
    assert llm.extract_json('{"a": 1}') == {"a": 1}
    assert llm.extract_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert llm.extract_json('<think>x {not json}</think> verdict: {"primary": "Labeling", "e": ["a}b"]} done') == \
        {"primary": "Labeling", "e": ["a}b"]}
    assert llm.extract_json("no json here") is None


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nNIM_TEST_A="from file"\nexport NIM_TEST_B=b\n', encoding="utf-8")
    monkeypatch.setenv("NIM_TEST_B", "already set")
    monkeypatch.delenv("NIM_TEST_A", raising=False)
    llm._load_dotenv(str(env))
    assert os.environ["NIM_TEST_A"] == "from file"
    assert os.environ["NIM_TEST_B"] == "already set"
