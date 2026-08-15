"""Provider-agnostic LLM calls (Anthropic + Google Gemini).

Keeps the pipeline honest about *independence*: the generator (author) and the
critic panel (judge) can be different providers, so acceptance isn't one model
family grading itself. Lazy imports — you only need the SDK for the providers
you actually use.

Notes / gotchas baked in here:
* Anthropic Opus 5 / Sonnet 5 **reject** `temperature` (400). So the Anthropic
  path never sends it; per-critic decorrelation via temperature is a Gemini-only
  lever. Anthropic depth is controlled with `output_config.effort` instead.
* Anthropic system prompts are sent with `cache_control` (the definitions block
  is identical every call). Gemini has its own caching; we just pass the system.
* Gemini model IDs move fast — the defaults below may be stale for your account.
  Override with --gen-model / --critic-model and use whatever you have access to.
"""

from __future__ import annotations

import json
import os

# Approx first-party prices ($/1M tokens): (input, output). Estimate only.
# Sonnet 5 shown at the intro rate (through 2026-08-31). Gemini figures are
# approximate and IGNORE the free tier — real pilot cost on Gemini is often ~$0.
PRICES = {
    ("anthropic", "claude-sonnet-5"): (2.0, 10.0),
    ("anthropic", "claude-opus-5"): (5.0, 25.0),
    ("anthropic", "claude-haiku-4-5"): (1.0, 5.0),
    ("gemini", "gemini-2.5-flash"): (0.30, 2.50),
    ("gemini", "gemini-2.5-pro"): (1.25, 10.0),
}

_clients: dict[str, object] = {}


def _anthropic():
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.Anthropic()
    return _clients["anthropic"]


def _gemini():
    if "gemini" not in _clients:
        from google import genai
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) for the Gemini critics.")
        _clients["gemini"] = genai.Client(api_key=key)
    return _clients["gemini"]


def estimate_cost(provider: str, model: str, usage: dict) -> float:
    pi, po = PRICES.get((provider, model), (0.0, 0.0))
    return (usage.get("in", 0) / 1e6) * pi + (usage.get("out", 0) / 1e6) * po


# --------------------------------------------------------------------------- #
# Anthropic                                                                    #
# --------------------------------------------------------------------------- #

def _anthropic_call(model, system, user, max_tokens, json_schema=None):
    client = _anthropic()
    output_config = {"effort": "low"}
    if json_schema is not None:
        output_config["format"] = {"type": "json_schema", "schema": json_schema}
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        output_config=output_config,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
    usage = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
    return text, usage


# --------------------------------------------------------------------------- #
# Gemini                                                                       #
# --------------------------------------------------------------------------- #

def _gemini_call(model, system, user, max_tokens, temperature=None, as_json=False):
    from google.genai import types
    client = _gemini()
    cfg_kwargs = dict(system_instruction=system, max_output_tokens=max_tokens)
    if temperature is not None:
        cfg_kwargs["temperature"] = temperature
    if as_json:
        cfg_kwargs["response_mime_type"] = "application/json"
    resp = client.models.generate_content(
        model=model, contents=user, config=types.GenerateContentConfig(**cfg_kwargs)
    )
    try:
        text = (resp.text or "").strip()
    except Exception:  # noqa: BLE001 - safety block / no candidate
        text = ""
    um = getattr(resp, "usage_metadata", None)
    usage = {
        "in": getattr(um, "prompt_token_count", 0) or 0,
        "out": getattr(um, "candidates_token_count", 0) or 0,
    }
    return text, usage


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #

def complete_text(provider, model, system, user, max_tokens, temperature=None):
    """Free-text completion. Returns (text, usage)."""
    if provider == "anthropic":
        return _anthropic_call(model, system, user, max_tokens)  # temperature not allowed
    if provider == "gemini":
        return _gemini_call(model, system, user, max_tokens, temperature=temperature)
    raise ValueError(f"unknown provider: {provider}")


def complete_json(provider, model, system, user, max_tokens, temperature=None, json_schema=None):
    """JSON completion. Returns (parsed_dict_or_None, usage)."""
    if provider == "anthropic":
        text, usage = _anthropic_call(model, system, user, max_tokens, json_schema=json_schema)
    elif provider == "gemini":
        text, usage = _gemini_call(model, system, user, max_tokens, temperature=temperature, as_json=True)
    else:
        raise ValueError(f"unknown provider: {provider}")
    try:
        return json.loads(text), usage
    except json.JSONDecodeError:
        return None, usage
