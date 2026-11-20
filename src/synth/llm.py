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


# --------------------------------------------------------------------------- #
# NVIDIA NIM (OpenAI-compatible, build.nvidia.com)                            #
# --------------------------------------------------------------------------- #
# Key: NVIDIA_NIMS_KEY (or NVIDIA_API_KEY), read from the environment or from a
# local, gitignored `.env` at the repo root. The key is never printed or logged.
# The free tier is rate-limited (~40 requests/min), so calls are paced and
# retried. Free-credit calls cost $0, so PRICES has no NVIDIA entries; the
# number of requests is counted instead (REQUEST_COUNT).

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MIN_INTERVAL = 60.0 / 35          # stay under ~35 requests/min
REQUEST_COUNT = {"nvidia": 0}
NIM_LOG = os.getenv("NIM_LOG", "1") != "0"   # one line per call: model, seconds, output tokens
_last_call = [0.0]


def _load_dotenv(path=".env") -> None:
    """Minimal KEY=VALUE reader (no dependency). Never overrides a variable already set."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


def _nvidia():
    if "nvidia" not in _clients:
        from openai import OpenAI
        _load_dotenv()
        key = os.getenv("NVIDIA_NIMS_KEY") or os.getenv("NVIDIA_API_KEY")
        if not key:
            raise RuntimeError("Set NVIDIA_NIMS_KEY (in .env or the environment) for NVIDIA NIM models.")
        _clients["nvidia"] = OpenAI(base_url=NIM_BASE_URL, api_key=key, max_retries=0, timeout=180)
    return _clients["nvidia"]


def strip_reasoning(text: str) -> str:
    """Drop <think>...</think> blocks some open models put before the answer."""
    import re
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    if "</think>" in text:                     # opening tag swallowed by the server
        text = text.split("</think>", 1)[1]
    return text.strip()


def extract_json(text: str):
    """Parse a JSON object from model text: whole string, ```json fences, or first {...} block."""
    import re
    text = strip_reasoning(text)
    for cand in (text, *re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)):
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            pass
    start = text.find("{")
    while start >= 0:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


# Per-model request options. Kimi reasons before writing by default (hundreds of
# hidden tokens for a 40-word reflection); the generator only needs the text.
NIM_EXTRA_BODY = {
    "moonshotai/kimi-k3.1-pro": {"chat_template_kwargs": {"thinking": False}},
}
NIM_TIMEOUT = 120          # seconds per attempt; the free tier queues, it rarely streams slowly
NIM_ATTEMPTS = 3           # then give up, so the caller can fall back to another model


class NimUnavailable(RuntimeError):
    """A NIM model did not answer after NIM_ATTEMPTS tries (timeouts, 5xx, overload)."""


def _nvidia_call(model, system, user, max_tokens, temperature=None, as_json=False):
    import time
    import openai
    client = _nvidia().with_options(timeout=NIM_TIMEOUT)
    kwargs = dict(model=model, max_tokens=max_tokens,
                  messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    if temperature is not None:
        kwargs["temperature"] = temperature
    if as_json:
        kwargs["response_format"] = {"type": "json_object"}
    if model in NIM_EXTRA_BODY:
        kwargs["extra_body"] = NIM_EXTRA_BODY[model]
    for attempt in range(NIM_ATTEMPTS):
        wait = NIM_MIN_INTERVAL - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = t0 = time.monotonic()
        REQUEST_COUNT["nvidia"] += 1
        try:
            resp = client.chat.completions.create(**kwargs)
            if NIM_LOG:
                u = getattr(resp, "usage", None)
                print(f"    [nim] {model} {time.monotonic() - t0:.1f}s "
                      f"out={getattr(u, 'completion_tokens', '?')}", flush=True)
            break
        except openai.BadRequestError:
            if as_json and "response_format" in kwargs:   # model doesn't support JSON mode: ask in-prompt instead
                kwargs.pop("response_format")
                continue
            raise
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                openai.InternalServerError) as e:
            if NIM_LOG:
                print(f"    [nim] {model} attempt {attempt + 1} failed after {time.monotonic() - t0:.1f}s: "
                      f"{type(e).__name__}", flush=True)
            if attempt + 1 < NIM_ATTEMPTS:
                time.sleep(5 * 2 ** attempt)
    else:
        raise NimUnavailable(f"NVIDIA NIM: {model} did not answer after {NIM_ATTEMPTS} attempts")
    msg = resp.choices[0].message
    text = strip_reasoning(msg.content or "")
    u = getattr(resp, "usage", None)
    usage = {"in": getattr(u, "prompt_tokens", 0) or 0, "out": getattr(u, "completion_tokens", 0) or 0}
    return text, usage


def nvidia_models() -> list[str]:
    """IDs the key can call right now (GET /v1/models)."""
    return sorted(m.id for m in _nvidia().models.list())


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
    if provider == "nvidia":
        return _nvidia_call(model, system, user, max_tokens, temperature=temperature)
    raise ValueError(f"unknown provider: {provider}")


def complete_json(provider, model, system, user, max_tokens, temperature=None, json_schema=None):
    """JSON completion. Returns (parsed_dict_or_None, usage)."""
    if provider == "anthropic":
        text, usage = _anthropic_call(model, system, user, max_tokens, json_schema=json_schema)
    elif provider == "gemini":
        text, usage = _gemini_call(model, system, user, max_tokens, temperature=temperature, as_json=True)
    elif provider == "nvidia":
        text, usage = _nvidia_call(model, system, user, max_tokens, temperature=temperature, as_json=True)
        return extract_json(text), usage
    else:
        raise ValueError(f"unknown provider: {provider}")
    try:
        return json.loads(text), usage
    except json.JSONDecodeError:
        return None, usage
