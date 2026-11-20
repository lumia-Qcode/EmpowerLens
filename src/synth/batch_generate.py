"""Batch synthetic-data generation on a local GPU (Kaggle T4 x2) with open models.

Same pipeline as ``generate.py`` — same planner specs, prompts, filters,
2-of-3 blind critic panel, one revision, same output row schema — but run in
**phases** so only one model is on the GPU at a time:

    generate (chunks) -> prefilter -> critic A, B, C over every survivor -> decide
      -> revise the failures -> critic A, B, C over the revisions -> decide -> write

Holding a generator and three critics at once would force every model down to
~7B on 2 x 16 GB; one model at a time across both GPUs (tensor parallel 2)
allows ~24-32B 4-bit models. Each phase runs the model in a **fresh worker
subprocess** (``--worker``), which exits when done, so GPU memory is fully
released between models (vLLM does not reliably free a tensor-parallel engine
in-process).

Differences from the sequential run (documented in docs/synth_calibration.md):
* the opener-diversity nudge is updated per chunk of ``--chunk`` specs, not per row;
* dedup runs in spec order against the seeds + earlier texts that passed the
  filters (slightly stricter than "earlier accepted texts").

Run (Kaggle notebook ``notebooks/kaggle_runner_synth.ipynb``)::

    python -m src.synth.batch_generate --n 20 --out /kaggle/working/synth/calibrate.jsonl

CPU check with a fake backend (no GPU, no models)::

    python -m src.synth.batch_generate --n 5 --backend fake --out /tmp/fake.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import attributes as A
from . import llm
from . import planner as PL
from . import prompts as P
from .generate import (RunStats, accepted_row, aggregate, decide, enforce_evidence,
                       failure_reason, load_seeds, parse_critic_slots, prefilter,
                       print_plan, rejected_row)

# Defaults: four model families (Qwen generator; Mistral, Microsoft, Meta critics),
# 4-bit AWQ checkpoints that run in fp16 on a T4. "|" lists a backup tried when a
# model fails to download or load. Gemma is avoided: it overflows in fp16 and the
# T4 has no bf16.
DEFAULT_GEN = "Qwen/Qwen3-32B-AWQ|Qwen/Qwen3-14B-AWQ"
DEFAULT_CRITICS = ("casperhansen/mistral-small-24b-instruct-2501-awq,"
                   "stelterlab/phi-4-AWQ,"
                   "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4|ibm-granite/granite-3.3-8b-instruct")

GEN_MAX_TOKENS = 800
CRITIC_MAX_TOKENS = 1500


# --------------------------------------------------------------------------- #
# Backends                                                                     #
# --------------------------------------------------------------------------- #

class BackendError(RuntimeError):
    """A model could not be loaded or run (the caller tries the slot's backup)."""


class FakeBackend:
    """Deterministic stand-in for tests and CPU dry checks.

    The generator writes a unique filler text of the right length tagged with the
    target label; the critics read the tag back. ``disagree_first`` lists target
    labels whose *first* attempt the critics read as No Distortion (to exercise
    the revise path); ``fail_models`` simulates models that won't load.
    """

    def __init__(self, disagree_first=(), fail_models=(), too_short=()):
        self.disagree_first = set(disagree_first)
        self.fail_models = set(fail_models)
        self.too_short = set(too_short)
        self.calls: list[tuple[str, int]] = []      # (model, n_prompts)
        self._label_of: dict[str, tuple[str, bool]] = {}
        self._n = 0

    def chat(self, model, prompts, *, max_tokens, temperature=None, json_schema=None):
        if model in self.fail_models:
            raise BackendError(f"fake: {model} failed to load")
        self.calls.append((model, len(prompts)))
        return [self._critic(u) if json_schema else self._gen(u) for _, u in prompts]

    def _gen(self, user):
        m = re.search(r"Target pattern: (.+?) — ", user)
        label = m.group(1) if m else A.NO_DISTORTION
        revised = "Previous attempt" in user
        n_words = 150 if "about 150 words" in user else 40
        if label in self.too_short and not revised:
            n_words = 5
        self._n += 1
        text = " ".join(f"w{self._n}x{k}" for k in range(n_words)) + "."
        self._label_of[text] = (label, revised)
        return text

    def _critic(self, user):
        text = user.split("\n\n", 1)[1]
        label, revised = self._label_of.get(text, (A.NO_DISTORTION, False))
        if label in self.disagree_first and not revised:
            label = A.NO_DISTORTION
        return json.dumps({"facts": "f", "thoughts": "t", "reasoning": "r",
                           "evidence": [text[:20], text[20:40]], "present": [label],
                           "primary": label, "secondary": None,
                           "distorted_span": text[:15], "confidence": 0.9})


class VllmBackend:
    """Runs each batch in a fresh ``--worker`` subprocess that loads one model with vLLM."""

    def __init__(self, tp=2, max_model_len=6144, gpu_mem=0.90, workdir=None, log=print):
        self.tp, self.max_model_len, self.gpu_mem = tp, max_model_len, gpu_mem
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="synth_batch_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.log = log
        self._k = 0

    def chat(self, model, prompts, *, max_tokens, temperature=None, json_schema=None):
        self._k += 1
        req = self.workdir / f"req_{self._k:03d}.jsonl"
        resp = self.workdir / f"resp_{self._k:03d}.jsonl"
        write_requests(req, prompts)
        cmd = [sys.executable, "-m", "src.synth.batch_generate", "--worker", str(req), str(resp),
               "--worker-model", model, "--max-tokens", str(max_tokens),
               "--tp", str(self.tp), "--max-model-len", str(self.max_model_len),
               "--gpu-mem", str(self.gpu_mem)]
        if temperature is not None:
            cmd += ["--temperature", str(temperature)]
        if json_schema is not None:
            cmd += ["--json"]
        t0 = time.monotonic()
        self.log(f"    [vllm] {model}: {len(prompts)} prompts ...")
        r = subprocess.run(cmd)
        if r.returncode != 0 or not resp.exists():
            raise BackendError(f"{model}: worker exited with code {r.returncode}")
        out = read_responses(resp)
        if len(out) != len(prompts):
            raise BackendError(f"{model}: {len(out)} responses for {len(prompts)} prompts")
        self.log(f"    [vllm] {model}: done in {time.monotonic() - t0:.0f}s")
        return out


def write_requests(path, prompts) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for system, user in prompts:
            fh.write(json.dumps({"system": system, "user": user}, ensure_ascii=False) + "\n")


def read_responses(path) -> list[str]:
    return [json.loads(line)["text"] for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _sampling_params(max_tokens, temperature, json_schema):
    """SamplingParams with JSON-schema constrained decoding, across vLLM API versions."""
    from vllm import SamplingParams
    kw = dict(max_tokens=max_tokens)
    if temperature is not None:
        kw["temperature"] = temperature
    if json_schema is not None:
        try:                                    # newer vLLM
            from vllm.sampling_params import StructuredOutputsParams
            kw["structured_outputs"] = StructuredOutputsParams(json=json_schema)
        except ImportError:                     # older vLLM
            from vllm.sampling_params import GuidedDecodingParams
            kw["guided_decoding"] = GuidedDecodingParams(json=json_schema)
    return SamplingParams(**kw)


def worker(args) -> None:
    """Load one model, answer every request in the file, write responses, exit."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from vllm import LLM
    reqs = [json.loads(line) for line in Path(args.worker[0]).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    engine = LLM(model=args.worker_model, tensor_parallel_size=args.tp, dtype="half",
                 max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
                 trust_remote_code=False, seed=0)
    messages = [[{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}]
                for r in reqs]
    schema = P.CRITIC_SCHEMA if args.json else None
    try:
        sp = _sampling_params(args.max_tokens, args.temperature, schema)
    except Exception as e:  # noqa: BLE001 - constrained decoding unavailable: ask in-prompt only
        print(f"[worker] structured output unavailable ({e}); parsing JSON from free text")
        sp = _sampling_params(args.max_tokens, args.temperature, None)
    # Qwen3 thinks before answering unless told not to; other templates ignore the flag.
    outs = engine.chat(messages, sp, use_tqdm=True, chat_template_kwargs={"enable_thinking": False})
    with Path(args.worker[1]).open("w", encoding="utf-8") as fh:
        for o in outs:
            fh.write(json.dumps({"text": o.outputs[0].text}, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #

def clean_generation(text: str) -> str:
    """Strip reasoning blocks and quotes wrapped round the whole reflection."""
    text = llm.strip_reasoning(text or "").strip()
    if len(text) > 1 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    if text[:1] == "“" and text[-1:] == "”":
        text = text[1:-1].strip()
    return text


def run_slot(backend, slot, prompts, *, log=print, **kw):
    """Try each model of a slot in order; return (model_used, outputs)."""
    last = None
    for model in slot:
        try:
            return model, backend.chat(model, prompts, **kw)
        except BackendError as e:
            log(f"    [slot] {e}; trying the slot's next model")
            last = e
    raise BackendError(f"no model in slot {slot} worked: {last}")


def critique_all(backend, critics, texts, *, log=print):
    """Run every critic slot over all texts. Returns (verdicts per text, models per text)."""
    verdicts = [[] for _ in texts]
    panels = [[] for _ in texts]
    if not texts:
        return verdicts, panels
    prompts = [(P.SYSTEM_CRITIC, P.build_critic_user(t)) for t in texts]
    for slot, temp in critics:
        try:
            model, outs = run_slot(backend, slot, prompts, log=log, max_tokens=CRITIC_MAX_TOKENS,
                                   temperature=temp, json_schema=P.CRITIC_SCHEMA)
        except BackendError as e:          # a whole critic missing: like a NIM slot that never answered
            log(f"    [panel] {e}; this critic is missing for this round")
            continue
        for j, out in enumerate(outs):
            v = llm.extract_json(out)
            if not isinstance(v, dict):
                continue
            v["critic_model"] = model
            verdicts[j].append(enforce_evidence(v))
            panels[j].append(model)
    return verdicts, panels


def run_batch(args, backend, log=print) -> dict:
    seeds = load_seeds(args.seeds, english_only=True)
    specs = PL.build_specs(args.n, seed=args.seed)
    rng = random.Random(args.seed)
    gen_slot = [m.strip() for m in args.gen_model.split("|") if m.strip()]
    slots = parse_critic_slots(args.critic_models)
    temps = [float(t) for t in args.critic_temps.split(",") if t.strip()]
    critics = [(slot, temps[i % len(temps)]) for i, slot in enumerate(slots)]
    n_critics = len(critics)

    print_plan(specs, seeds)
    log(f"[panel] generator = {'|'.join(gen_slot)}")
    log("[panel] critics   = " + ", ".join(f"{'|'.join(s)}@{t}" for s, t in critics))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rej_path = out_path.with_suffix(".rejected.jsonl")
    timings: dict[str, float] = {}
    stats = RunStats(n_critics)
    nudge = RunStats(n_critics)          # opener nudges come from generated texts, per chunk
    accepted_texts = [s["text"] for s in seeds]
    gen_used: list[str] = []

    def _generate(items, revise=False):
        """items: list of (spec, prior_text, note). Builds prompts in spec order, recording seed ids."""
        prompts = []
        for spec, prior, note in items:
            recent, overused = nudge.recent_openers(), nudge.overused_starts()
            if revise:
                user = P.build_reviser_user(spec, prior, note, seeds, rng,
                                            recent_openers=recent, overused_starts=overused)
            else:
                user = P.build_generator_user(spec, seeds, rng, recent_openers=recent,
                                              overused_starts=overused)
            seen = spec.setdefault("seed_ids", [])
            seen.extend(s for s in P.LAST_SEED_IDS if s not in seen)
            prompts.append((P.build_generator_system(spec.get("granularity", "snippet")), user))
        model, outs = run_slot(backend, gen_slot, prompts, log=log, max_tokens=GEN_MAX_TOKENS,
                               temperature=args.gen_temperature)
        if model not in gen_used:
            gen_used.append(model)
        return model, [clean_generation(o) for o in outs]

    # ---- phase 1: generate in chunks, prefilter in spec order ---------------- #
    t0 = time.monotonic()
    state = {}       # spec index -> dict(text, gen_model, ...)
    rejects = []     # (index, row) for the rejected file
    for start in range(0, len(specs), args.chunk):
        chunk = list(range(start, min(start + args.chunk, len(specs))))
        log(f"[generate] specs {chunk[0]}..{chunk[-1]}")
        model, texts = _generate([(specs[i], None, None) for i in chunk])
        for i, text in zip(chunk, texts):
            spec = specs[i]
            stats.attempt(spec)
            failed = prefilter(text, spec, accepted_texts)
            if failed is not None:
                reason, detail = failed
                stats.reject(spec, reason)
                rejects.append((i, rejected_row(spec, reason, text=text, detail=detail, gen_model=model)))
                continue
            accepted_texts.append(text)          # dedup later texts against this one
            nudge.note_opener(text)
            state[i] = {"text": text, "gen_model": model}
    timings["generate"] = time.monotonic() - t0

    # ---- phase 2: critics over every survivor, decide ------------------------ #
    t0 = time.monotonic()
    idx = sorted(state)
    verdicts, panels = critique_all(backend, critics, [state[i]["text"] for i in idx], log=log)
    to_revise = []
    for i, v, panel in zip(idx, verdicts, panels):
        agg = aggregate(v, n_critics)
        ok, dominant, secondary, note = decide(specs[i], agg, n_critics)
        state[i].update(verdicts=v, panel=panel, agg=agg, ok=ok, dominant=dominant,
                        secondary=secondary, note=note, agg_first=None if ok else agg)
        if not ok:
            to_revise.append(i)
    timings["critique"] = time.monotonic() - t0

    # ---- phase 3: revise the failures once, re-critique, decide -------------- #
    t0 = time.monotonic()
    if to_revise:
        log(f"[revise] {len(to_revise)} specs")
        try:
            model, texts = _generate([(specs[i], state[i]["text"], state[i]["note"]) for i in to_revise],
                                     revise=True)
            stats.n_revised += len(to_revise)
            verdicts, panels = critique_all(backend, critics, texts, log=log)
            for i, text, v, panel in zip(to_revise, texts, verdicts, panels):
                agg = aggregate(v, n_critics)
                ok, dominant, secondary, note = decide(specs[i], agg, n_critics)
                state[i].update(text=text, gen_model=model, verdicts=v, panel=panel, agg=agg, ok=ok,
                                dominant=dominant, secondary=secondary, note=note)
        except BackendError as e:
            log(f"[revise] skipped: {e}")
    timings["revise"] = time.monotonic() - t0

    # ---- write, in spec order ------------------------------------------------- #
    accepted = []
    for i in idx:
        s, spec = state[i], specs[i]
        if s["ok"]:
            stats.accept(s["text"], s["dominant"], s["agg"])
            accepted.append(accepted_row(i, spec, s["text"], s["dominant"], s["secondary"], s["agg"],
                                         s["verdicts"], revised=s["agg_first"] is not None,
                                         n_critics=n_critics, gen_model=s["gen_model"],
                                         critic_models=s["panel"]))
        else:
            reason = failure_reason(s["agg"], n_critics)
            stats.reject(spec, reason, s["agg"])
            rejects.append((i, rejected_row(spec, reason, text=s["text"], agg=s["agg"],
                                            agg_first=s["agg_first"], note=s["note"], revised=True,
                                            gen_model=s["gen_model"], critic_models=s["panel"])))
    with out_path.open("w", encoding="utf-8") as fh:
        for row in accepted:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with rej_path.open("w", encoding="utf-8") as fh:
        for _, row in sorted(rejects, key=lambda r: r[0]):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats.print_summary()
    accepted_by_type: dict[str, int] = {}
    for row in accepted:
        accepted_by_type[row["dominant_distortion"]] = accepted_by_type.get(row["dominant_distortion"], 0) + 1
    summary = {
        "n_requested": len(specs), **stats.to_dict(),
        "accepted_by_type": dict(sorted(accepted_by_type.items(), key=lambda kv: -kv[1])),
        "gen_models_used": gen_used, "critic_slots": [s for s, _ in critics],
        "critic_temps": [t for _, t in critics], "chunk": args.chunk, "seed": args.seed,
        "seconds_by_phase": {k: round(v, 1) for k, v in timings.items()},
        "seconds_total": round(sum(timings.values()), 1),
        "cost_usd": 0.0, "note": "local open models on the notebook GPU; no API cost",
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    log(f"\n[time] " + " | ".join(f"{k} {v:.0f}s" for k, v in timings.items()))
    log(f"[out]      {out_path}\n[rejects]  {rej_path}\n[summary]  {out_path.with_suffix('.summary.json')}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EmpowerLens synthetic generation, batched on a local GPU (vLLM).")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="data/synthetic_gen/seeds/entrepreneurial_seeds.jsonl")
    ap.add_argument("--out", default="data/synthetic/calibrate.jsonl")
    ap.add_argument("--gen-model", default=DEFAULT_GEN, help="HF id; '|' lists backups")
    ap.add_argument("--gen-temperature", type=float, default=0.9)
    ap.add_argument("--critic-models", default=DEFAULT_CRITICS,
                    help="comma-separated critic slots (one critic each); '|' lists a slot's backups")
    ap.add_argument("--critic-temps", default="0.3,0.6,0.9")
    ap.add_argument("--chunk", type=int, default=75,
                    help="specs per generator batch; opener nudges update between chunks")
    ap.add_argument("--backend", choices=["vllm", "fake"], default="vllm")
    ap.add_argument("--tp", type=int, default=2, help="tensor-parallel size (GPUs per model)")
    ap.add_argument("--max-model-len", type=int, default=6144)
    ap.add_argument("--gpu-mem", type=float, default=0.90)
    ap.add_argument("--workdir", default=None, help="where request/response files go")
    # worker mode (internal): python -m src.synth.batch_generate --worker REQ RESP --worker-model M ...
    ap.add_argument("--worker", nargs=2, metavar=("REQUESTS", "RESPONSES"), help=argparse.SUPPRESS)
    ap.add_argument("--worker-model", help=argparse.SUPPRESS)
    ap.add_argument("--max-tokens", type=int, default=GEN_MAX_TOKENS, help=argparse.SUPPRESS)
    ap.add_argument("--temperature", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if args.worker:
        worker(args)
        return
    if args.backend == "fake":
        backend = FakeBackend()
    else:
        backend = VllmBackend(tp=args.tp, max_model_len=args.max_model_len, gpu_mem=args.gpu_mem,
                              workdir=args.workdir)
    summary = run_batch(args, backend, log=lambda m: print(m, flush=True))
    if summary["accepted"] == 0:
        sys.exit("no rows accepted — read the .rejected.jsonl before a bigger run")


if __name__ == "__main__":
    main()
