"""Synthetic-data generation orchestrator (sync path — pilot).

Cross-provider by design: the **generator** (author) and the **critic panel**
(judge) are different providers, so acceptance isn't one model family grading
itself. Default: Claude Sonnet generates; a panel of blind Gemini critics
(decorrelated by temperature) independently diagnoses each row.

Cycle per row:
    planner spec -> generator (Claude) -> format/English/dedup filters
      -> N BLIND critics (Gemini, Diagnosis-of-Thought, varied temperature)
      -> aggregate agreement -> accept / revise-once / reject
      -> write JSONL row (hard primary+secondary capped at 2, plus 10-dim y_soft)

Run
---
    # cheap sanity check, no API calls, no spend:
    venv\\Scripts\\python.exe -m src.synth.generate --n 300 --dry-run

    # real pilot (needs ANTHROPIC_API_KEY for the generator and
    #             GEMINI_API_KEY for the critics):
    venv\\Scripts\\python.exe -m src.synth.generate --n 200 \
        --out data/synthetic/pilot_en.jsonl

Output rows match the seed/Annotated_data schema (raw label strings via
src/data.py's LABEL_CANON) plus y_soft. Training data only — never the test set.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

from . import attributes as A
from . import prompts as P
from . import planner as PL
from . import dedup as D
from . import llm


def load_seeds(path: str, english_only: bool = True) -> list[dict]:
    seeds, p = [], Path(path)
    if not p.exists():
        return seeds
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if english_only and row.get("register") == "urdu_english_codemix":
            continue
        if row.get("text"):
            seeds.append(row)
    return seeds


def _word_count(text: str) -> int:
    return len(text.split())


def _opener(text: str, n: int = 4) -> str:
    """First n words — the unit the banned-opener list works on."""
    return " ".join(text.split()[:n]).strip(' ,.!?"\'').lower()


def _contains_banned_term(text: str) -> str | None:
    """Return the first banned clinical/label term found, or None.

    Belt and braces: the prompt forbids these, but a generation that names its
    own distortion would teach the classifier the label word instead of the
    thinking pattern, so it is cheaper to drop the row than to trust the prompt.
    """
    low = text.lower()
    for term in A.BANNED_TERMS:
        if term in low:
            return term
    return None


def _english_dominant(text: str) -> bool:
    if not text:
        return False
    return sum(1 for c in text if ord(c) < 128) / len(text) >= 0.90


# --------------------------------------------------------------------------- #
# Aggregation                                                                  #
# --------------------------------------------------------------------------- #

def span_agreement(spans: list[str]) -> float:
    """Mean pairwise token-overlap of the spans the critics quoted.

    Critics can agree on the label while pointing at different sentences — that
    marks a diffuse row the vote count alone cannot show. 1.0 = identical spans,
    0.0 = disjoint. Returns 0.0 when fewer than two spans came back.
    """
    real = [s for s in spans if s]
    if len(real) < 2:
        return 0.0
    pairs = [D.jaccard(a, b) for i, a in enumerate(real) for b in real[i + 1:]]
    return round(sum(pairs) / len(pairs), 3) if pairs else 0.0


def aggregate(verdicts: list[dict], n_critics: int) -> dict:
    soft = {lab: 0.0 for lab in A.DISTORTION_LABELS}
    votes = {lab: 0 for lab in A.DISTORTION_LABELS}
    no_dist_votes = 0
    unknown_votes = 0
    spans: list[str] = []
    for v in verdicts:
        present = set(v.get("present", []))
        # Unknown is checked first and is exclusive: "I cannot judge this" is a
        # different verdict from "I judged this and it is sound", and counting a
        # row as both would inflate the No-Distortion class with unjudgeable text.
        if A.UNKNOWN in present or v.get("primary") == A.UNKNOWN:
            unknown_votes += 1
            continue
        if A.NO_DISTORTION in present or v.get("primary") == A.NO_DISTORTION:
            no_dist_votes += 1
        for lab in A.DISTORTION_LABELS:
            if lab in present:
                votes[lab] += 1
        sp = v.get("distorted_span")
        if sp:
            spans.append(sp)
    for lab in A.DISTORTION_LABELS:
        soft[lab] = round(votes[lab] / n_critics, 3) if n_critics else 0.0
    # Longest span wins as the representative — it carries the most context and
    # maps onto the `Distorted part` column in Annotated_data.csv.
    return {"soft": soft, "votes": votes, "no_distortion_votes": no_dist_votes,
            "unknown_votes": unknown_votes,
            "spans": spans,
            "span": max(spans, key=len) if spans else None,
            "span_agreement": span_agreement(spans)}


def decide(spec, agg, n_critics):
    majority = math.ceil(n_critics / 2)
    primary = spec["primary"]

    # Checked first, and never counted as a taxonomy signal: a majority Unknown
    # means the generator produced something unjudgeable (a bare vent, a plain
    # situation report), which is a generation failure, not evidence that the
    # clinical label fails to fit entrepreneurial reasoning.
    if agg.get("unknown_votes", 0) >= majority:
        return False, None, None, "critics could not judge the text (Unknown)"

    if primary == A.NO_DISTORTION:
        if agg["no_distortion_votes"] >= majority and max(agg["votes"].values(), default=0) < majority:
            return True, A.NO_DISTORTION, None, "ok"
        top = max(agg["votes"], key=agg["votes"].get) if any(agg["votes"].values()) else "?"
        return False, None, None, f"critics read a distortion ({top}) rather than No Distortion"

    if agg["votes"].get(primary, 0) >= majority:
        ranked = [l for l in sorted(A.DISTORTION_LABELS, key=lambda l: -agg["soft"][l]) if agg["soft"][l] > 0]
        ranked = [primary] + [l for l in ranked if l != primary]
        secondary = ranked[1] if len(ranked) > 1 else None
        return True, primary, secondary, "ok"

    top = max(agg["votes"], key=agg["votes"].get) if any(agg["votes"].values()) else A.NO_DISTORTION
    return False, None, None, f"target '{primary}' not agreed; critics leaned '{top}'"


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def run(args) -> None:
    seeds = load_seeds(args.seeds, english_only=True)
    specs = PL.build_specs(args.n, seed=args.seed)
    rng = random.Random(args.seed)

    temps = [float(t) for t in args.critic_temps.split(",") if t.strip()]
    critic_temps = [temps[i % len(temps)] for i in range(args.n_critics)]

    print(f"[planner] {len(specs)} specs | primary distribution:")
    for lab, c in PL.distribution(specs).items():
        print(f"    {c:>4}  {lab}")
    gran_counts: dict[str, int] = {}
    for s in specs:
        gran_counts[s["granularity"]] = gran_counts.get(s["granularity"], 0) + 1
    print(f"[planner] granularity: {gran_counts}")
    print(f"[planner] distinct (distortion x trigger) cells covered: "
          f"{len({(s['primary'], s['trigger']) for s in specs})}")
    print(f"[planner] English seeds loaded: {len(seeds)}")
    print(f"[panel] generator = {args.gen_provider}:{args.gen_model}")
    print(f"[panel] critics   = {args.n_critics} x {args.critic_provider}:{args.critic_model} "
          f"@ temps {critic_temps}")

    if args.dry_run:
        print("\n[dry-run] sample generator user prompt for spec[0]:\n")
        print(P.build_generator_user(specs[0], seeds, rng))
        print("\n[dry-run] no API calls made, nothing written.")
        return

    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")) \
            and args.gen_provider == "anthropic":
        sys.exit("Set ANTHROPIC_API_KEY for the generator (or use --dry-run).")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Rejects are written alongside the accepted rows. An abandoned spec is NOT
    # noise: it is a case where we asked for distortion X in an entrepreneurial
    # setting and independent critics twice declined to see X. The pattern of
    # which intended labels abandon most — and what the panel read instead — is
    # direct quantitative evidence that the clinical taxonomy does not map
    # cleanly onto founder thinking, which is the entrepreneurial-taxonomy
    # novelty argument. Discarding it throws away the strongest figure.
    rej_path = out_path.with_suffix(".rejected.jsonl")

    accepted_texts = [s["text"] for s in seeds]
    used_openers: list[str] = []          # 4-word openers, for the ban display
    opener_starts: dict[str, int] = {}    # opening bigram -> count
    OVERUSED_AT = 3                       # ban a bigram once it's clearly a default
    cost = 0.0
    n_accepted = n_rejected = n_revised = 0
    n_banned_term = n_length = n_unknown = 0
    abandoned_by_label: dict[str, int] = {}
    attempted_by_label: dict[str, int] = {}
    confusion: dict[tuple[str, str], int] = {}   # (intended, what critics leaned)
    # Split out the abandonments that actually reached the panel — a spec
    # dropped by the length or dedup filter says nothing about the taxonomy.
    critic_rejected_by_label: dict[str, int] = {}
    unknown_by_label: dict[str, int] = {}
    # Calibrated pessimism is the central domain risk: "I probably won't close
    # this round in this market" is realistic, not Fortune-telling. If the panel
    # cannot reliably pass intended No-Distortion rows, that is itself a result
    # about how hard the distinction is — and it means the No-Distortion class
    # will come out under-populated and skewed. Track the margin on the ones
    # that do pass, not just the pass rate.
    nd_accept_margin: list[int] = []

    def _gen(spec, prior=None, note=None):
        nonlocal cost
        # Only the last 12 openers are injected: the list is a diversity nudge,
        # not an exhaustive constraint, and an ever-growing block would bloat
        # every prompt and eventually crowd out the seed exemplars.
        recent = used_openers[-12:]
        # A 4-word ban is weaker than it looks — same tic, different 4-gram. So
        # also surface the short starts that have already become habitual.
        overused = [s for s, c in sorted(opener_starts.items(), key=lambda kv: -kv[1])
                    if c >= OVERUSED_AT]
        if prior is None:
            user = P.build_generator_user(spec, seeds, rng, recent_openers=recent,
                                          overused_starts=overused)
        else:
            user = P.build_reviser_user(spec, prior, note, seeds, rng,
                                        recent_openers=recent, overused_starts=overused)
        # 800, not 400: on Sonnet 5 adaptive thinking is on by default and its
        # tokens count against max_tokens, so a 400 cap left ~150 tokens for
        # thinking and truncated the reflection mid-sentence (which then failed
        # the word-count filter below). Billing is on tokens produced, not the
        # cap, so the headroom is free.
        system = P.build_generator_system(spec.get("granularity", "snippet"))
        text, u = llm.complete_text(args.gen_provider, args.gen_model, system, user, 800)
        cost += llm.estimate_cost(args.gen_provider, args.gen_model, u)
        return text

    def _critique(text):
        nonlocal cost
        verdicts = []
        for t in critic_temps:
            v, u = llm.complete_json(args.critic_provider, args.critic_model,
                                     P.SYSTEM_CRITIC, P.build_critic_user(text),
                                     1100, temperature=t, json_schema=P.CRITIC_SCHEMA)
            cost += llm.estimate_cost(args.critic_provider, args.critic_model, u)
            if v is None:
                continue
            # The prompt says: cannot find two supporting sentences -> Unknown.
            # Enforce it rather than trust it. A label asserted without the
            # evidence it was told to produce is exactly the speculative verdict
            # the evidence rule exists to prevent.
            ev = v.get("evidence") or []
            if len(ev) < 2 and v.get("primary") not in (A.UNKNOWN,):
                v["primary"] = A.UNKNOWN
                v["present"] = [A.UNKNOWN]
                v["coerced_unknown"] = True
            verdicts.append(v)
        return verdicts

    with out_path.open("w", encoding="utf-8") as fh, \
            rej_path.open("w", encoding="utf-8") as rfh:

        def _leaned(agg) -> str:
            """The label the panel actually gravitated to, when the target failed."""
            if agg is None or not any(agg["votes"].values()):
                return A.NO_DISTORTION
            return max(agg["votes"], key=agg["votes"].get)

        def _reject(spec, reason, text=None, agg=None, agg_first=None,
                    note=None, revised=False, detail=None):
            nonlocal n_rejected
            n_rejected += 1
            lab = spec["primary"]
            abandoned_by_label[lab] = abandoned_by_label.get(lab, 0) + 1

            row = {
                "spec_id": spec["spec_id"],
                "reason": reason,
                "detail": detail,
                "intended_primary": lab,
                "intended_secondary": spec["secondary"],
                "granularity": spec.get("granularity", "snippet"),
                "trigger": spec["trigger"],
                "register": spec["register"],
                "stage": spec["stage"],
                "sector": spec["sector"],
                "cultural_marker": spec["cultural_marker"],
                "revised": revised,
                "text": text,
                "critic_note": note,
            }
            # Only critic rejections carry a verdict — filter rejections never
            # reached the panel.
            if agg is not None:
                row.update({
                    "critic_votes": agg["votes"],
                    "y_soft": agg["soft"],
                    "no_distortion_votes": agg["no_distortion_votes"],
                    "unknown_votes": agg["unknown_votes"],
                    "span_agreement": agg["span_agreement"],
                })
                # Only a genuine label disagreement is taxonomy evidence. An
                # Unknown majority means the text was unjudgeable, which says
                # nothing about whether the clinical label fits founder thinking
                # — counting it would corrupt the figure.
                if reason == "critics_disagreed":
                    leaned = _leaned(agg)
                    confusion[(lab, leaned)] = confusion.get((lab, leaned), 0) + 1
                    critic_rejected_by_label[lab] = \
                        critic_rejected_by_label.get(lab, 0) + 1
                    row["critics_leaned"] = leaned
                elif reason == "critics_unknown":
                    unknown_by_label[lab] = unknown_by_label.get(lab, 0) + 1
            if agg_first is not None:
                row["first_attempt_votes"] = agg_first["votes"]
                row["first_attempt_leaned"] = _leaned(agg_first)
            rfh.write(json.dumps(row, ensure_ascii=False) + "\n")

        for i, spec in enumerate(specs):
            attempted_by_label[spec["primary"]] = \
                attempted_by_label.get(spec["primary"], 0) + 1
            try:
                text = _gen(spec)
            except Exception as e:  # noqa: BLE001
                print(f"[{spec['spec_id']}] generator error: {e}")
                _reject(spec, "generator_error", detail=str(e))
                continue

            # Length bounds now come from the spec's granularity, not one global
            # range — a 30-50 word snippet and a 150-word reflection cannot
            # share a filter.
            g = A.GRANULARITIES[spec.get("granularity", "snippet")]
            wc = _word_count(text)
            if not (g["min_words"] <= wc <= g["max_words"]):
                n_length += 1
                _reject(spec, "length", text=text, detail=f"{wc} words")
                continue

            banned = _contains_banned_term(text)
            if banned is not None:
                print(f"[{spec['spec_id']}] dropped: named its own pattern ({banned!r})")
                n_banned_term += 1
                _reject(spec, "named_own_pattern", text=text, detail=banned)
                continue

            if not _english_dominant(text):
                _reject(spec, "not_english", text=text)
                continue
            if D.is_duplicate(text, accepted_texts):
                _reject(spec, "duplicate", text=text)
                continue

            verdicts = _critique(text)
            agg = aggregate(verdicts, args.n_critics)
            ok, dominant, secondary, note = decide(spec, agg, args.n_critics)
            agg_first = agg if not ok else None      # keep round 1 for the log
            first_text = text

            if not ok:
                try:
                    text = _gen(spec, prior=text, note=note)
                    n_revised += 1
                    verdicts = _critique(text)
                    agg = aggregate(verdicts, args.n_critics)
                    ok, dominant, secondary, note = decide(spec, agg, args.n_critics)
                except Exception as e:  # noqa: BLE001
                    print(f"[{spec['spec_id']}] reviser error: {e}")
                    text = first_text

            if not ok:
                # Two different failures, kept apart. "critics_unknown" = the
                # text was unjudgeable (generator problem). "critics_disagreed" =
                # asked for X, two independent rounds of blind critics read
                # something else (taxonomy signal).
                majority = math.ceil(args.n_critics / 2)
                reason = ("critics_unknown"
                          if agg.get("unknown_votes", 0) >= majority
                          else "critics_disagreed")
                if reason == "critics_unknown":
                    n_unknown += 1
                _reject(spec, reason, text=text, agg=agg,
                        agg_first=agg_first, note=note, revised=True)
                continue

            fh.write(json.dumps({
                "id": f"syn_{i:05d}",
                "text": text,
                "dominant_distortion": dominant,
                "secondary_distortion": secondary,
                # Maps onto the `Distorted part` column in Annotated_data.csv.
                "distorted_part": agg["span"],
                "span_agreement": agg["span_agreement"],
                # Sentence-level evidence and the causal chain from the critic
                # whose verdict matched. Free span-level explainability material
                # to sit alongside SHAP/Captum later.
                "evidence": next((v.get("evidence") for v in verdicts
                                  if v.get("primary") == dominant), None),
                "critic_reasoning": next((v.get("reasoning") for v in verdicts
                                          if v.get("primary") == dominant), None),
                "y_soft": agg["soft"],
                "critic_votes": agg["votes"],
                # How decisively the panel called it No Distortion — the
                # calibrated-pessimism margin. 3/3 is clean; 2/3 is contested.
                "no_distortion_votes": agg["no_distortion_votes"],
                "n_critics": args.n_critics,
                "granularity": spec.get("granularity", "snippet"),
                # True if this row is a revision. Revised rows are generated
                # while being told what the critics objected to, so they risk
                # overcorrecting into caricature — keep the flag so accepted
                # revised rows can be compared against first-pass ones.
                "revised": agg_first is not None,
                "trigger": spec["trigger"],
                "register": spec["register"],
                "stage": spec["stage"],
                "sector": spec["sector"],
                "cultural_marker": spec["cultural_marker"],
                "intended_primary": spec["primary"],
                "intended_secondary": spec["secondary"],
                "source": "synthetic",
            }, ensure_ascii=False) + "\n")
            accepted_texts.append(text)
            used_openers.append(_opener(text))
            bg = _opener(text, 2)
            opener_starts[bg] = opener_starts.get(bg, 0) + 1
            if dominant == A.NO_DISTORTION:
                nd_accept_margin.append(agg["no_distortion_votes"])
            n_accepted += 1

            if (i + 1) % 25 == 0:
                print(f"  ...{i + 1}/{len(specs)} | accepted {n_accepted} | est ${cost:.2f}")

    print(f"\n[done] accepted {n_accepted} | rejected {n_rejected} | revised {n_revised}")
    print(f"[filters] length {n_length} | named-own-pattern {n_banned_term}")
    # Unknown is a generation-quality counter, not a taxonomy one. A high number
    # means the generator is producing bare vents or plain situation reports with
    # no reasoning in them — fix the prompt, not the taxonomy.
    print(f"[unjudgeable] critics returned Unknown on {n_unknown} specs")
    # distinct-n at three widths. The 4-word figure flatters: "I have always
    # felt" and "I have never felt" are distinct 4-grams but the same tic, so
    # watch the 1- and 2-word columns for real opener collapse.
    if used_openers:
        d1 = len({o.split()[0] for o in used_openers if o.split()})
        d2 = len(opener_starts)
        d4 = len(set(used_openers))
        n = len(used_openers)
        print(f"[diversity] distinct openers among {n} accepted rows: "
              f"1-word {d1}/{n} | 2-word {d2}/{n} | 4-word {d4}/{n}")
        top = sorted(opener_starts.items(), key=lambda kv: -kv[1])[:3]
        if top and top[0][1] > 1:
            print("            most repeated starts: "
                  + " | ".join(f'"{s}" x{c}' for s, c in top if c > 1))

    # Abandonment rate per intended label. A label that abandons far above the
    # mean is one the critics could not find in entrepreneurial text — evidence
    # for taxonomy inadequacy, not a generation bug.
    if abandoned_by_label:
        print("\n[abandonment] intended label -> abandoned / attempted")
        rows = sorted(
            ((lab, abandoned_by_label.get(lab, 0), n) for lab, n in attempted_by_label.items()),
            key=lambda r: -(r[1] / r[2]) if r[2] else 0,
        )
        for lab, bad, tot in rows:
            pct = 100 * bad / tot if tot else 0.0
            print(f"    {pct:5.1f}%  {bad:>3}/{tot:<3}  {lab}")

    if confusion:
        print("\n[taxonomy] asked for X, panel read Y (critic rejections only)")
        for (intended, leaned), c in sorted(confusion.items(), key=lambda kv: -kv[1])[:15]:
            print(f"    {c:>3}x  {intended}  ->  {leaned}")

    # The calibrated-pessimism readout, called out separately because it is the
    # central domain risk rather than one row in the table above.
    nd_tot = attempted_by_label.get(A.NO_DISTORTION, 0)
    if nd_tot:
        nd_bad = abandoned_by_label.get(A.NO_DISTORTION, 0)
        # Only the critic rejections are evidence about the distinction — a row
        # dropped by the length or dedup filter never reached the panel.
        nd_critic = critic_rejected_by_label.get(A.NO_DISTORTION, 0)
        nd_unknown = unknown_by_label.get(A.NO_DISTORTION, 0)
        nd_filtered = nd_bad - nd_critic - nd_unknown
        print(f"\n[calibrated-pessimism] No-Distortion specs: {nd_tot} attempted, "
              f"{nd_bad} abandoned ({100 * nd_bad / nd_tot:.1f}%)")
        print(f"    {nd_critic} read as distorted by the panel  <- the number that matters")
        print(f"    {nd_unknown} judged unjudgeable (Unknown)   <- generator problem")
        print(f"    {nd_filtered} dropped before reaching the panel (length / dedup / etc.)")
        misread = {lean: c for (intended, lean), c in confusion.items()
                   if intended == A.NO_DISTORTION}
        if misread:
            parts = " | ".join(f"{lab} {c}x" for lab, c in
                               sorted(misread.items(), key=lambda kv: -kv[1]))
            print(f"    critics read a distortion instead: {parts}")
        if nd_accept_margin:
            clean = sum(1 for m in nd_accept_margin if m == args.n_critics)
            print(f"    accepted margin: {clean}/{len(nd_accept_margin)} rows had a "
                  f"unanimous {args.n_critics}/{args.n_critics} No-Distortion vote")

    print(f"\n[cost] est ${cost:.2f} (sync; ignores Anthropic cache discount and Gemini free tier)")
    print(f"[out]      {out_path}")
    print(f"[rejects]  {rej_path}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="EmpowerLens synthetic data generator (English track, cross-provider).")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", default="data/seeds/entrepreneurial_seeds.jsonl")
    ap.add_argument("--out", default="data/synthetic/synthetic_train_en.jsonl")
    ap.add_argument("--gen-provider", default="anthropic", choices=["anthropic", "gemini"])
    ap.add_argument("--gen-model", default="claude-sonnet-5")
    ap.add_argument("--critic-provider", default="gemini", choices=["anthropic", "gemini"])
    ap.add_argument("--critic-model", default="gemini-2.5-flash")
    ap.add_argument("--n-critics", type=int, default=3)
    ap.add_argument("--critic-temps", default="0.3,0.6,0.9",
                    help="comma-separated temperatures cycled across critics (decorrelation)")
    ap.add_argument("--dry-run", action="store_true")
    return ap


if __name__ == "__main__":
    run(build_parser().parse_args())
