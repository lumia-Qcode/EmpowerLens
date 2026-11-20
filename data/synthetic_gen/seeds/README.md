# Entrepreneurial seed set

Hand-curated, small seed set of entrepreneurial self-reflections labeled with CBT
cognitive distortions. **Purpose: grounding a multi-agent synthetic data
generator** — not a training set. It anchors the entrepreneurial register and
defensible distortion labels that the generator would otherwise miss. Since S04 it is
English only and not conditioned on gender (see `seed_counts.md`). It is **built** by
`python -m src.synth.build_seeds` from `team_seeds.jsonl` (team-authored) and the S03
seed pool (`data/supplementary/supplementary_labelled.csv`, role `seed`); edit those, not
`entrepreneurial_seeds.jsonl`.

These seeds are **not** part of the frozen `Annotated_data.csv` benchmark and are
never mixed into `data/splits/`.

## Record schema (`entrepreneurial_seeds.jsonl`, one JSON per line)

| field | notes |
|---|---|
| `id` | `seed_NNN` |
| `text` | the reflection (team seeds 1–3 sentences; scraped seeds whole if ≤ 120 words, else a verbatim excerpt) |
| `distorted_part` | span carrying the distortion, or `null` for No Distortion |
| `dominant_distortion` | **exact raw label string** from `LABEL_CANON` in `src/data.py` (e.g. `"Mind Reading"`, `"All-or-nothing thinking"`, `"No Distortion"`) |
| `secondary_distortion` | optional second label or `null` |
| `trigger` | situational axis (e.g. `investor_rejection`, `family_disapproval`, `cofounder_conflict`) |
| `register` | `english_formal` \| `english_casual` (code-mix seeds removed in S04) |
| `cultural_marker` | e.g. `family_pressure`, or `null` (`gender_expectation` removed in S04) |
| `source` | provenance — `authored_from_<X>_pattern` (public-post pattern, re-authored) or `authored_from_pakistani_women_entrepreneur_voice` or primary-collected |

`dominant_distortion` / `secondary_distortion` use the raw strings so they map
straight through `_canon` → `y_bin` / `y_mc` / `y_ml` without new mappings.

Added in S04: `seed_source` (`supplementary` | `team-authored`, on every seed), `item_id`
(scraped seeds: the S03 row), `excerpt` (scraped seed cut to its span ± 1 sentence),
`calibrated_pessimism`, `team_id` (team seeds: id in `team_seeds.jsonl`).

## Coverage (expand toward ~2–3 per distortion × spread of triggers)

- **Distortions:** all 10 + several `No Distortion` (calibrated realistic
  pessimism — feeds novelty direction D1: distortion vs. calibrated pessimism).
- **Triggers:** investor/loan rejection, cofounder conflict, low sales, family
  disapproval, imposter feelings, competitor comparison, customer criticism.
- **Register:** English formal/casual (English only since S04).
- **Cultural markers:** family pressure, none (no gender markers since S04).
- **Current counts:** 4 per distortion type (Mind Reading 5) + 3 No Distortion; see `seed_counts.md`.

## Provenance & ethics

- **Public founder posts (Failory, Starter Story, Indie Hackers, r/Entrepreneur,
  r/smallbusiness):** used as *patterns only* — read the distortion pattern, then
  re-author a short reflection. No verbatim reproduction (copyright).
- **Pakistani women-entrepreneur voice:** authored in-register; if any real
  primary responses are added (e.g. a short consent-based Google Form to incubator
  participants), they must be anonymized, consent-based, stored here, and never
  presented diagnostically.

## How the generator uses these

1. **Few-shot exemplars** — 3–5 seeds injected into the generator agent's prompt as voice anchors.
2. **Distribution spec** — the coverage table becomes the planner agent's stratification target.
3. **Critic reference** — the critic/validator agent scores synthetic candidates against seed voice and checks each label is defensible (the generator→critic loop in the FYP plan).
