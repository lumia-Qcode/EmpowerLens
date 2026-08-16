# Real-reflection labeling guide (gold test set)

This guide is for the **human annotators** (you + the BNU Psychology collaborator)
who assign cognitive-distortion labels to the **real** reflections collected from
the Google Form. These labels become the **gold test set** — the measuring stick
for the whole domain-shift experiment — so consistency matters more than speed.

**Never label synthetic data with this sheet.** Synthetic rows are already
labelled by construction; this is only for real, human-written reflections.

## What to produce

Each annotator labels **independently** (do not discuss while labelling), so we
can compute **inter-annotator agreement (IAA)**. Fill one row per reflection in
`real_reflections_labeling_sheet.csv` (keep your own copy per annotator).

## The label set (use these exact strings)

Match `src/data.py` → `LABEL_CANON`. Pick the **Dominant** distortion; add a
**Secondary** only if a second pattern is clearly present.

| Label | Present when the writer… |
|---|---|
| **Mind Reading** | assumes what others think (usually negatively) with no real evidence |
| **Fortune-telling** | predicts the future negatively and treats it as certain |
| **Magnification** | blows consequences out of proportion (catastrophises) or shrinks positives |
| **Labeling** | attaches a fixed global label to self/others from one event ("I'm a failure") |
| **Personalization** | over-blames self for an outcome with other causes ("it's all my fault") |
| **Overgeneralization** | treats one event as a never-ending pattern ("this always happens") |
| **Emotional Reasoning** | takes a feeling as proof of fact ("I feel like a failure, so I am") |
| **Mental filter** | fixates on one negative, ignoring the larger positive picture |
| **Should statements** | rigid rules about how self/others must behave ("a founder should never rest") |
| **All-or-nothing thinking** | sees things in absolute black-and-white terms |
| **No Distortion** | balanced, evidence-based reaction. **Realistic concern / a calm plan is NOT a distortion.** |

## Decision rules (keep everyone consistent)

1. **Diagnosis-of-Thought first.** Separate the *facts* of the situation from the
   writer's *thoughts* about them, then ask: does the evidence in the text
   actually justify the thought? If yes → lean **No Distortion**.
2. **Dominant = the reasoning that drives the reflection**, not just a stray
   phrase. Secondary only if a second pattern is clearly, separately present.
3. **At most 2 labels** (1 dominant + 1 secondary). No third label.
4. **No Distortion takes no secondary.**
5. **Don't force a distortion.** A calibrated, realistic response is a valid — and
   valuable — `No Distortion` label (it also feeds the D1 direction).
6. **Distorted part (optional):** copy the shortest span that carries the dominant
   distortion, if one is identifiable. Leave blank for No Distortion.
7. **Confidence (1–5):** how sure you are of the dominant label. Low-confidence
   rows are exactly where annotators are expected to disagree — that's signal, not
   error (it mirrors the ~33.7% agreement ceiling of the benchmark).

## After labelling

1. Merge annotators' sheets by `id`.
2. Compute IAA (e.g. Cohen's/Fleiss' kappa on the dominant label).
3. Resolve disagreements by discussion (or a third adjudicator) into a final
   `dominant`/`secondary`, and **keep the per-annotator labels** — the spread is
   useful for reporting and for soft-label calibration.
4. Export the resolved rows into the `Annotated_data.csv` column shape
   (`Patient Question`, `Dominant Distortion`, `Secondary Distortion (Optional)`)
   so they flow through `src/data.py` unchanged.
