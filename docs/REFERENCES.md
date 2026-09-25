# References — detection & training side

Papers behind the modelling work in `src/` and `notebooks/`, with **where each one
is actually used**. A citation with no corresponding code is not listed.

For the synthetic-data-generation side (AttrPrompt, KoACD, Self-Instruct, Diagnosis
of Thought, distribution matching) see `src/synth/PIPELINE.md`, which has its own
reference list. This file does not duplicate it.

> **Verification status.** Entries marked ✅ have been checked against the source.
> Entries marked ⚠️ are recorded from memory and the arXiv/DOI identifier should be
> confirmed before the identifier appears in a submitted document. The *claims*
> attributed to each paper are what the code relies on; check those too if you
> quote them.

---

## Task & data

**Shreevastava & Foltz (2021).** *Detecting Cognitive Distortions from
Patient-Therapist Interactions.* CLPsych workshop. ⚠️

The replication target. Source of the 10-distortion taxonomy plus No Distortion, of
`Annotated_data.csv`, and of the reported ~0.79 binary F1 this project benchmarks
against.

> ⚠️ **Two figures still need checking before citation** (also flagged in
> `docs/EXPERIMENTS.md`): which F1 variant the 0.79 refers to (weighted vs
> positive-class), and the ~33.7% inter-annotator agreement figure. The latter
> matters more than it looks — `docs/codipas_agreement.md` independently measures
> 36.8% agreement between CODIPAS's derived labels and the human ones, and if those
> two numbers are measuring the same thing that is a substantive finding rather
> than a coincidence.

**Maddela, Ung, Xu, Madotto, Foran & Boureau (2023).** *Training Models to Generate,
Recognize, and Reframe Unhelpful Thoughts.* ACL 2023. ⚠️ (arXiv:2307.02768)

PatternReframe — the 9,688 crowdsourced unhelpful thoughts used as borrowed training
data. Loader: `src/make_splits_patternreframe.py`. The 2.4 MB source tarball is
committed at `data/patternreframe/`.

Nine of its ten patterns map onto this taxonomy; "Discounting the positive" is
dropped rather than merged into `mental_filter`, and `emotional_reasoning` has zero
coverage in the source.

---

## Model backbones

**Ji, Zhang, Chen, Cambria & Tiedemann (2022).** *MentalBERT: Publicly Available
Pretrained Language Models for Mental Healthcare.* LREC 2022. ⚠️ (arXiv:2110.15621)

Source of both `mental/mental-bert-base-uncased` and `mental/mental-roberta-base`.
`mental-roberta-base` is the default in `notebooks/cascade_bootstrap.py` and the
backbone for every headline result.

**Liu et al. (2019).** *RoBERTa: A Robustly Optimized BERT Pretraining Approach.*
⚠️ (arXiv:1907.11692)

`roberta-base`, the general-domain control.

**He, Gao & Chen (2021).** *DeBERTaV3.* ⚠️ (arXiv:2111.09543)

`microsoft/deberta-v3-base`. Needs `sentencepiece`; its config ships a
`torch_dtype` hint that `train_transformer.py` explicitly overrides for fp16.

---

## Training techniques

**Phang, Févry & Bowman (2018).** *Sentence Encoders on STILTs: Supplementary
Training on Intermediate Labeled-data Tasks.* ✅ (arXiv:1811.01088)

**The basis for `notebooks/kaggle_runner_sequential.ipynb`.** Fine-tune on an
intermediate labelled task first, then on the small target task. This is the
standard move when the target set is small and a larger, noisier, related set
exists — precisely the situation here: 1,278 Annotated distorted rows against 8,712
PatternReframe ones.

Two details of the setup follow directly from this line of work:

- Stage B **keeps** stage A's classifier head rather than reinitialising it, which
  is only valid because both stages are 10-way multilabel over the same `ml_*`
  columns in the same order.
- Stage B runs at a **lower learning rate** (1e-5 vs 3e-5) so it re-aligns the head
  to the target label convention without overwriting the representations stage A
  learned. Partial forgetting is intended; total forgetting is the failure mode,
  and it shows up as stage B scoring the same as the Annotated-only baseline.

**Pruksachatkun, Phang, Liu et al. (2020).** *Intermediate-Task Transfer Learning
with Pretrained Language Models: When and Why Does It Work?* ACL 2020. ⚠️
(arXiv:2005.00628)

The follow-up that asks when STILTs actually helps. Relevant because it finds
intermediate-task transfer is **not** reliably positive — which is why the
sequential notebook treats a negative result as reportable rather than as a failed
run, and why `emotional_reasoning` (the one class PatternReframe cannot pretrain)
is used as a control.

**Sun, Qiu, Xu & Huang (2019).** *How to Fine-Tune BERT for Text Classification?*
⚠️ (arXiv:1905.05583)

Source of the **head+tail truncation** strategy (`--truncation head_tail`): keep the
first 128 tokens and the last (budget − 128), rather than truncating the tail
outright. Matters here because Annotated reflections have a median of ~129 words and
the conclusion of a reflection often carries the distortion.

**Lin, Goyal, Girshick, He & Dollár (2017).** *Focal Loss for Dense Object
Detection.* ⚠️ (arXiv:1708.02002)

`--loss focal --focal-gamma`. Down-weights easy examples so the rare distortion
classes are not drowned out. Used alongside `pos_weight` in
`src/train_transformer.py`.

---

## Evaluation & agreement

**Cohen (1960).** *A Coefficient of Agreement for Nominal Scales.* ⚠️

**Landis & Koch (1977).** *The Measurement of Observer Agreement for Categorical
Data.* Biometrics 33(1):159–174. ⚠️

Cohen's κ and the conventional interpretation bands (slight / fair / moderate /
substantial). Both used in `src/codipas_agreement.py`, which reports κ = 0.321
(binary) and κ = 0.199 (11-class) between CODIPAS's derived labels and the human
annotation.

The bands are a convention, not a statistical test, which is why that script also
bootstraps a 95% CI — "is 0.32 meaningfully different from 0.20" is a question the
bands cannot answer, and the intervals do not overlap.

---

## Not cited, deliberately

- **CODIPAS** has no accompanying paper in this repo; it arrived as `CODIPAS.json`.
  Its `y_mc` is derived by an aggregation rule rather than human-annotated, which
  `docs/codipas_agreement.md` quantifies. Do not cite it as an annotated corpus.
- The frozen `cd_pipeline.py` replication uses offline embedding substitutes
  (Word2Vec/Doc2Vec/SIF) whose original papers are not listed here, because that
  file is a submitted artifact and is not being extended.
