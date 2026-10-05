# EmpowerLens: Cognitive Distortion Detection in Entrepreneurial Self-Reflection

**Project Proposal**

**Project Advisor:**
[Advisor Name]

**Group Members:**

| Name | Roll No. |
|---|---|
| [Member Name] | [00L-0000] |
| [Member Name] | [00L-0000] |
| [Member Name] | [00L-0000] |

National University of Computer and Emerging Sciences
Department of Computer Science
[Campus], Pakistan

---

## Abstract

Cognitive distortions are recurring patterns of biased reasoning that shape how
setbacks are interpreted, how individuals evaluate their abilities, and how they
make decisions. In entrepreneurship, such patterns can contribute to reduced
self-efficacy, fear of failure, and difficulty translating intentions into action,
particularly among female entrepreneurs. Existing research has focused almost
entirely on clinical and mental-health settings, with no validation in
entrepreneurial contexts. It also offers only two verdicts, distorted or not, and
so cannot express a third and very common state: that a person is worried and has
good reason to be. A founder who reports that twelve investors declined and
concludes the round is unlikely to close is reasoning accurately, yet current
models flag her. Specificity on such cases is consequently unreported, and a
system that misclassifies a legitimate business concern is worse than no system at
all, because it fails precisely when the user is reasoning well.

This study develops a conservative framework for detecting cognitive distortions in
female entrepreneurs' self-reflections that models this state, termed warranted
concern, as an explicit class. Preliminary experiments motivate the design:
restricting the classifier to the annotator-marked distorted sentence raises
multi-label macro-F1 by 0.080, more than twice the 0.035 gained by replacing a
bag-of-words model with a domain-pretrained transformer, so what text the model
reads matters more than which model reads it. We therefore propose a joint model
over a shared transformer encoder with four outputs: binary detection,
sentence-level localisation of the distorted region, ten-way multi-label typing,
and warranted concern. Models will be fine-tuned on an available benchmark corpus
and adapted to the entrepreneurial domain, with reflections elicited through
scenario-based and recall-based prompts from women entrepreneurs in Pakistani
incubation programmes held out as the gold domain test set at its natural class
distribution. Because reported annotator agreement on distortion type in the
benchmark is low, a targeted subset will be independently re-annotated by two
annotators, separating the contribution of label noise from that of model capacity.
Evaluation reports specificity and false-positive rate on a purpose-built challenge
set of negative-but-justified statements first, followed by binary and per-class
multi-label F1 with support counts, and span-selection quality. The resulting
framework is expected to provide a more reliable and domain-adapted approach to
cognitive-distortion detection while minimising harmful misclassification of
legitimate entrepreneurial concerns. It may further support Pakistani incubation
programmes in identifying cohort-level cognitive patterns and informing targeted
soft-skill development, without serving as a diagnostic or therapeutic tool.

## 1. Introduction

Cognitive behavioural therapy identifies a set of recurring reasoning errors, termed
cognitive distortions, that shape how a person interprets events [1], [2]. Ten
categories are in common use: all-or-nothing thinking, overgeneralisation, mental
filter, mind reading, fortune-telling, magnification, emotional reasoning, should
statements, personalisation, and labelling. Each is defined by the *form* of the
reasoning rather than its subject matter. For example:

- *"I have never been able to hold a room and I never will."* — overgeneralisation
  from one event, combined with fortune-telling about the future.
- *"The investor checked his phone; he thinks the idea is worthless."* — mind
  reading, an inference about another person's thoughts presented as fact.
- *"I am a failure."* — labelling, in which a single outcome becomes an identity.

Automatic detection of these patterns has applications in digital mental health,
guided self-reflection, and coaching. The task is usually posed as two questions
about a piece of text: whether a distortion is present at all, and which of the ten
types it is.

Research in this area is recent. Shreevastava and Foltz [3] released the first
English dataset drawn from patient–therapist interactions, comprising 2,530
annotated utterances, and reported approximately 0.79 F1 on the binary question but
only around 0.30 on the fine-grained one. That gap is the central difficulty of the
field: deciding *whether* distorted reasoning is present is tractable, while deciding
*which kind* is not. Reported inter-annotator agreement on the type label is
approximately one third, which places a ceiling on any supervised model trained
against it. Subsequent work has extended the task to multi-label settings, to other
languages, and to zero-shot prompting with large language models, but has remained
within the clinical domain.

**Why entrepreneurs.** Early-stage founders operate in an environment dense with
rejection: investors decline, customers churn, co-founders leave, and launches fail.
Interpreting these events is unavoidable and consequential — a founder who concludes
from one rejection that she is fundamentally unsuited to the work may abandon a
viable venture, while one who dismisses a consistent pattern of negative signals may
persist with an unviable one. Incubators and accelerators run structured cohorts in
which participants are routinely asked to reflect in writing, yet trainers have no
systematic visibility into how those participants are interpreting their setbacks.

**Why this population.** Women entrepreneurs in Pakistani incubation programmes — the
National Incubation Centres, SMEDA, and Kamyab Jawan among them — are an underserved
group in both the entrepreneurship and NLP literatures. Their written reflections
also differ in register from clinical transcripts: shorter, less disclosive, framed
around business events rather than personal history. A model trained on clinical text
should not be assumed to transfer, and measuring that gap is one of this project's
contributions.

**Two gaps this project addresses.** The first is *domain shift*. Every available
annotated resource is clinical, and no work quantifies how far performance falls when
such a model is applied to entrepreneurial self-reflection.

The second is *specificity*, and it is a safety problem rather than an accuracy one.
Existing systems offer two verdicts: distorted, or not. They have no way to express a
third and very common state — that a person is worried and has good reason to be.
Consider *"Twelve investors passed, so I do not expect this round to close."* The
statement is negative, it predicts a poor outcome, and on surface features it
resembles fortune-telling. It is also correct. A system that tells this founder her
arithmetic is a cognitive distortion is worse than no system at all, because it
undermines trust precisely when the user is reasoning well. We refer to this state as
**warranted concern**, and model it explicitly.

**Framing.** EmpowerLens is explicitly non-diagnostic. It is not a clinical
instrument, not a mental-health screening tool, and not an evaluator of business
viability. It reports patterns in language; it does not assess people.

## 2. Goals and Objectives

The goal of this project is to build and evaluate a cognitive distortion detection
system for entrepreneurial self-reflection that locates the distorted text, separates
distortion from warranted concern, and is assessed on specificity rather than
accuracy alone. Our main objectives are:

- Constructing an annotated corpus of written self-reflections from women
  entrepreneurs in Pakistani incubators, labelled for cognitive distortion type and
  for warranted concern, with inter-annotator agreement reported.
- Quantifying the label noise in the existing benchmark by re-annotating a targeted
  subset with two independent annotators and reporting Cohen's κ [4], [5].
- Designing and training a joint model on a shared transformer encoder with four
  outputs: binary detection, sentence-level selection of the distorted region,
  ten-way multi-label typing, and warranted concern.
- Constructing a challenge set of negative-but-justified statements and reporting
  the false-positive rate on it as a primary result.
- Measuring and reporting the performance difference between clinical and
  entrepreneurial text.

## 3. Scope of the Project

The project consists of three phases.

**Phase 1 — Data.** We will collect written self-reflections from women entrepreneurs
through a structured online form distributed via incubation programmes, asking
participants to describe a recent setback and what was going through their mind at
the time. Two annotators — one of them a psychology student — will independently
label every response for distortion type and for warranted concern using a codebook
written in advance, and agreement will be reported. In parallel, we will re-annotate
a targeted subset of the existing benchmark, selected by identifying rows where a
trained model confidently disagrees with the recorded label, together with a random
control sample. We will additionally author a challenge set of statements that are
negative but evidence-based.

**Phase 2 — Model.** We will train a joint model on a shared transformer encoder.
Sentence-level selection identifies the region carrying the distorted reasoning; the
typing heads read a representation weighted toward that region rather than the whole
document. A binary head decides whether any distortion is present, and a
warranted-concern head can suppress the output when the negativity is justified.
Single-head variants will be trained as ablations so that the contribution of each
component is measurable.

**Phase 3 — Evaluation.** Results will be reported on three test sets kept separate:
the clinical benchmark, the collected entrepreneurial responses at their natural
class distribution, and the challenge set. Specificity and false-positive rate on the
challenge set are reported first, followed by binary F1, per-class multi-label F1
with support counts, and span selection quality.

**Out of scope.** The following are explicitly excluded: Urdu-language processing,
which is left to future work; automated reframing or advice generation, since the
system reports patterns and does not counsel; and any form of clinical diagnosis or
screening. The system is not an evaluator of business ideas and plays no gatekeeping
role.

**Ethical considerations.** Participation will be voluntary and informed. Collected
responses will be used solely as evaluation data and will not be used as input to
generative models. No personally identifying information will be collected or
published.

## 4. Initial Study and Work Done so Far

Our literature review found that automatic cognitive distortion detection has been
approached from four directions. Shreevastava and Foltz [3] established the benchmark
task and dataset and applied classical classifiers over lexical and embedding
features. Later work has framed the problem as multi-label rather than single-label,
recognising that a single utterance frequently exhibits more than one distortion, and
has extended the task to languages other than English. A separate line of work uses
large language models with zero-shot prompting to extract the distorted text before
classifying it, reporting that supplying the ground-truth distorted region
substantially improves classification accuracy — an observation that motivates the
present design, although that work does not train an extraction model. Maddela et al.
[6] released a large crowdsourced corpus of unhelpful thought patterns and their
reframings, which provides related but non-clinical training data. Domain-adapted
encoders such as MentalBERT and MentalRoBERTa [7] have been proposed for
mental-health text and serve as our backbone.

For the architecture, we draw on rationale extraction [8], [9], in which a model
identifies the supporting evidence for its own prediction and is trained end to end,
and on intermediate-task transfer [10] for making use of related corpora. For
identifying probable annotation errors efficiently, we follow confident learning
[11]. Our review found no prior work that jointly localises the distorted text,
assigns multiple labels, and models warranted concern; nor any work applying this
task to entrepreneurial self-reflection.

**Work completed.** We have reproduced the benchmark on a leakage-free pipeline with
fixed splits and three random seeds, obtaining a binary positive-class F1 of 0.794 ±
0.012, consistent with the 0.79 reported by the original authors.

We have also run a preliminary experiment that shapes the proposed architecture. On
identical splits and metrics, restricting the classifier's input to the
annotator-marked distorted sentence raised multi-label macro-F1 from 0.322 to
**0.402**, whereas keeping the full document and replacing the bag-of-words model
with a domain-pretrained transformer raised it from 0.322 to **0.357**. The input
representation is therefore worth **+0.080** and the encoder **+0.035** — the choice
of *what text the model reads* matters more than twice as much as the choice of
model. We further find that a document with its distorted sentence removed still
scores 0.281, so the surrounding text is weakly informative but acts as a net drag on
performance.

This result is an upper bound: it uses the human-marked region, which is unavailable
at prediction time. It establishes the value of learning to find that region, which
is what the proposed span-selection head is designed to do.

## References

[1] A. T. Beck, *Cognitive Therapy and the Emotional Disorders*. New York:
International Universities Press, 1976.

[2] D. D. Burns, *Feeling Good: The New Mood Therapy*. New York: William Morrow,
1980.

[3] S. Shreevastava and P. Foltz, "Detecting cognitive distortions from
patient-therapist interactions," in *Proceedings of the Seventh Workshop on
Computational Linguistics and Clinical Psychology (CLPsych)*, 2021, pp. 151–158.

[4] J. Cohen, "A coefficient of agreement for nominal scales," *Educational and
Psychological Measurement*, vol. 20, no. 1, pp. 37–46, 1960.

[5] J. R. Landis and G. G. Koch, "The measurement of observer agreement for
categorical data," *Biometrics*, vol. 33, no. 1, pp. 159–174, 1977.

[6] M. Maddela, M. Ung, J. Xu, A. Madotto, H. Foran, and Y.-L. Boureau, "Training
models to generate, recognize, and reframe unhelpful thoughts," in *Proceedings of
the 61st Annual Meeting of the Association for Computational Linguistics (ACL)*,
2023.

[7] S. Ji, T. Zhang, L. Ansari, J. Fu, P. Tiwari, and E. Cambria, "MentalBERT:
Publicly available pretrained language models for mental healthcare," in
*Proceedings of the Thirteenth Language Resources and Evaluation Conference (LREC)*,
2022, pp. 7184–7190.

[8] T. Lei, R. Barzilay, and T. Jaakkola, "Rationalizing neural predictions," in
*Proceedings of the 2016 Conference on Empirical Methods in Natural Language
Processing (EMNLP)*, 2016, pp. 107–117.

[9] J. DeYoung, S. Jain, N. F. Rajani, E. Lehman, C. Xiong, R. Socher, and B. C.
Wallace, "ERASER: A benchmark to evaluate rationalized NLP models," in *Proceedings
of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*,
2020, pp. 4443–4458.

[10] J. Phang, T. Févry, and S. R. Bowman, "Sentence encoders on STILTs:
Supplementary training on intermediate labeled-data tasks," *arXiv preprint*
arXiv:1811.01088, 2018.

[11] C. G. Northcutt, L. Jiang, and I. L. Chuang, "Confident learning: Estimating
uncertainty in dataset labels," *Journal of Artificial Intelligence Research*, vol.
70, pp. 1373–1411, 2021.

---

## ⚠ Before submission — remove this section

**Placeholders to fill:** advisor name, all member names and roll numbers, campus
(title page).

**Citations requiring verification.** `docs/REFERENCES.md` marks most entries as
recorded from memory. Confirm page numbers, venue, and identifiers against the
source before submitting: **[3]** (page range and the exact F1 variant behind 0.79,
and the ~1/3 agreement figure, both already flagged in `docs/EXPERIMENTS.md`),
**[6]**, **[7]** (author list), **[8]**, **[9]**, **[10]**, **[11]**.

**Citations deliberately left descriptive.** §4 refers to multi-label clinical work,
a Chinese-language dataset, and zero-shot span-extraction prompting without numbered
references, because verified citation details for those papers are not recorded in
this repository. Add them from your own literature review — they are named in the
planning documents at `Desktop\FYP\`. The zero-shot extraction paper is the source of
the ground-truth-span improvement referenced in §4 and should be cited there.

**Consistency check:** this proposal is aligned with `docs/FYP_PLAN.md`. If scope
changes, update both.
