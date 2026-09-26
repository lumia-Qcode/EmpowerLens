# EmpowerLens Experimental Results (Run 2)

## Dataset Characteristics & Audit
**Sequence Length & Truncation (Exp 8)**
*   **Total Samples:** 2,530
*   **Median Length:** 155 tokens
*   **Max Length:** 1,394 tokens
*   **Truncation at 512 tokens:** 3.16% (Safe for standard Roberta limits)

**Label Imbalance (Exp 1 - Train Split Top 5)**
| Cognitive Distortion | Prevalence (%) | Count |
|----------------------|----------------|-------|
| Mind Reading         | 11.46%         | 232   |
| Overgeneralization   | 10.97%         | 222   |
| Magnification        | 9.68%          | 196   |
| Fortune Telling      | 8.30%          | 168   |
| Personalization      | 8.00%          | 162   |

## Dataset Ablation (Exp 2)
Comparing training data configurations on the Multilabel test set (metrics are averaged across seeds 42, 1337, and 2024).

| Configuration       | Test Macro-F1 | Test Micro-F1 |
|---------------------|---------------|---------------|
| Annotated Only      | 0.268         | 0.281         |
| Codipas Only        | 0.336         | 0.373         |
| Annotated + Codipas | 0.230         | 0.231         |

## Class Imbalance Mitigation (Exp 3–5)
Evaluating loss functions and sampling strategies on the Multiclass task. *(Note: `results_multiclass_v2.csv` logs identical baseline Cross-Entropy metrics).*

Note:  results from exp 3,4,5 are grouped together because they all target class imbalance

| Strategy (Experiment)   | Test Macro-F1 (Mean ± Std) | Test Micro-F1 (Mean ± Std) |
|-------------------------|----------------------------|----------------------------|
| Cross-Entropy (Baseline)| 0.1906 ± 0.0092            | 0.3372 ± 0.0060            |
| Focal Loss (Exp 3)      | 0.1555 ± 0.0363            | 0.1884 ± 0.0471            |
| Class-Balanced (Exp 4)  | 0.2127 ± 0.0088            | 0.3175 ± 0.0120            |
| Weighted CE (Exp 4)     | 0.2115 ± 0.0287            | 0.3056 ± 0.0194            |
| Weighted Sampler (Exp 5)| 0.1304 ± 0.0137            | 0.1331 ± 0.0091            |

## Architecture: Flat vs. Cascade (Exp 6–7)
Comparison of the flat multilabel model (`results_multilabel_flat.csv`) against the two-stage cascade architecture (`results_cascade.csv`).

Note:  results from exp 6,7 are grouped together because they both focus on the model's architecture


**Stage 1: Binary Detection (`results_stage1.csv`)**
| Seed | Test Macro-F1 | Test Positive Class F1 |
|------|---------------|------------------------|
| 42   | 0.7073        | 0.7975                 |
| 1337 | 0.7215        | 0.7831                 |
| 2024 | 0.7057        | 0.7987                 |

**Stage 2: End-to-End Multilabel Classification**
| Architecture                   | Seed | Test Macro-F1 | Test Micro-F1 |
|--------------------------------|------|---------------|---------------|
| **Flat Baseline** (Exp 6)      | 42   | 0.2568        | 0.2776        |
|                                | 1337 | 0.2656        | 0.2661        |
|                                | 2024 | 0.2886        | 0.3014        |
| **Cascade End-to-End** (Exp 7) | 42   | 0.1660        | 0.1887        |
|                                | 1337 | 0.2613        | 0.2741        |
|                                | 2024 | 0.2605        | 0.2739        |

**Stage 2: Isolated Component Evaluation (`results_stage2.csv`)**
*Evaluating the Stage 2 multilabel classifier independently (trained on gold labels without cascading error propagation from Stage 1).*
| Seed | Test Macro-F1 | Test Micro-F1 |
|------|---------------|---------------|
| 42   | 0.2207        | 0.2531        |
| 1337 | 0.3030        | 0.3192        |
| 2024 | 0.2976        | 0.3183        |