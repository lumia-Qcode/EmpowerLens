=== CBT-BENCH COMBINED TRANSFER EVALUATION ===
Items successfully mapped: 146
Items dropped (exclusion audit): 0

=== WHY WE RECORDED BOTH SCORES ===
The original research paper formulated cognitive distortion detection strictly as a multi-class classification problem. We recorded the multi-class scores first to establish a faithful baseline that accurately replicates those original constraints. We then recorded the multi-label scores to demonstrate empirically how upgrading to a dedicated multi-label architecture resolves structural bottlenecks and correctly handles the overlapping, real-world data found in CBT-Bench.

--- PHASE 1: MULTI-CLASS BASELINE RESULTS ---
                     precision    recall  f1-score   support

emotional_reasoning       0.38      0.08      0.14        36
 overgeneralization       0.00      0.00      0.00        32
      mental_filter       0.11      0.10      0.10        21
  should_statements       0.41      0.25      0.31        28
     all_or_nothing       0.60      0.05      0.09        65
       mind_reading       0.43      0.13      0.20        47
    fortune_telling       0.31      0.11      0.17        44
      magnification       0.25      0.04      0.07        25
    personalization       0.27      0.07      0.11        42
           labeling       0.00      0.00      0.00        29

          micro avg       0.30      0.08      0.13       369
          macro avg       0.28      0.08      0.12       369
       weighted avg       0.32      0.08      0.12       369
        samples avg       0.21      0.08      0.12       369

--- PHASE 2: MULTI-LABEL ARCHITECTURE RESULTS ---
                     precision    recall  f1-score   support

emotional_reasoning       0.27      0.42      0.33        36
 overgeneralization       0.33      0.41      0.36        32
      mental_filter       0.09      0.19      0.12        21
  should_statements       0.23      0.39      0.29        28
     all_or_nothing       0.73      0.42      0.53        65
       mind_reading       0.47      0.49      0.48        47
    fortune_telling       0.34      0.41      0.37        44
      magnification       0.19      0.56      0.29        25
    personalization       0.30      0.43      0.35        42
           labeling       0.17      0.28      0.21        29

          micro avg       0.30      0.41      0.34       369
          macro avg       0.31      0.40      0.33       369
       weighted avg       0.37      0.41      0.37       369
        samples avg       0.24      0.41      0.27       369
