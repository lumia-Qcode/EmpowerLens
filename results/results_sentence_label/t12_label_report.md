# T12 sentence-label report

Rows in corpus: 2530 | distorted rows with a span: 1597 | kept for selector: 1584
Sentences: 16931 (positive 2772, 16.4%)

## How spans were located (distorted rows)
- exact: 1355 (84.8%)
- normalized: 27 (1.7%)
- anchor: 199 (12.5%)
- fuzzy: 3 (0.2%)
- none (excluded): 13 (0.8%)

## Positives per row
- 1: 974 | 2: 363 | 3+: 247
- anchor rows that fell back to overlap (no sentence fully contained): 1
- excluded no_distortion rows: 933

## Rule
Overlap (>=1 char) for exact/normalized/fuzzy; containment for anchor rows; unlocated and no_distortion rows excluded. Sentence splitter: T11 regex (src/span_check.split_sentences).