# Inter-annotator agreement (overlap set)

- Rows merged: **1665**
- Overlap rows: **60** x 4 annotators
- Invalid rows (not dropped): **24**
- Primary filled from the original label (CS majors blanks): **81** (overlap: 0)
- Unsure rows flagged for adjudication: **45**

## Invalid rows

All invalid rows are blank Primary cells in the returned workbooks (Laiba 1, Reesha 5, Hurema 18). They were left blank, not filled, so they carry no vote: the annotator's label is simply missing for that row. Two of Hurema's blanks (2057, 414) sit in the overlap set, so those two items have 3 raters instead of 4. Fleiss' kappa keeps only items with the most common rater count, so it is computed on the remaining items. Two rows are absent from the returned workbooks altogether (1924 from Reesha, 1908 from CS majors).

| file                    |   Row ID |   annotator | reason           |
|:------------------------|---------:|------------:|:-----------------|
| annotator_1_Laiba.xlsx  |     2546 |           1 | Primary is blank |
| annotator_2_Reesha.xlsx |     2401 |           2 | Primary is blank |
| annotator_2_Reesha.xlsx |      257 |           2 | Primary is blank |
| annotator_2_Reesha.xlsx |     4670 |           2 | Primary is blank |
| annotator_2_Reesha.xlsx |      423 |           2 | Primary is blank |
| annotator_2_Reesha.xlsx |     2222 |           2 | Primary is blank |
| annotator_3_Hurema.xlsx |     2057 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |      414 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |     1057 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |     4575 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |     1589 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |     1762 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |     2084 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |     2163 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |      296 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |     1138 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |      695 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |      781 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |     2326 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |     1449 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |      957 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |     1688 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |      600 |           3 | Primary is blank |
| annotator_3_Hurema.xlsx |     2318 |           3 | Primary is blank |

## 12-way (with CP_LABEL)

- Items: 60, categories used: 12

| Metric | Value | 95% CI |
| --- | --- | --- |
| Percent agreement (pairwise) | 0.300 | [0.236, 0.364] |
| Percent agreement (full) | 0.067 | — |
| Fleiss' kappa | 0.172 | [0.103, 0.233] |
| Light's kappa | 0.195 | [0.135, 0.257] |
| Cohen's kappa range | 0.097 – 0.407 | — |

**Pairwise Cohen's kappa:** 1-2: 0.206, 1-3: 0.407, 1-4: 0.116, 2-3: 0.177, 2-4: 0.169, 3-4: 0.097

**By discipline:** psych-psych: 0.263, cs-cs: n/a, psych-cs: 0.127

**Label distribution:** {'No Distortion': 72, 'Magnification': 25, 'Labeling': 25, 'Mind Reading': 23, 'No Distortion (calibrated pessimism)': 20, 'Fortune-telling': 13, 'Personalization': 13, 'Overgeneralization': 13, 'Emotional Reasoning': 11, 'All-or-nothing thinking': 11, 'Should statements': 6, 'Mental filter': 5}

## 11-way (CP collapsed)

- Items: 60, categories used: 11

| Metric | Value | 95% CI |
| --- | --- | --- |
| Percent agreement (pairwise) | 0.383 | [0.308, 0.461] |
| Percent agreement (full) | 0.150 | — |
| Fleiss' kappa | 0.226 | [0.149, 0.299] |
| Light's kappa | 0.242 | [0.173, 0.314] |
| Cohen's kappa range | 0.177 – 0.430 | — |

**Pairwise Cohen's kappa:** 1-2: 0.203, 1-3: 0.430, 1-4: 0.224, 2-3: 0.177, 2-4: 0.223, 3-4: 0.199

**By discipline:** psych-psych: 0.270, cs-cs: n/a, psych-cs: 0.215

**Label distribution:** {'No Distortion': 92, 'Magnification': 25, 'Labeling': 25, 'Mind Reading': 23, 'Fortune-telling': 13, 'Personalization': 13, 'Overgeneralization': 13, 'Emotional Reasoning': 11, 'All-or-nothing thinking': 11, 'Should statements': 6, 'Mental filter': 5}

## binary

- Items: 60, categories used: 2

| Metric | Value | 95% CI |
| --- | --- | --- |
| Percent agreement (pairwise) | 0.714 | [0.642, 0.783] |
| Percent agreement (full) | 0.483 | — |
| Fleiss' kappa | 0.367 | [0.219, 0.505] |
| Light's kappa | 0.393 | [0.258, 0.521] |
| Cohen's kappa range | 0.220 – 0.655 | — |

**Pairwise Cohen's kappa:** 1-2: 0.329, 1-3: 0.655, 1-4: 0.440, 2-3: 0.220, 2-4: 0.441, 3-4: 0.276

**By discipline:** psych-psych: 0.401, cs-cs: n/a, psych-cs: 0.386

**Label distribution:** {'distorted': 145, 'none': 92}

## Calibrated pessimism

Rows originally labelled as a distortion that the majority marked `No Distortion (calibrated pessimism)`: **161**

IDs: [18, 25, 45, 58, 65, 72, 78, 93, 106, 124, 131, 141, 151, 176, 178, 199, 219, 227, 230, 231, 232, 233, 234, 249, 263, 275, 279, 292, 300, 303, 329, 347, 354, 384, 385, 392, 418, 427, 435, 455, 481, 490, 524, 548, 580, 586, 601, 603, 605, 630, 638, 642, 656, 666, 704, 708, 709, 730, 731, 766, 770, 777, 784, 789, 791, 800, 850, 868, 885, 886, 919, 931, 967, 976, 978, 1007, 1030, 1044, 1045, 1047, 1053, 1081, 1140, 1190, 1197, 1202, 1206, 1231, 1237, 1288, 1289, 1442, 1444, 1454, 1475, 1526, 1578, 1587, 1588, 1637, 1655, 1675, 1679, 1692, 1724, 1747, 1781, 1798, 1803, 1838, 1857, 1877, 1879, 1881, 1902, 1905, 1939, 1956, 1981, 1994, 2007, 2019, 2139, 2158, 2167, 2184, 2218, 2220, 2221, 2297, 2309, 2323, 2350, 2371, 2407, 2418, 2437, 2466, 2475, 2488, 2494, 2513, 2528, 2536, 2557, 4510, 4514, 4515, 4528, 4549, 4555, 4563, 4570, 4592, 4618, 4621, 4625, 4627, 4667, 4669, 4699]
