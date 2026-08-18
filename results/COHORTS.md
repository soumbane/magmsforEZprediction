# Cohort composition and how the reported numbers are computed

Reference for the manuscript. Machine-readable version: `cohort_composition.xlsx`.

## Informative vs degenerate nodes

The model is one binary classifier **per node** — 711 of them. At each node the validation cohort's
subjects split into EZ (class 1) and non-EZ (class 0).

- **Informative node** — the split contains at least one EZ subject. Both classes are present, so
  balanced accuracy is a genuine balanced accuracy.
- **Degenerate node** — the split contains no EZ subject at all. Every subject is non-EZ.

At a degenerate node the metric changes meaning. `ezpred/metrics/bal_accuracy.py` uses
`class_index = 0`, i.e. **non-EZ is the positive class**, so at such a node every subject is a
positive, `tn + fp = 0`, and specificity is undefined. The implementation falls back:

```python
specificity  = tn / (tn + fp + eps) if torch.sum(tn + fp) > 0 else sensitivity
balanced_acc = 0.5 * (sensitivity + specificity)      # collapses to sensitivity
```

So the value reported at a degenerate node is **non-EZ accuracy**, not balanced accuracy, and it is
easy to score high on: predicting non-EZ everywhere gives 1.0. For the AddCohort five-sequence
result, **95.6% of degenerate-node values are exactly 1.0**.

## How the two sheets are computed

Both sheets report, per sequence combination and hemisphere, the **mean over nodes of each node's
best balanced accuracy across its 3 training trials** (max first, then mean). They differ only in
which nodes enter the mean.

- `all_nodes` — every node of that hemisphere.
- `nodes_with_EZ` — informative nodes only.

`all_nodes` is a plain unweighted mean over nodes, which is algebraically a weighted average of the
two groups with weights equal to their node counts. Worked example, AddCohort left, all five
sequences:

```
informative:  n = 195, mean = 0.812018
degenerate :  n =  91, mean = 0.994829
(195 x 0.812018 + 91 x 0.994829) / 286 = 0.870185
reported all_nodes value                = 0.870185     identical
```

The headline 0.8702 is therefore 68% real balanced accuracy and 32% near-1.0 filler.

## Composition

| Cohort | Group | Hemi | Nodes | Subj/node | Pairs | EZ pairs | EZ % | Informative | Degenerate |
|---|---|---|---|---|---|---|---|---|---|
| Original | 58-patient training (pre-SMOTE) | left | 286 | 51–58 | 16,051 | 600 | 3.74 | 286 (100%) | 0 |
| Original | 58-patient training (pre-SMOTE) | right | 425 | 45–58 | 23,265 | 1,274 | 5.48 | 425 (100%) | 0 |
| Original | 10-patient validation | left | 286 | **9–10** | 2,840 | 45 | 1.58 | 42 (14.7%) | **244 (85.3%)** |
| Original | 10-patient validation | right | 425 | **7–10** | 4,016 | 167 | 4.16 | 140 (32.9%) | 285 (67.1%) |
| AddDataSet | 17 subjects, all 5 sequences | left | 286 | 17 | 4,862 | 276 | 5.68 | 195 (68.2%) | 91 (31.8%) |
| AddDataSet | 17 subjects, all 5 sequences | right | 425 | 17 | 7,225 | 549 | 7.60 | 308 (72.5%) | 117 (27.5%) |
| AddDataSet | 13 subjects, no DWIC | left | 286 | 13 | 3,718 | 164 | 4.41 | 127 (44.4%) | 159 (55.6%) |
| AddDataSet | 13 subjects, no DWIC | right | 425 | 13 | 5,525 | 124 | 2.24 | 108 (25.4%) | 317 (74.6%) |
| BonnDataSet | 85 subjects, T1+FLAIR only | left | 286 | 85 | 24,310 | 269 | 1.11 | 145 (50.7%) | 141 (49.3%) |
| BonnDataSet | 85 subjects, T1+FLAIR only | right | 425 | 85 | 36,125 | 592 | 1.64 | 310 (72.9%) | 115 (27.1%) |

After SMOTE the training set is balanced to **120 samples per node** (60 EZ / 60 non-EZ):
17,160 / 17,160 left and 25,500 / 25,500 right.

Note the original validation cohort is **not 10 subjects at every node** — it varies 9–10 (left) and
7–10 (right), because not every patient contributes every node.

## Why all_nodes cannot compare cohorts

The degenerate fraction differs by cohort and hemisphere (31.8% AddCohort left, 49.3% Bonn left,
27.1% Bonn right), and degenerate nodes score near 1.0. So the `all_nodes` column mixes two
different quantities in a different ratio for every column, and the comparison inverts:

| T1-FLAIR | informative | degenerate | all nodes |
|---|---|---|---|
| AddCohort left | 0.5782 (n=195) | 0.9825 (n=91) | 0.7068 |
| Bonn left | **0.5455** (n=145) | 0.9408 (n=141) | **0.7403** |

On comparable nodes Bonn is *worse*; in the all-nodes column it looks *better*, purely because it has
50 more degenerate nodes. Report `nodes_with_EZ` for any cross-cohort claim, and if `all_nodes` is
reported, state the degenerate counts beside it.
