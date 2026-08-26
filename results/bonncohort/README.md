# BonnCohort results

Second external validation cohort: **85 people with focal cortical dysplasia type II** from the
public OpenNeuro dataset [ds004199](https://doi.org/10.18112/openneuro.ds004199.v1.0.5) (University
of Bonn, CC0), covering the same 711 nodes as the additional cohort.

**No training happened here.** Every node reuses its three `AddCohort` checkpoints, trained on the
original SMOTE-augmented training data (58 patients) with all five sequences and cross-sequence
distillation. The cohort is therefore fully external: nothing about it influenced training or
checkpoint selection.

Copies of the files under `Data/All_Hemispheres/BonnCohort/`, regenerated with
`combine_bonn_cohort_results.py` and `make_bonncohort_combination_table.py`.

## What the cohort has

These subjects were scanned with **T1 and FLAIR only**. In the source, the T2 and DWI segments are
`NaN` and the connectome is 100% `NaN`, so those are missing acquisitions rather than zero
measurements. At inference the absent sequences are **dropped from the model's `target_dict`**, not
supplied as all-zero branches — which is the sequence-agnostic claim doing real work, verified on a
node with 6 EZ subjects where dropping and zero-filling give different answers.

So the sweep is the 3 non-empty subsets of {T1, FLAIR} (`get_target_dict` numbers 4, 16 and 20),
not the 31 of the additional cohort.

## Files

| File | Contents |
|---|---|
| `BonnCohort_combination_table.xlsx` | the 3-row table, sheets `all_nodes` and `nodes_with_EZ` |
| `BonnCohort_left_combined.xlsx` | 286 left nodes, sheets `per_trial` / `max_over_trials` / `mean_over_trials` |
| `BonnCohort_right_combined.xlsx` | 425 right nodes, same three sheets |
| `info_bonn_cohort.xlsx` | per-node EZ and non-EZ counts |

Column A (`Combination`) names the sequences used at inference. Sensitivity and specificity treat
**non-EZ as the positive class** (class 0), matching the rest of the project.

## The labels had to be recovered — and the recovery was later confirmed

When this cohort was first staged, the per-node export at `/BonnData/` carried **no labels**: all
60,435 (85 × 711) entries were zero. They were recovered from the BIDS source
`Bonn_Cohort_Label.mat`, laid out subject-major over the 998-ROI Lausanne parcellation as
`row = subject * 998 + (node - 1)`.

**The export was subsequently regenerated with real labels, and they agree with the recovered ones
in every one of the 60,435 cells** (861 EZ labels in both, 0 nodes differing). `data/prepare_bonn_cohort.py`
now asserts that agreement per node, so the recovery is a checked invariant rather than a one-off.
All reported results were computed against these labels and are unaffected.

Features still come from the export: staging asserts per node that the two agree bit-for-bit (0
mismatches over all 711 nodes). `verify_label_alignment` in `data/prepare_bonn_cohort.py` proves
features and labels share one subject ordering — `BonnData_ROI` recovers the per-subject lesion side,
and that one side vector reassembles both `Bonn_Cohort_RI` and `Bonn_Cohort_Label` from their
ipsilateral/contralateral halves.

Recovered: **861 EZ labels over 455 of 711 nodes** (145/286 left, 310/425 right), 80 of 85 subjects,
prevalence 1.42%.

Note the 85 rows are **not** ordered by `participant_id` — matching them that way agrees with
`participants.tsv` on only 34/85, i.e. chance. Row alignment is proven; subject identity is not.

## Reading the numbers

**Use the `nodes_with_EZ` sheet.** At the 256 nodes with no EZ subject, `BalancedAccuracyScore`
falls back to `specificity = sensitivity`, so the reported value is really just non-EZ accuracy and
is often 1.0. Because this cohort's prevalence is low (1.42% vs 6.83% for the AddCohort 17-group),
the `all_nodes` mean is inflated more here than anywhere else in the project — it reads *higher* than
the AddCohort figures despite the model performing *worse*.

Balanced accuracy on the informative nodes, best of 3 trials, averaged over nodes:

| Combination | Bonn left (n=145) | Bonn right (n=310) | AddCohort left (n=195) | AddCohort right (n=308) |
|---|---|---|---|---|
| FLAIR | 0.5513 | 0.5491 | 0.5647 | 0.5623 |
| T1 | 0.5475 | 0.5390 | 0.5706 | 0.5619 |
| T1-FLAIR | 0.5455 | 0.5424 | 0.5782 | 0.5819 |
| *all five sequences* | n/a | n/a | *0.8120* | *0.8107* |

**The honest reading.** Cross-site transfer is cheap; the missing sequences are expensive. Going from
the AddCohort to Bonn on the *same* two sequences costs only about 0.03 balanced accuracy
(0.578 → 0.546 left). Going from five sequences to two costs about 0.23 on the *same* cohort
(0.812 → 0.578). The model is close to chance on T1+FLAIR alone, on either cohort, so this result
bounds what the architecture can do without the diffusion sequences rather than showing a site-
specific failure.

Two further caveats worth stating in the manuscript:

- These are **FCD type II lesion masks** used as EZ ground truth. Defensible for FCD II, and the
  source names the files `SOZ_*`, but they are not electrographically-defined SOZ.
- FLAIR sits roughly one standard deviation above the training distribution (train 0.472 /
  AddCohort 0.519 / **Bonn 0.605**), while T1 aligns well (0.531 / 0.508 / 0.528).

## The joint table, and why 28 rows are NA

`../Combined_AddCohort_BonnCohort_table.xlsx` puts both external cohorts side by side in the 1-31
layout used for the AddCohort table:

```
 #  T1w  T2w  FLAIR  DWI  DWIC  AddCohort_Left  AddCohort_Right  BonnCohort_Left  BonnCohort_Right
 1    0    0      0    0     1          0.6797           0.6889               NA                NA
 4    0    0      1    0     0          0.6769           0.6698           0.7280            0.6483
16    1    0      0    0     0          0.6978           0.6657           0.7514            0.6375
20    1    0      1    0     0          0.7068           0.6898           0.7403            0.6421
31    1    1      1    1     1          0.8702           0.8624               NA                NA
```

The AddCohort subjects have all five sequences, so all 31 rows are populated. The Bonn subjects have
**T1 and FLAIR only**, so only rows 4, 16 and 20 exist for them; the rest are `NA`.

Those 28 cells could have been *computed* — `target_dict` selects which encoders run and does not
check whether data exists, so asking for "T2" on this cohort is accepted and the T2 encoder simply
receives an all-zero tensor. That number would be an artefact, not a measurement:

- a zero input does **not** give a zero output. Every encoder emits a non-zero constant (abs-mean
  0.0175 for FLAIR up to 0.0929 for T2) because of conv bias and normalisation;
- at inference the branch outputs are summed (`MAGNET2.forward`: `if not self.training: y = y.sum(1)`),
  so each empty branch injects a constant into the logits;
- measured at node 58, trial 3: T1-FLAIR predicted 68/85 subjects as EZ, while "all five sequences"
  predicted 83/85 — **15 of 85 predictions flipped** purely from adding three empty branches.

So a value in row 31 for this cohort would mean *T1 + FLAIR + three constant biases*, not "all five
sequences". `NA` is the honest entry.

## Specificity analysis (every pair treated as non-EZ)

The same 85 subjects and the *same* features are also scored with **every label forced to non-EZ**.
Files: `BonnCohort_{left,right}_all_nonEZ_combined.xlsx`, and sheet `bonn_specificity` of the joint
table.

This label set is constructed in code, not read from a file. It was first run when the `/BonnData/`
export shipped all-zero labels, so the two coincided then; the export has since been regenerated
with real labels and no longer does, which is why the construction is now explicit.

Because every node-subject pair counts as non-EZ, there are no negatives and
`BalancedAccuracyScore` falls back to `specificity = sensitivity`. **The number is non-EZ accuracy —
the fraction of the 85 subjects the model correctly declines to call EZ, i.e. the true-negative rate
for EZ detection. It is not a balanced accuracy** and must not be placed in the same column as one.
Every value is an exact multiple of 1/85, which confirms it is a plain count.

| # | Sequences | Left (n=286) | Right (n=425) |
|---|---|---|---|
| 4 | FLAIR | 0.9053 | 0.9154 |
| 16 | T1 | 0.9548 | 0.9132 |
| 20 | T1-FLAIR | 0.9399 | 0.9262 |

There is no `nodes_with_EZ` counterpart: under these labels no node contains an EZ subject.

Two caveats. The metric is **contaminated by 1.42%** — 861 of the 60,435 node-subject pairs really
are EZ, and here they are counted as non-EZ, so the true specificity is very slightly lower. And a
high value here is easy to obtain by predicting non-EZ everywhere; read it alongside the
balanced accuracy above, never on its own.

## Reproducing

```bash
python data/prepare_bonn_cohort.py          # stage + recover labels + verify alignment
./run_all_bonn_evaluation.sh                # 8 shards across 4 GPUs, ~30 min
./finish_bonn_evaluation.sh                 # waits, then combines and tabulates
```
