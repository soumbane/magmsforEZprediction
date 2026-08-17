# AddCohort results

Results of the additional validation cohort: 30 independent subjects covering the same 711 nodes,
13 of whom were scanned without DWIC.

For every node, the model was trained on the original SMOTE-augmented training data (58 patients,
120 balanced samples) with all five sequences and cross-sequence distillation, for 3 trials. The
best checkpoint of each trial was selected on the 17 subjects that have all five sequences, and then
scored on both subject groups.

Copies of the files under
`Data/All_Hemispheres/AddCohort/`, regenerated with `combine_add_cohort_results.py` and
`combine_add_cohort_training_results.py`.

## Combination sweep

| File | Group | Combinations | Nodes |
|---|---|---|---|
| `AddCohort_left_group17_combined.xlsx` | 17 subjects, all 5 sequences | 31 | 286 |
| `AddCohort_right_group17_combined.xlsx` | 17 subjects, all 5 sequences | 31 | 425 |
| `AddCohort_left_group13_combined.xlsx` | 13 subjects, no DWIC | 15 | 286 |
| `AddCohort_right_group13_combined.xlsx` | 13 subjects, no DWIC | 15 | 425 |

Each holds balanced accuracy in three sheets:

- `per_trial` — one row per modality combination, one column per node and trial (`Node_42_Trial_1`, ...)
- `max_over_trials` — best of the 3 trials, one column per node
- `mean_over_trials` — average of the 3 trials, one column per node

Column A (`Combination`) names the sequences used at inference, e.g. `T1-FLAIR-DWIC`. The
`group17` files cover all 31 non-empty subsets of the five sequences. The `group13` files cover the
15 non-empty subsets of {T1, T2, FLAIR, DWI}: those subjects have no DWIC, so DWIC is dropped from
the model's `target_dict` rather than supplied as an all-zero input.

## Training-time validation

`AddCohort_training_results_all_nodes.xlsx`

- `validation` — one row per node: balanced accuracy, sensitivity and specificity of the 3 trials on
  the 17-subject group with all five sequences, plus `Max_Val_Bal_Acc` and `Mean_Val_Bal_Acc`. This
  is the metric that selected each trial's checkpoint.
- `training` — the same metrics on the 120-sample training set.

These numbers are the same as the `T1-T2-FLAIR-DWI-DWIC` row of the `group17` sweep.

## Cohort composition

`info_add_cohort.xlsx` — per-node EZ and non-EZ counts for both groups.

## Reading the numbers

Sensitivity and specificity treat **non-EZ as the positive class** (class 0), matching the rest of
the project.

Filter to nodes whose validation split actually contains an EZ subject: 503 of 711 for the
17-subject group and 235 of 711 for the 13-subject group. For the rest, `BalancedAccuracyScore`
falls back to `specificity = sensitivity`, so the reported balanced accuracy is really just the
non-EZ accuracy and is often 1.0. Use the `Has_EZ_subject` column of the training file, or the
counts in `info_add_cohort.xlsx`, to exclude them. Averaged over the informative nodes, taking the
max of the 3 trials, the full five-sequence combination reaches 0.812 (left) and 0.811 (right).

Comparing the two groups directly understates the model: they are different subjects with different
EZ prevalence. The like-for-like comparison is `T1-T2-FLAIR-DWI` scored on the same 17 subjects.
