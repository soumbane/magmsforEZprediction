# Sequence-Agnostic Model with Cross-Sequence Distillation for Localization of Seizure Onset Zone

Official PyTorch implementation of **"Non-invasive Localization of Seizure Onset Zone using
Clinically Acquired MRI in Children with Drug-Resistant Epilepsy: a Sequence-Agnostic Model with
Cross-Sequence Distillation"**, submitted to *Medical Image Analysis* (under review).

The model is a MAG-MS network that localizes the seizure onset zone from features derived from up to
five MRI sequences — T1w, T2w, FLAIR, DWI and DWIC (connectome). It is **sequence-agnostic**: one
model, trained once with all five sequences and cross-sequence distillation, can be evaluated on any
non-empty subset of them, because the absent encoders are dropped from the network's `target_dict`
rather than fed all-zero inputs. One binary classifier is trained per brain node, over 711 nodes of
the Lausanne parcellation.

## Requirements

* Python >= 3.9
* [PyTorch](https://pytorch.org) >= 2.0.1
* [torchmanager](https://github.com/kisonho/torchmanager) >= 1.2.0rc8
* [Monai](https://monai.io) >= 1.2
* [Multimodality / magnet](https://github.com/kisonho/multimodality/tree/feature-0201) >= 2.1a8 —
  not on PyPI; check out that branch and place it on your `PYTHONPATH`.
* numpy, scipy, pandas, openpyxl, scikit-learn, imbalanced-learn, tqdm

Exact versions used for the reported results are pinned in `pyproject.toml`.

## Repository layout

| Path | Contents |
|---|---|
| `ezpred/` | the model: multi-scale encoders, fusion, classifier head, metrics, configs |
| `data/` | dataset loader, SMOTE augmentation, and the two external-cohort staging scripts |
| `train_left.py`, `train_right.py` | per-node training entry points |
| `eval_left.py`, `eval_right.py` | evaluation on the additional cohort, all 31 sequence combinations |
| `eval_bonn_left.py`, `eval_bonn_right.py` | evaluation on the Bonn cohort, T1/FLAIR only |
| `train_ALL_add_*.sh`, `eval_ALL_add_*.sh`, `eval_ALL_bonn_*.sh` | per-GPU node shards |
| `run_all_*.sh`, `finish_bonn_evaluation.sh` | detached launchers that survive closing SSH |
| `combine_*.py`, `make_*_table.py` | aggregate per-node results into the publication tables |
| `results/` | **the reported numbers**, with a README per cohort |
| `sota_node_level/` | baseline and state-of-the-art comparisons |

Paths to the data and experiment roots are absolute constants at the top of the scripts that use
them; change them to match your own layout.

## Data

This project uses a private clinical dataset of MRI-derived features from children with
drug-resistant epilepsy: for every node, a 1899-dimensional vector laid out as

```
T1 [0:300]   T2 [300:500]   FLAIR [500:700]   DWI [700:1400]   DWIC [1400:1899]
```

For access to similar data for research purposes, please contact the authors. The second external
cohort is public — see below.

### Preprocessing

SMOTE augmentation balances the two classes and enlarges the training set:

```bash
python data/augment_single_node_left_hemis_part2.py
python data/augment_single_node_right_hemis_part2.py
```

`data/dataset_ez.py` then loads the augmented data. `data/left_hemis_part2_subgroups.py` and its
right-hemisphere counterpart build the MR-/MR+/SF/SR subgroup splits used in the subgroup analysis.

## Cohorts

Three cohorts are reported. Composition, EZ counts and the informative-vs-degenerate node
distinction are documented in **[`results/COHORTS.md`](results/COHORTS.md)**.

| Cohort | Subjects | Sequences | Selected in code by |
|---|---|---|---|
| Original held-out validation | 10 patients | all five | `data.EZMode.TEST` |
| Additional cohort | 30 (17 with DWIC, 13 without) | all five / no DWIC | `data.EZMode.ADD_17`, `ADD_13` |
| Bonn cohort (OpenNeuro [ds004199](https://doi.org/10.18112/openneuro.ds004199.v1.0.5)) | 85 | T1 + FLAIR only | `data.EZMode.BONN` |

Stage the two external cohorts before evaluating them:

```bash
python data/prepare_add_cohort.py     # writes Node_{N}/Add_Val_Data_{17_withDWIC,13_noDWIC}/
python data/prepare_bonn_cohort.py    # writes Node_{N}/Bonn_Val_Data_85/, recovers + verifies labels
```

> **Choosing the validation cohort.** The scripts as committed train and select checkpoints against
> the additional cohort, which is what the headline results report. To use the original 10-patient
> held-out cohort instead, switch the validation dataset in `train_left.py` / `train_right.py`
> (`EZMode.ADD_17` → `EZMode.TEST`, marked by a comment) and the output `path` near the end of the
> same file; in `eval_*.py`, change the `GROUPS` entry to `EZMode.TEST`.

## Training

Trains 3 trials per node with all five sequences and cross-sequence distillation, writing
checkpoints to `experiments/exp_node{N}/AddCohort/magms_trial{i}.exp/`.

```bash
# every shard, detached across the 4 GPUs, survives closing the SSH connection
./run_all_add_training.sh

# progress
grep -c "saved at" logs_add_root/*.log
```

Individual shards (`train_ALL_add_left.sh` … `train_ALL_add_right_5.sh`) can be run one at a time,
but they run in the foreground. Batch size, learning rate, epochs, sequence selection and device are
set as flags inside each shard; see `train_left.py` for the defaults.

To ablate cross-sequence distillation, uncomment the `NO Distillation` loss block in
`train_left.py` / `train_right.py`, which zeroes the KL and MSE terms and the per-modality
supervision, leaving only cross-entropy on the fused branch.

## Evaluation

### Additional cohort — all 31 sequence combinations

```bash
./run_all_add_evaluation.sh          # 8 shards across 4 GPUs

python combine_add_cohort_results.py --hemisphere left  --group 17
python combine_add_cohort_results.py --hemisphere left  --group 13
python combine_add_cohort_results.py --hemisphere right --group 17
python combine_add_cohort_results.py --hemisphere right --group 13
python combine_add_cohort_training_results.py
python make_addcohort_combination_table.py
```

Each node is scored on both subject groups with the same 3 checkpoints: the 17 subjects with all
five sequences on all 31 combinations, and the 13 without DWIC on the 15 DWIC-free combinations.
Every trial is kept as its own column, so `max` or `mean` over trials can be chosen afterwards.

### Bonn cohort — T1, FLAIR, T1+FLAIR

No training happens here: the additional cohort's checkpoints are reused unchanged, which makes this
a fully external test.

```bash
./run_all_bonn_evaluation.sh                                          # 8 shards, ~30 min
setsid nohup ./finish_bonn_evaluation.sh > logs_bonn_eval/finish.log 2>&1 &
```

`finish_bonn_evaluation.sh` waits for the shards, then combines both label variants and both
hemispheres and rebuilds the tables. It deliberately skips the publication table if any node is
missing, so no average over an unknown subset of nodes is ever reported.

### Original held-out cohort

After switching the cohort as described above, combine the per-node sheets with:

```bash
python combine_node_excel_sheet_results_eval_left.py
python combine_node_excel_sheet_results_eval_right.py
```

## Results

All reported numbers are committed under [`results/`](results/).

| File | Contents |
|---|---|
| **[`results/Combined_AddCohort_BonnCohort_table.xlsx`](results/Combined_AddCohort_BonnCohort_table.xlsx)** | the main table — both external cohorts side by side, rows 1-31 |
| [`results/COHORTS.md`](results/COHORTS.md) | cohort composition and how the means are computed |
| [`results/addcohort/`](results/addcohort/) | per-node additional-cohort results, with a README |
| [`results/bonncohort/`](results/bonncohort/) | per-node Bonn results, with a README |

The main workbook has three sheets:

* `all_nodes` — every node of the hemisphere;
* `nodes_with_EZ` — only the **informative** nodes, whose validation split contains at least one EZ
  subject. At a degenerate node balanced accuracy collapses to non-EZ accuracy, so this is the sheet
  to read for model quality;
* `bonn_specificity` — the Bonn subjects scored with every label forced non-EZ. This is a
  true-negative rate, **not** a balanced accuracy.

The Bonn columns are populated only at rows 4, 16 and 20 (FLAIR, T1, T1+FLAIR) and are `NA`
elsewhere: those subjects have no T2/DWI/DWIC acquisition, and zero-filling a missing encoder shifts
the logits rather than removing the branch, so any number in those cells would be an artefact.
`results/bonncohort/README.md` quantifies this.

## Baselines

`sota_node_level/` holds the comparison methods: random forest, MLP, deep-FCD and a relational
reasoning network, each swept over the same node set and sequence combinations.

## Citation

```bibtex
@article{banerjee2025sequence,
  title   = {Non-invasive Localization of Seizure Onset Zone using Clinically Acquired MRI in
             Children with Drug-Resistant Epilepsy: a Sequence-Agnostic Model with
             Cross-Sequence Distillation},
  author  = {Banerjee, Soumyanil and He, Qisheng and others},
  journal = {Medical Image Analysis},
  year    = {2025},
  note    = {Under review}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
