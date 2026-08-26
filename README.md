# Sequence-Agnostic Model with Cross-Sequence Distillation for Localization of Seizure Onset Zone

Official PyTorch implementation of **"Non-invasive Localization of Seizure Onset Zone using
Clinically Acquired MRI in Children with Drug-Resistant Epilepsy: a Sequence-Agnostic Model with
Cross-Sequence Distillation"**, submitted to *Medical Image Analysis* (under review).

A MAG-MS network localizes the seizure onset zone from features derived from up to five MRI
sequences — T1w, T2w, FLAIR, DWI and DWIC (connectome). One binary classifier is trained per brain
node, over 711 nodes. The model is **sequence-agnostic**: trained once with all five sequences and
cross-sequence distillation, it can then be evaluated on any subset of them, because absent encoders
are dropped from the network's `target_dict` rather than fed all-zero inputs.

## Requirements

Python >= 3.9. Install [PyTorch](https://pytorch.org) >= 2.0.1 for your CUDA version first, then:

```bash
pip install "torchmanager>=1.2.0rc8" --pre
pip install monai numpy scipy pandas openpyxl scikit-learn imbalanced-learn tqdm
```

`magnet` is not on PyPI — clone the [multimodality](https://github.com/kisonho/multimodality/tree/feature-0201)
`feature-0201` branch and put it on your `PYTHONPATH`. The exact versions the reported results were
produced with are pinned in `pyproject.toml`.

## Repository layout

| Path | Contents |
|---|---|
| `ezpred/` | the model — encoders, fusion, classifier head, metrics, configs |
| `data/` | dataset loader, SMOTE augmentation, external-cohort staging |
| `train_left.py`, `train_right.py` | training, one hemisphere each |
| `eval_left.py`, `eval_right.py` | evaluation on the additional cohort |
| `eval_bonn_left.py`, `eval_bonn_right.py` | evaluation on the Bonn cohort |
| `*_ALL_*.sh` | per-GPU node shards |
| `run_all_*.sh` | launch every shard detached, surviving SSH disconnect |
| `combine_*.py`, `make_*_table.py` | aggregate per-node results into tables |
| `results/` | **the reported numbers** |
| `sota_node_level/` | baseline and state-of-the-art comparisons |

Data and experiment roots are absolute constants at the top of each script — change them to match
your layout.

## Data

A private clinical dataset of MRI-derived features: one 1899-dimensional vector per node, laid out as

```
T1 [0:300]   T2 [300:500]   FLAIR [500:700]   DWI [700:1400]   DWIC [1400:1899]
```

Contact the authors for access to similar data. Three cohorts are reported:

| Cohort | Subjects | Sequences | Selected by | Step |
|---|---|---|---|---|
| Original held-out validation | 10 | all five | `--cohort orig` | 3 |
| Additional cohort | 30 (17 with DWIC, 13 without) | all five / no DWIC | `--cohort add` (default) | 4 |
| Bonn — OpenNeuro [ds004199](https://doi.org/10.18112/openneuro.ds004199.v1.0.5) | 85 | T1 + FLAIR only | `eval_bonn_*.py` | 5 |

All three are held out of training: the model is always fitted on the SMOTE-augmented data from the
58 training patients.

## Running the pipeline

Every training and evaluation step has two forms: a `run_all_*.sh` launcher that shards all 711
nodes across the GPUs and runs detached, and the underlying Python script, which handles **one node
at a time** and is the thing to reach for when debugging or running a subset.

The examples below use these two roots — set them to match your layout:

```bash
DATA=/path/to/Data/All_Hemispheres
REPO=/path/to/magmsforEZprediction
```

### 1. Prepare the data

SMOTE-balance the training set, then stage the two external cohorts:

```bash
python data/augment_single_node_left_hemis_part2.py
python data/augment_single_node_right_hemis_part2.py

python data/prepare_add_cohort.py     # writes Node_{N}/Add_Val_Data_{17_withDWIC,13_noDWIC}/
python data/prepare_bonn_cohort.py    # writes Node_{N}/Bonn_Val_Data_85/, recovers + verifies labels
```

### 2. Train

Training always uses the SMOTE-augmented data from the 58 training patients, with all five sequences
and cross-sequence distillation, for 3 trials per node. `--cohort` chooses the validation set that
selects each trial's best checkpoint, and therefore where the checkpoints and results are written.

All nodes, sharded and detached:

```bash
./run_all_add_training.sh                  # 10 shards across 4 GPUs
grep -c "saved at" logs_add_root/*.log     # progress
```

One node directly — this is what each shard runs in a loop:

```bash
# --cohort add (default): checkpoints selected on the 17 complete subjects of the additional cohort
python train_left.py $DATA/Left_Hemis/Part_2/ $REPO/trained_models/magms_last_left.model \
    -b 4 -lr 1e-2 --num_mod 5 -e 30 --train_mod ALL \
    --node_num 6 -exp exp_node6/AddCohort/magms \
    --replace_experiment --show_verbose --device cuda:0

# --cohort orig: checkpoints selected on the original 10 held-out patients
python train_left.py $DATA/Left_Hemis/Part_2/ $REPO/trained_models/magms_last_left.model \
    -b 4 -lr 1e-2 --num_mod 5 -e 30 --train_mod ALL --cohort orig \
    --node_num 6 -exp exp_node6/Part_2/magms \
    --replace_experiment --show_verbose --device cuda:0
```

Use `train_right.py` for nodes above 479. Checkpoints land in
`experiments/exp_node{N}/{AddCohort,Part_2}/magms_trial{i}.exp/`.

### 3. Evaluate on the original 10 held-out patients

Scores the `Part_2` checkpoints from step 2 on all 31 sequence combinations, averaged over the 3
trials. Run over every node, then concatenate:

```bash
for n in 6 11 12 14 18 ...; do
    python eval_left.py $DATA/Left_Hemis/Part_2/ $REPO/experiments/ \
        -b 4 --node_num $n --cohort orig -exp eval_orig_left.exp \
        --replace_experiment --show_verbose --device cuda:0
done

python combine_node_excel_sheet_results_eval_left.py
python combine_node_excel_sheet_results_eval_right.py
```

Each node writes one column of 31 balanced accuracies to
`Node_{N}_Results/Eval_Results/results_val_ALL_modalities_Part_2.xlsx`, which is the layout the two
combiners concatenate.

### 4. Evaluate on the additional cohort

All 31 combinations for the 17 subjects with every sequence, the 15 DWIC-free ones for the other 13,
keeping each trial as its own column:

```bash
./run_all_add_evaluation.sh                # 8 shards across 4 GPUs

# or one node at a time
python eval_left.py $DATA/Left_Hemis/Part_2/ $REPO/experiments/ \
    -b 4 --node_num 6 -exp eval_add_left.exp \
    --replace_experiment --show_verbose --device cuda:0

python combine_add_cohort_results.py --hemisphere left  --group 17
python combine_add_cohort_results.py --hemisphere left  --group 13
python combine_add_cohort_results.py --hemisphere right --group 17
python combine_add_cohort_results.py --hemisphere right --group 13
python combine_add_cohort_training_results.py
python make_addcohort_combination_table.py
```

### 5. Evaluate on the Bonn cohort

T1, FLAIR and T1+FLAIR. No training — the additional cohort's checkpoints are reused unchanged, so
this is a fully external test:

```bash
./run_all_bonn_evaluation.sh
setsid nohup ./finish_bonn_evaluation.sh > logs_bonn_eval/finish.log 2>&1 &

# or one node at a time
python eval_bonn_left.py $DATA/Left_Hemis/Part_2/ $REPO/experiments/ \
    -b 4 --node_num 6 -exp eval_bonn_left.exp \
    --replace_experiment --show_verbose --device cuda:0
```

`finish_bonn_evaluation.sh` waits for the shards, then combines both hemispheres and rebuilds the
tables. It skips the final table if any node is missing, so no partial average is ever reported.

### Other configurations

* **No cross-sequence distillation** — uncomment the `NO Distillation` loss block in `train_*.py`.
* **Subgroup analysis** (MR-/MR+/SF/SR) — build the splits with
  `data/{left,right}_hemis_part2_subgroups.py`.

## Results

All reported numbers are committed under [`results/`](results/).

| File | Contents |
|---|---|
| **[`Combined_AddCohort_BonnCohort_table.xlsx`](results/Combined_AddCohort_BonnCohort_table.xlsx)** | the main table — both external cohorts side by side, rows 1-31 |
| [`COHORTS.md`](results/COHORTS.md) | cohort composition and how the means are computed |
| [`addcohort/`](results/addcohort/), [`bonncohort/`](results/bonncohort/) | per-node results, one README each |

Two things to know when reading the main table:

* Use the **`nodes_with_EZ`** sheet for model quality. At a node whose validation split contains no
  EZ subject, balanced accuracy collapses to plain non-EZ accuracy, which inflates the `all_nodes`
  sheet. [`COHORTS.md`](results/COHORTS.md) explains this in full.
* The Bonn columns are populated only at rows 4, 16 and 20 and are `NA` elsewhere — those subjects
  have no T2/DWI/DWIC acquisition, and zero-filling a missing encoder shifts the logits rather than
  removing the branch. [`bonncohort/README.md`](results/bonncohort/README.md) quantifies it.

## Citation

The paper is under review; the full reference will be added on acceptance. Until then, please cite
this repository and the preprint if you use the code.

```bibtex
@misc{magmsforEZprediction,
  title  = {Non-invasive Localization of Seizure Onset Zone using Clinically Acquired MRI in
            Children with Drug-Resistant Epilepsy: a Sequence-Agnostic Model with
            Cross-Sequence Distillation},
  note   = {Under review at Medical Image Analysis},
  url    = {https://github.com/soumbane/magmsforEZprediction}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
