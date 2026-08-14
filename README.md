# Sequence-Agnostic Model with Cross-Sequence Distillation for Localization of Seizure Onset Zone

This is the official PyTorch implementation of the paper **Non-invasive Localization of Seizure Onset Zone using Clinically Acquired MRI in Children with Drug-Resistant Epilepsy: a Sequence-Agnostic Model with Cross-Sequence Distillation** that was submitted to **Medical Image Analysis** journal and is currently under review.

## Requirements
* Python >= 3.9
* [PyTorch](https://pytorch.org) >= 1.12.0
* [torchmanager](https://github.com/kisonho/torchmanager) >= 1.1.0
* [Monai](https://monai.io) >= 1.1
* [Multimodality](https://github.com/kisonho/multimodality/tree/feature-0201)
* Download the script from the above Multimodality branch in your code directory - since it will be used in the main code.

## Get Started
The following steps are required to replicate our work:

1. Dataset
   * This project uses a private clinical dataset of MRI scans from children with drug-resistant epilepsy.
   * The dataset basically consists of features derived from various MRI sequences such as T1w, T2w, FLAIR, DWI and DWIC, for every node of the brain of children with Drug-Resistant Epilepsy.
   * For access to similar datasets for research purposes, please contact the authors.

2. Data preprocessing
   * The SMOTE augmentation is performed by `augment_single_node_left_hemis_part2.py` and `augment_single_node_right_hemis_part2.py` for the nodes of the left and right hemispheres respectively.
   * The data loading steps are provided in the `data/dataset_ez.py` file. This file uses the SMOTE augmented data.
   * It performs all the necessary preprocessing steps to prepare the MRI data for training and validation.

3. Additional validation cohort
   * An independent cohort of 30 subjects covering the same 711 nodes is staged by `data/prepare_add_cohort.py`.
   * 13 of the 30 subjects were scanned without DWIC (connectome) data, so the cohort is split into two groups, written to `Node_{N}/Add_Val_Data_17_withDWIC/` and `Node_{N}/Add_Val_Data_13_noDWIC/`:
     * the 17 subjects with all five sequences, evaluated on all 31 modality combinations;
     * the 13 subjects without DWIC, evaluated on the 15 DWIC-free combinations, with DWIC dropped from the model's `target_dict` rather than supplied as an all-zero input.
   * Unlike the original cohort, the labels of this cohort are already binary, so no IZ patients are removed and no label remapping is applied.
   ```
   python data/prepare_add_cohort.py
   ```

## Training
1. Refer to the training configuration in `train_right.py` for the default hyper-parameters to train the models for the right hemisphere nodes:
```
# To train the model on nodes of the right hemisphere
./train_ALL_right.sh
```

2. Refer to the training configuration in `train_left.py` for the default hyper-parameters to train the models for the left hemisphere nodes:
```
# To train the model on nodes of the left hemisphere
./train_ALL_left.sh
```

3. Please note that depending on the node numbers, we may need to change the shell script name. This is because the training on all the nodes is sequential and we used multiple shell scripts to obtain faster results.

4. The training process can be customized by modifying the following parameters in the shell scripts:
   * Batch size
   * Learning rate
   * Number of epochs
   * Number of Modality or sequence selection
   * Device selection

## Testing
1. The pre-trained models are stored in the `experiments/` folder. 
   * The pre-trained models are not included in this GitHub repository due to space limitations.

2. Refer to the `eval_right.py` for the default settings to test the models for the right hemisphere nodes:
```
# To evaluate all nodes of the right hemisphere
./eval_ALL_right.sh
```

3. Refer to the `eval_left.py` for the default settings to test the models for the left hemisphere nodes:
```
# To evaluate all nodes of the left hemisphere
./eval_ALL_left.sh
```

4. Please note that depending on the node numbers, we may need to change the shell script name. This is because the evaluation on all the nodes is sequential and we used multiple shell scripts to obtain faster results.

5. The testing process can be customized by modifying the parameters in the shell scripts or directly in the `eval_left.py` and `eval_right.py` files.

## Additional validation cohort
The model is trained on the same SMOTE-augmented training data (58 patients) with all five sequences and cross-sequence distillation, and the best checkpoint of each trial is selected on the 17 subjects of the additional cohort that have all five sequences. That checkpoint is then evaluated on both subject groups.

1. Train all nodes, writing the checkpoints under `experiments/exp_node{N}/AddCohort/`:
```
# All nodes sequentially
./train_ALL_add_left.sh
./train_ALL_add_right.sh

# Or in parallel shards across GPUs
./train_sh_scripts/nh_train_add_left0.sh
./train_sh_scripts/nh_train_add_right0.sh
```

2. Evaluate all nodes on both subject groups. Each of the 3 training trials is written as its own column, so the maximum or the mean over trials can be chosen afterwards:
```
./eval_ALL_add_left.sh
./eval_ALL_add_right.sh
```
This writes `results_add17_ALL31_trials.xlsx` (31 combinations) and `results_add13_DWICfree15_trials.xlsx` (15 combinations) per node.

3. Combine the per-node results:
```
python combine_add_cohort_results.py --hemisphere left --group 17
python combine_add_cohort_results.py --hemisphere left --group 13
python combine_add_cohort_results.py --hemisphere right --group 17
python combine_add_cohort_results.py --hemisphere right --group 13
```
Each combined file has a `per_trial` sheet along with `max_over_trials` and `mean_over_trials`.

## Final Results
1. After evaluating the results, combine the results for all nodes of the left hemisphere with:
```
python combine_node_excel_sheet_results_eval_left.py
```

2. After evaluating the results, combine the results for all nodes of the right hemisphere with:
```
python combine_node_excel_sheet_results_eval_right.py
```

The above will provide excel files with the results for all nodes.

## Citation

