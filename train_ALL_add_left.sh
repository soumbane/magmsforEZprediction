#!/bin/bash

# Shard left0 of the left hemisphere (82 nodes) on cuda:0
# Trains on the original SMOTE-augmented training data (58 patients) with all 5 sequences and
# cross-sequence distillation, selecting the best checkpoint on the 17 subjects of the additional
# cohort that have all five sequences. Runs sequentially in the foreground; use
# ./run_all_add_training.sh to launch every shard detached so the run survives closing SSH.

node_nums=(6 11 12 14 18 19 20 33 34 35 36 39 41 42 43 44 45 46 47 48 49 51 52 53 54 55 56 57 58 59 60 61 62 63 64 66 68 79 80 81 84 85 86 87 88 90 91 93 94 95 96 97 98 101 102 103 104 108 109 111 120 121 122 123 124 125 126 127 128 129 130 131 140 144 145 147 148 150 151 155 156 158)

# Loop through each node_num
for node_num in "${node_nums[@]}"; do
    # Define experiment file path
    exp_file="exp_node${node_num}/AddCohort/magms"

    # Run the training script with specified arguments
    python train_left.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_lefthemis.model -b 4 -lr 1e-2 --num_mod 5 --node_num ${node_num} --train_mod ALL -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:0

    # Record the experiment file
    echo "Experiment for node_num ${node_num} saved at: ${exp_file}"

done
