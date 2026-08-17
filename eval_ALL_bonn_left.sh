#!/bin/bash

# Shard left0 of the left hemisphere (102 nodes) on cuda:0
# For every node, evaluates the 3 AddCohort trial checkpoints on the 85 Bonn subjects
# (OpenNeuro ds004199), which have T1 and FLAIR only, across the 3 non-empty subsets of
# {T1, FLAIR}. The absent sequences are dropped from the model's target_dict rather than
# fed as all-zero branches. No training happens here.
# Each trial is written as its own column, into
#   BonnCohort/Left_Hemis/Node_<N>_Results/Eval_Results/results_bonn_T1FLAIR3_trials.xlsx
#
# Runs sequentially in the foreground; use ./run_all_bonn_evaluation.sh to launch every shard
# detached so the run survives closing the SSH session.
#
# -exp is unique per shard: eval recreates experiments/<exp> on every node, so shards sharing one
# experiment name would delete each other's log directory mid-run.

node_nums=(6 11 12 14 18 19 20 33 34 35 36 39 41 42 43 44 45 46 47 48 49 51 52 53 54 55 56 57 58 59 60 61 62 63 64 66 68 79 80 81 84 85 86 87 88 90 91 93 94 95 96 97 98 101 102 103 104 108 109 111 120 121 122 123 124 125 126 127 128 129 130 131 140 144 145 147 148 150 151 155 156 158 159 160 163 164 165 166 169 175 176 177 192 193 194 195 197 198 199 200 202 204)

# Loop through each node_num
for node_num in "${node_nums[@]}"
do
    # Run the evaluation script with specified arguments
    python eval_bonn_left.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} -exp eval_bonn_left0.exp --replace_experiment --show_verbose --device cuda:0

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done
