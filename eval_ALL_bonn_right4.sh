#!/bin/bash

# Shard right4 of the right hemisphere (69 nodes) on cuda:0
# For every node, evaluates the 3 AddCohort trial checkpoints on the 85 Bonn subjects
# (OpenNeuro ds004199), which have T1 and FLAIR only, across the 3 non-empty subsets of
# {T1, FLAIR}. The absent sequences are dropped from the model's target_dict rather than
# fed as all-zero branches. No training happens here.
# Each trial is written as its own column, into
#   BonnCohort/Right_Hemis/Node_<N>_Results/Eval_Results/results_bonn_T1FLAIR3_trials.xlsx
#
# Runs sequentially in the foreground; use ./run_all_bonn_evaluation.sh to launch every shard
# detached so the run survives closing the SSH session.
#
# -exp is unique per shard: eval recreates experiments/<exp> on every node, so shards sharing one
# experiment name would delete each other's log directory mid-run.

node_nums=(910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 960 961 962 963 964 965 968 969 970 971 973 974 975 976 977 978 979 980 981 982 983)

# Loop through each node_num
for node_num in "${node_nums[@]}"
do
    # Run the evaluation script with specified arguments
    python eval_bonn_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} -exp eval_bonn_right4.exp --replace_experiment --show_verbose --device cuda:0

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done
