#!/bin/bash

# Shard right5 of the right hemisphere (63 nodes) on cuda:3
# Trains on the original SMOTE-augmented training data (58 patients) with all 5 sequences and
# cross-sequence distillation, selecting the best checkpoint on the 17 subjects of the additional
# cohort that have all five sequences. Runs sequentially in the foreground; use
# ./run_all_add_training.sh to launch every shard detached so the run survives closing SSH.

node_nums=(916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 960 961 962 963 964 965 968 969 970 971 973 974 975 976 977 978 979 980 981 982 983)

# Loop through each node_num
for node_num in "${node_nums[@]}"; do
    # Define experiment file path
    exp_file="exp_node${node_num}/AddCohort/magms"

    # Run the training script with specified arguments
    python train_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_righthemis.model -b 4 -lr 1e-2 --num_mod 5 --node_num ${node_num} --train_mod ALL -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:3

    # Record the experiment file
    echo "Experiment for node_num ${node_num} saved at: ${exp_file}"

done
