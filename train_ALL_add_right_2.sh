#!/bin/bash

# Shard right2 of the right hemisphere (78 nodes) on cuda:1
# Trains on the original SMOTE-augmented training data (58 patients) with all 5 sequences and
# cross-sequence distillation, selecting the best checkpoint on the 17 subjects of the additional
# cohort that have all five sequences. Runs sequentially in the foreground; use
# ./run_all_add_training.sh to launch every shard detached so the run survives closing SSH.

node_nums=(835 836 837 838 839 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 877 878 879 880 881 882 883 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915)

# Loop through each node_num
for node_num in "${node_nums[@]}"; do
    # Define experiment file path
    exp_file="exp_node${node_num}/AddCohort/magms"

    # Run the training script with specified arguments
    python train_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_righthemis.model -b 4 -lr 1e-2 --num_mod 5 --node_num ${node_num} --train_mod ALL -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:1

    # Record the experiment file
    echo "Experiment for node_num ${node_num} saved at: ${exp_file}"

done
