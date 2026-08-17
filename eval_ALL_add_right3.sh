#!/bin/bash

# Shard right3 of the right hemisphere (89 nodes) on cuda:3
# For every node, evaluates the 3 AddCohort trial checkpoints on both subject groups:
#   - the 17 subjects with all five sequences, on ALL 31 modality combinations
#   - the 13 subjects without DWIC, on the 15 DWIC-free combinations
# Each trial is written as its own column, into
#   AddCohort/Right_Hemis/Node_<N>_Results/Eval_Results/results_add17_ALL31_trials.xlsx
#   AddCohort/Right_Hemis/Node_<N>_Results/Eval_Results/results_add13_DWICfree15_trials.xlsx
#
# Runs sequentially in the foreground; use ./run_all_add_evaluation.sh to launch every shard
# detached so the run survives closing the SSH session.
#
# -exp is unique per shard: eval recreates experiments/<exp> on every node, so shards sharing one
# experiment name would delete each other's log directory mid-run.

node_nums=(817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 834 835 836 837 838 839 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 877 878 879 880 881 882 883 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909)

# Loop through each node_num
for node_num in "${node_nums[@]}"
do
    # Run the evaluation script with specified arguments
    python eval_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} -exp eval_add_right3.exp --replace_experiment --show_verbose --device cuda:3

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done
