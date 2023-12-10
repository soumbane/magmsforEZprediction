#!/bin/bash

# Define the list of node_num values
node_nums=(916 919 921 923 924 926 927 928 929 930 931 932 935 938 939 940 943)  # part 2 - right hemis

# Loop through each node_num
for node_num in "${node_nums[@]}"
do    
    # Run the evaluation script with specified arguments
    python eval_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/SubGroups/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:1

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done

