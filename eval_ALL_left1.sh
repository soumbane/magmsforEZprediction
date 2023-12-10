#!/bin/bash

# Define the list of node_num values
node_nums=(421 422 423 424 426 429 431 432 433 436 439 440 441 442 446 447 448 451 454 458 459 461 463 465 467 469 477 479)  # part 2 - left hemis

# Loop through each node_num
for node_num in "${node_nums[@]}"
do    
    # Run the evaluation script with specified arguments
    python eval_left.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/SubGroups/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:0

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done