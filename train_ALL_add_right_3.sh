#!/bin/bash

# Shard right3 of the right hemisphere (64 nodes) on cuda:1
# Trains on the original SMOTE-augmented training data (58 patients) with all 5 sequences and
# cross-sequence distillation, selecting the best checkpoint on the 17 subjects of the additional
# cohort that have all five sequences. Runs sequentially in the foreground; for a detached run use
# train_sh_scripts/nh_train_add_right3.sh

node_nums=(599 600 601 602 603 604 605 606 607 608 609 610 612 613 614 615 616 617 618 619 620 621 622 623 624 625 627 628 629 630 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 655 656 657 658 659 660 661 662 663 664 665 666 668)

# Loop through each node_num
for node_num in "${node_nums[@]}"; do
    # Define experiment file path
    exp_file="exp_node${node_num}/AddCohort/magms"

    # Run the training script with specified arguments
    python train_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_righthemis.model -b 4 -lr 1e-2 --num_mod 5 --node_num ${node_num} --train_mod ALL -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:1

    # Record the experiment file
    echo "Experiment for node_num ${node_num} saved at: ${exp_file}"

done
