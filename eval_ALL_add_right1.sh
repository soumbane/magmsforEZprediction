#!/bin/bash

# Shard right1 of the right hemisphere (89 nodes) on cuda:2
# Evaluates the AddCohort checkpoints on both groups of the additional cohort:
#   - the 17 subjects with all five sequences, on all 31 modality combinations
#   - the 13 subjects without DWIC, on the 15 DWIC-free combinations
# Each of the 3 training trials is written as its own column. For a detached run use
# eval_sh_scripts/nh_eval_add_right1.sh

node_nums=(610 612 613 614 615 616 617 618 619 620 621 622 623 624 625 627 628 629 630 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 655 656 657 658 659 660 661 662 663 664 665 666 668 669 670 671 672 673 674 675 676 677 678 681 683 685 686 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711)

# Loop through each node_num
for node_num in "${node_nums[@]}"
do
    # Run the evaluation script with specified arguments
    python eval_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:2

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done
