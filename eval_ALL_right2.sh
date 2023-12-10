#!/bin/bash

# Define the list of node_num values
node_nums=(671 672 673 674 675 676 677 678 681 683 685 686 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 730 731 732 733 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750)  # part 2 - right hemis

# Loop through each node_num
for node_num in "${node_nums[@]}"
do    
    # Run the evaluation script with specified arguments
    python eval_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/SubGroups/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:2

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done

