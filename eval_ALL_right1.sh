#!/bin/bash

# Define the list of node_num values
node_nums=(646 647 648 649 650 651 652 655 656 657 658 659 660 661 662 663 664 665 666 668 669 670 671 672 673 674 675 676 677 678 681 683 685 686 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 730 731 732 733 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 756 757 758 759 760 761 762 763 764 765 766 767 770 771 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 795 796 797 798 799 800 801 802 803 804 805 806 808 809 810 811 812 813 816 817 818 819 820)  # part 2 - right hemis

# Loop through each node_num
for node_num in "${node_nums[@]}"
do    
    # Run the evaluation script with specified arguments
    python eval_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:1

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done

