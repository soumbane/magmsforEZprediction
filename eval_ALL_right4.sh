#!/bin/bash

# Define the list of node_num values
node_nums=(841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 877 878 879 880 881 882 883 885 886 887 888 892 897 898 902 904 910 911 912 913 914 915 917 918 920 922 925 933 934 937 941 942 944 945 946 947 948 949 950 951 952 954 955 957 963 965 979 980 981 982 983)  # part 2 - right hemis

# Loop through each node_num
for node_num in "${node_nums[@]}"
do    
    # Run the evaluation script with specified arguments
    python eval_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:2

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done

