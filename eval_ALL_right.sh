#!/bin/bash

# Define the list of node_num values
node_nums=(888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 936 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 959 960 961 962 963 964 965 966 968 969 970 971 973 974 975 976 977 978 979 980 981 982 983) 

# Loop through each node_num
for node_num in "${node_nums[@]}"
do    
    # Run the evaluation script with specified arguments
    python eval_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Temporal_Lobe/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:0

    # python eval_right.py /home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/Right_Temporal_Lobe/ /home/neil/Lab_work/Jeong_Lab_Multi_Modal_MRI/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:1

    # Record the experiment file
    echo "Done Evaluating ALL modalities for node_num ${node_num}"
done

