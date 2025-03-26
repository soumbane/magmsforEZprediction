#!/bin/bash

# Define the list of node_num values
# node_nums=(948)  # part 2 - test nodes
# trial_nums=(4)  # part 2 - test trials for each node

# node_nums=(910)  # part 2 - test nodes
# trial_nums=(4)  # part 2 - test trials for each node

# node_nums=(919)  # part 2 - test nodes
# trial_nums=(3)  # part 2 - test trials for each node

# node_nums=(923)  # part 2 - test nodes
# trial_nums=(4)  # part 2 - test trials for each node

# node_nums=(911)  # part 2 - test nodes
# trial_nums=(0 2)  # part 2 - test trials for each node

node_nums=(504 506 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 524 525 526 529 530 534 535 536 537 538 539 540 541 542 543 546 547 548 549 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 581 582 584 585 586 587 588 589 590 591 592 593 594 595 596 598)  # part 2 - right hemis except temporal lobe

# node_nums=(888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 960 961 962 963 964 965 968 969 970 971 973 974 975 976 977 978 979 980 981 982 983)  # part 2 - temporal lobe only

# Loop through each node_num
for node_num in "${node_nums[@]}"; do
    
    # Define experiment file path
    exp_file="exp_node${node_num}/NO_Distillation/magms"
    # exp_file="exp_node${node_num}/Part_2/magms"

    # Run the training script with specified arguments
    python train_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_righthemis.model -b 4 -lr 1e-2 --num_mod 3 --node_num ${node_num} --train_mod T1-T2-FLAIR -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:0

    # Record the experiment file
    echo "Experiment for node_num ${node_num} saved at: ${exp_file}"
    
done

# # For performing the different trials for each node
# # Loop through each node_num
# for node_num in "${node_nums[@]}"; do
#     for trial_num in "${trial_nums[@]}"; do
#         # Define experiment file path
#         # exp_file="exp_node${node_num}/NO_Distillation/magms"
#         exp_file="exp_node${node_num}/Part_2/magms"

#         # Run the training script with specified arguments
#         python train_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_righthemis.model -b 4 -lr 1e-2 --num_mod 5 --node_num ${node_num} --trial_num ${trial_num} --train_mod ALL -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:3

#         # Record the experiment file
#         echo "Experiment for node_num ${node_num} saved at: ${exp_file}"
#     done
    
# done

