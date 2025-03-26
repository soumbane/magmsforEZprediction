#!/bin/bash

# Create logs directory
mkdir -p logs_right2

# Log file
LOG_FILE="logs_right2/training_progress.log"

# Create a temporary script file for the loop
cat > temp_training_script.sh << 'EOF'
#!/bin/bash
LOG_FILE="logs_right2/training_progress.log"
node_nums=($@)  # Get the node numbers from command line

# Start time
echo "Starting training at $(date)" > $LOG_FILE
echo "Total nodes to process: ${#node_nums[@]}" >> $LOG_FILE

# Loop through each node_num
for node_num in "${node_nums[@]}"
do
    # Define experiment file path
    exp_file="exp_node${node_num}/NO_Distillation/magms"
    
    echo "[$(date)] Starting training for node_num ${node_num}" >> $LOG_FILE
    
    # Run the training script with specified arguments
    python train_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_righthemis.model -b 4 -lr 1e-2 --num_mod 3 --node_num ${node_num} --train_mod T1-T2-FLAIR -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:1 >> $LOG_FILE 2>&1
    
    # Record the experiment completion
    echo "[$(date)] Done training for node_num ${node_num}, saved at: ${exp_file}" >> $LOG_FILE
    
    # Save progress information
    echo "$(date) - Completed: ${node_num}" >> logs_right2/completed_nodes.txt
done

echo "[$(date)] All training completed!" >> $LOG_FILE
EOF

# Make the temp script executable
chmod +x temp_training_script.sh

# Define the node numbers array
node_nums=(835 836 837 838 839 841 842 843 844 845 846 847 848 849 850 851 852 853 854 855 856 857 858 859 860 861 862 863 864 865 866 867 868 869 870 871 872 873 874 875 877 878 879 880 881 882 883 885 886 887 888 889 890 891 892 893 894 895 896 897 898 899 900 901 902 903 904 905 906 907 908 909 910 911 912 913 914 915)

# Run the temp script with nohup, passing the array elements as arguments
nohup ./temp_training_script.sh "${node_nums[@]}" > /dev/null 2>&1 &

# Save the process ID
PID=$!
echo $PID > logs_right2/training_pid.txt
echo "Training process started with PID: $PID"
echo "Progress is being logged to: $LOG_FILE"
echo "You can safely disconnect. To check status later, use: tail -f $LOG_FILE"