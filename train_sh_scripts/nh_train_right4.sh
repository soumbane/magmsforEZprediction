#!/bin/bash

# Create logs directory
mkdir -p logs_right4

# Log file
LOG_FILE="logs_right4/training_progress.log"

# Create a temporary script file for the loop
cat > temp_training_script.sh << 'EOF'
#!/bin/bash
LOG_FILE="logs_right4/training_progress.log"
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
    python train_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_righthemis.model -b 4 -lr 1e-2 --num_mod 3 --node_num ${node_num} --train_mod T1-T2-FLAIR -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:2 >> $LOG_FILE 2>&1
    
    # Record the experiment completion
    echo "[$(date)] Done training for node_num ${node_num}, saved at: ${exp_file}" >> $LOG_FILE
    
    # Save progress information
    echo "$(date) - Completed: ${node_num}" >> logs_right4/completed_nodes.txt
done

echo "[$(date)] All training completed!" >> $LOG_FILE
EOF

# Make the temp script executable
chmod +x temp_training_script.sh

# Define the node numbers array
node_nums=(760 761 762 763 764 765 766 767 770 771 776 777 778 779 780 781 782 783 784 785 786 787 788 789 790 791 792 793 795 796 797 798 799 800 801 802 803 804 805 806 808 809 810 811 812 813 816 817 818 819 820 821 822 823 824 825 826 827 828 829 830 831 832 834)

# Run the temp script with nohup, passing the array elements as arguments
nohup ./temp_training_script.sh "${node_nums[@]}" > /dev/null 2>&1 &

# Save the process ID
PID=$!
echo $PID > logs_right4/training_pid.txt
echo "Training process started with PID: $PID"
echo "Progress is being logged to: $LOG_FILE"
echo "You can safely disconnect. To check status later, use: tail -f $LOG_FILE"