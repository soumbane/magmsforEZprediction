#!/bin/bash

# Create logs directory
mkdir -p logs_right1

# Log file
LOG_FILE="logs_right1/training_progress.log"

# Create a temporary script file for the loop
cat > temp_training_script.sh << 'EOF'
#!/bin/bash
LOG_FILE="logs_right1/training_progress.log"
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
    python train_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_righthemis.model -b 4 -lr 1e-2 --num_mod 3 --node_num ${node_num} --train_mod T1-T2-FLAIR -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:0 >> $LOG_FILE 2>&1
    
    # Record the experiment completion
    echo "[$(date)] Done training for node_num ${node_num}, saved at: ${exp_file}" >> $LOG_FILE
    
    # Save progress information
    echo "$(date) - Completed: ${node_num}" >> logs_right1/completed_nodes.txt
done

echo "[$(date)] All training completed!" >> $LOG_FILE
EOF

# Make the temp script executable
chmod +x temp_training_script.sh

# Define the node numbers array
node_nums=(669 670 671 672 673 674 675 676 677 678 681 683 685 686 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711 712 713 714 715 716 717 718 719 720 721 722 723 724 725 726 727 728 730 731 732 733 735 736 737 738 739 740 741 742 743 744 745 746 747 748 749 750 751 756 757 758 759)

# Run the temp script with nohup, passing the array elements as arguments
nohup ./temp_training_script.sh "${node_nums[@]}" > /dev/null 2>&1 &

# Save the process ID
PID=$!
echo $PID > logs_right1/training_pid.txt
echo "Training process started with PID: $PID"
echo "Progress is being logged to: $LOG_FILE"
echo "You can safely disconnect. To check status later, use: tail -f $LOG_FILE"