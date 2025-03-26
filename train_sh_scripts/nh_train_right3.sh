#!/bin/bash

# Create logs directory
mkdir -p logs_right3

# Log file
LOG_FILE="logs_right3/training_progress.log"

# Create a temporary script file for the loop
cat > temp_training_script.sh << 'EOF'
#!/bin/bash
LOG_FILE="logs_right3/training_progress.log"
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
    echo "$(date) - Completed: ${node_num}" >> logs_right3/completed_nodes.txt
done

echo "[$(date)] All training completed!" >> $LOG_FILE
EOF

# Make the temp script executable
chmod +x temp_training_script.sh

# Define the node numbers array
node_nums=(599 600 601 602 603 604 605 606 607 608 609 610 612 613 614 615 616 617 618 619 620 621 622 623 624 625 627 628 629 630 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 655 656 657 658 659 660 661 662 663 664 665 666 668)

# Run the temp script with nohup, passing the array elements as arguments
nohup ./temp_training_script.sh "${node_nums[@]}" > /dev/null 2>&1 &

# Save the process ID
PID=$!
echo $PID > logs_right3/training_pid.txt
echo "Training process started with PID: $PID"
echo "Progress is being logged to: $LOG_FILE"
echo "You can safely disconnect. To check status later, use: tail -f $LOG_FILE"