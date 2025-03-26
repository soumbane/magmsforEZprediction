#!/bin/bash

# Create logs directory
mkdir -p logs_right1

# Define the list of node_num values
node_nums=(610 612 613 614 615 616 617 618 619 620 621 622 623 624 625 627 628 629 630 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 648 649 650 651 652 655 656 657 658 659 660 661 662 663 664 665 666 668 669 670 671 672 673 674 675 676 677 678 681 683 685 686 690 691 692 693 694 695 696 697 698 699 700 701 702 703 704 705 706 707 708 709 710 711)  

# Log file
LOG_FILE="logs_right1/evaluation_progress.log"

# Create a temporary script file for the loop
cat > temp_eval_script.sh << 'EOF'
#!/bin/bash
LOG_FILE="logs_right1/evaluation_progress.log"
node_nums=($@)  # Get the node numbers from command line

# Start time
echo "Starting evaluation at $(date)" > $LOG_FILE
echo "Total nodes to process: ${#node_nums[@]}" >> $LOG_FILE

# Loop through each node_num
for node_num in "${node_nums[@]}"
do
    echo "[$(date)] Starting evaluation for node_num ${node_num}" >> $LOG_FILE
    
    # Run the evaluation script with specified arguments
    python eval_right.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Right_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:1 >> $LOG_FILE 2>&1
    
    # Record the experiment completion
    echo "[$(date)] Done Evaluating ALL modalities for node_num ${node_num}" >> $LOG_FILE
    
    # Save progress information
    echo "$(date) - Completed: ${node_num}" >> logs_right1/completed_nodes.txt
done

echo "[$(date)] All evaluations completed!" >> $LOG_FILE
EOF

# Make the temp script executable
chmod +x temp_eval_script.sh

# Run the temp script with nohup, passing the array elements as arguments
nohup ./temp_eval_script.sh "${node_nums[@]}" > /dev/null 2>&1 &

# Save the process ID
PID=$!
echo $PID > logs_right1/evaluation_pid.txt
echo "Evaluation process started with PID: $PID"
echo "Progress is being logged to: $LOG_FILE"
echo "You can safely disconnect. To check status later, use: tail -f $LOG_FILE"