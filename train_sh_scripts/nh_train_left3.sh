#!/bin/bash

# Create logs directory
mkdir -p logs_left3

# Log file
LOG_FILE="logs_left3/training_progress.log"

# Create a temporary script file for the loop
cat > temp_training_script.sh << 'EOF'
#!/bin/bash
LOG_FILE="logs_left3/training_progress.log"
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
    python train_left.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/trained_models/magms_trained_last_lefthemis.model -b 4 -lr 1e-2 --num_mod 3 --node_num ${node_num} --train_mod T1-T2-FLAIR -e 30 -exp ${exp_file} --replace_experiment --show_verbose --device cuda:1 >> $LOG_FILE 2>&1
    
    # Record the experiment completion
    echo "[$(date)] Done training for node_num ${node_num}, saved at: ${exp_file}" >> $LOG_FILE
    
    # Save progress information
    echo "$(date) - Completed: ${node_num}" >> logs_left3/completed_nodes.txt
done

echo "[$(date)] All training completed!" >> $LOG_FILE
EOF

# Make the temp script executable
chmod +x temp_training_script.sh

# Define the node numbers array
node_nums=(413 414 415 416 417 418 419 420 421 422 423 424 426 427 428 429 430 431 432 433 435 436 437 438 439 440 441 442 443 444 445 446 447 448 450 451 452 453 454 455 456 458 459 460 461 462 463 464 465 466 467 469 470 471 472 473 474 475 476 477 478 479)

# Run the temp script with nohup, passing the array elements as arguments
nohup ./temp_training_script.sh "${node_nums[@]}" > /dev/null 2>&1 &

# Save the process ID
PID=$!
echo $PID > logs_left3/training_pid.txt
echo "Training process started with PID: $PID"
echo "Progress is being logged to: $LOG_FILE"
echo "You can safely disconnect. To check status later, use: tail -f $LOG_FILE"