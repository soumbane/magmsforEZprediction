#!/bin/bash

# Create logs directory
mkdir -p logs_left2

# Define the list of node_num values
node_nums=(374 375 376 377 378 381 382 383 384 386 387 388 389 390 391 394 395 396 397 398 399 400 401 402 403 404 405 406 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 426 427 428 429 430 431 432 433 435 436 437 438 439 440 441 442 443 444 445 446 447 448 450 451 452 453 454 455 456 458 459 460 461 462 463 464 465 466 467 469 470 471 472 473 474 475 476 477 478 479) 

# Log file
LOG_FILE="logs_left2/evaluation_progress.log"

# Create a temporary script file for the loop
cat > temp_eval_script.sh << 'EOF'
#!/bin/bash
LOG_FILE="logs_left2/evaluation_progress.log"
node_nums=($@)  # Get the node numbers from command line

# Start time
echo "Starting evaluation at $(date)" > $LOG_FILE
echo "Total nodes to process: ${#node_nums[@]}" >> $LOG_FILE

# Loop through each node_num
for node_num in "${node_nums[@]}"
do
    echo "[$(date)] Starting evaluation for node_num ${node_num}" >> $LOG_FILE
    
    # Run the evaluation script with specified arguments
    python eval_left.py /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/ /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/experiments/ -b 4 --node_num ${node_num} --replace_experiment --show_verbose --device cuda:0 >> $LOG_FILE 2>&1
    
    # Record the experiment completion
    echo "[$(date)] Done Evaluating ALL modalities for node_num ${node_num}" >> $LOG_FILE
    
    # Save progress information
    echo "$(date) - Completed: ${node_num}" >> logs_left2/completed_nodes.txt
done

echo "[$(date)] All evaluations completed!" >> $LOG_FILE
EOF

# Make the temp script executable
chmod +x temp_eval_script.sh

# Run the temp script with nohup, passing the array elements as arguments
nohup ./temp_eval_script.sh "${node_nums[@]}" > /dev/null 2>&1 &

# Save the process ID
PID=$!
echo $PID > logs_left2/evaluation_pid.txt
echo "Evaluation process started with PID: $PID"
echo "Progress is being logged to: $LOG_FILE"
echo "You can safely disconnect. To check status later, use: tail -f $LOG_FILE"