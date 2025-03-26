#!/bin/bash

# Create logs directory
mkdir -p logs_left1

# Define the list of node_num values
node_nums=(205 211 213 214 216 217 220 221 222 224 225 226 227 228 229 230 231 232 233 234 235 238 239 240 241 245 246 247 248 251 252 253 256 257 260 261 275 290 291 292 294 295 296 297 298 299 301 302 303 304 305 306 316 320 321 322 326 331 332 334 335 336 337 338 339 340 343 346 349 352 353 354 355 356 357 359 360 361 362 363 364 365 366 367 368 369 370 371 372) 

# Log file
LOG_FILE="logs_left1/evaluation_progress.log"

# Create a temporary script file for the loop
cat > temp_eval_script.sh << 'EOF'
#!/bin/bash
LOG_FILE="logs_left1/evaluation_progress.log"
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
    echo "$(date) - Completed: ${node_num}" >> logs_left1/completed_nodes.txt
done

echo "[$(date)] All evaluations completed!" >> $LOG_FILE
EOF

# Make the temp script executable
chmod +x temp_eval_script.sh

# Run the temp script with nohup, passing the array elements as arguments
nohup ./temp_eval_script.sh "${node_nums[@]}" > /dev/null 2>&1 &

# Save the process ID
PID=$!
echo $PID > logs_left1/evaluation_pid.txt
echo "Evaluation process started with PID: $PID"
echo "Progress is being logged to: $LOG_FILE"
echo "You can safely disconnect. To check status later, use: tail -f $LOG_FILE"