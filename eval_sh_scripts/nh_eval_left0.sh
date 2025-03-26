#!/bin/bash

# Create logs directory
mkdir -p logs_left0

# Define the list of node_num values
node_nums=(6 11 12 14 18 19 20 33 34 35 36 39 41 42 43 44 45 46 47 48 49 51 52 53 54 55 56 57 58 59 60 61 62 63 64 66 68 79 80 81 84 85 86 87 88 90 91 93 94 95 96 97 98 101 102 103 104 108 109 111 120 121 122 123 124 125 126 127 128 129 130 131 140 144 145 147 148 150 151 155 156 158 159 160 163 164 165 166 169 175 176 177 192 193 194 195 197 198 199 200 202 204) 

# Log file
LOG_FILE="logs_left0/evaluation_progress.log"

# Create a temporary script file for the loop
cat > temp_eval_script.sh << 'EOF'
#!/bin/bash
LOG_FILE="logs_left0/evaluation_progress.log"
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
    echo "$(date) - Completed: ${node_num}" >> logs_left0/completed_nodes.txt
done

echo "[$(date)] All evaluations completed!" >> $LOG_FILE
EOF

# Make the temp script executable
chmod +x temp_eval_script.sh

# Run the temp script with nohup, passing the array elements as arguments
nohup ./temp_eval_script.sh "${node_nums[@]}" > /dev/null 2>&1 &

# Save the process ID
PID=$!
echo $PID > logs_left0/evaluation_pid.txt
echo "Evaluation process started with PID: $PID"
echo "Progress is being logged to: $LOG_FILE"
echo "You can safely disconnect. To check status later, use: tail -f $LOG_FILE"