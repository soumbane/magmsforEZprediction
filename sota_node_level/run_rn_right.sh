#!/bin/bash

# Navigate to the script directory
cd /media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Models/magmsforEZprediction/sota_node_level

# # Create a logs directory if it doesn't exist
mkdir -p logs_right

# Initialize conda (adjust this path if your conda installation is different)
source /home/user1/anaconda3/etc/profile.d/conda.sh

# Activate the conda environment
conda activate magms_ez

# Get the name of the Python script (adjust if different)
SCRIPT_NAME="relational_reasoning_all_nodes_all_trials_right.py"  # Change this to your actual Python script name

# Run the Python script with nohup
nohup python $SCRIPT_NAME > logs_right/output.log 2>&1 &

# Save the process ID to a file for later reference
echo $! > logs_right/process_pid.txt

echo "Process started with PID $(cat logs_right/process_pid.txt)"
echo "Output is being logged to logs_right/output.log"