#!/bin/bash

# Create logs directory
mkdir -p logs_right4

# Define the list of node_num values
node_nums=(910 911 912 913 914 915 916 917 918 919 920 921 922 923 924 925 926 927 928 929 930 931 932 933 934 935 937 938 939 940 941 942 943 944 945 946 947 948 949 950 951 952 953 954 955 956 957 958 960 961 962 963 964 965 968 969 970 971 973 974 975 976 977 978 979 980 981 982 983)  

# Log file
LOG_FILE="logs_right4/evaluation_progress.log"

# Create a temporary script file for the loop
cat > temp_eval_script.sh << 'EOF'
#!/bin/bash
LOG_FILE="logs_right4/evaluation_progress.log"
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
    echo "$(date) - Completed: ${node_num}" >> logs_right4/completed_nodes.txt
done

echo "[$(date)] All evaluations completed!" >> $LOG_FILE
EOF

# Make the temp script executable
chmod +x temp_eval_script.sh

# Run the temp script with nohup, passing the array elements as arguments
nohup ./temp_eval_script.sh "${node_nums[@]}" > /dev/null 2>&1 &

# Save the process ID
PID=$!
echo $PID > logs_right4/evaluation_pid.txt
echo "Evaluation process started with PID: $PID"
echo "Progress is being logged to: $LOG_FILE"
echo "You can safely disconnect. To check status later, use: tail -f $LOG_FILE"