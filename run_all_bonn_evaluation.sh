#!/bin/bash

# Launch every root-level BonnCohort evaluation shard detached, so the run survives closing the SSH
# session. Each shard keeps its own log and its own PID file under logs_bonn_eval/.
#
# No training is involved. Every node reuses its 3 AddCohort checkpoints and scores them on the 85
# Bonn subjects (OpenNeuro ds004199), which have T1 and FLAIR only, across the 3 non-empty subsets
# of {T1, FLAIR}. The absent sequences are dropped from the model's target_dict.
#
# Stage the cohort first:
#   python data/prepare_bonn_cohort.py
#
# Usage (from this directory):
#   ./run_all_bonn_evaluation.sh
#
# Progress:
#   grep -c "Done Evaluating" logs_bonn_eval/*.log     # nodes finished per shard
#   tail -f logs_bonn_eval/eval_ALL_bonn_left.log      # follow one shard
#
# Stop everything:
#   kill $(cat logs_bonn_eval/pids.txt)

LOG_DIR="logs_bonn_eval"
mkdir -p "$LOG_DIR"

# one entry per shard; the GPU assignment is inherited from the AddCohort evaluation, which pairs a
# large shard with a small one so each of the four GPUs carries a comparable number of nodes
# (171/184/178/178)
scripts=(
    eval_ALL_bonn_left.sh        # 102 nodes, cuda:0
    eval_ALL_bonn_left1.sh       #  89 nodes, cuda:2
    eval_ALL_bonn_left2.sh       #  95 nodes, cuda:1
    eval_ALL_bonn_right.sh       #  89 nodes, cuda:1
    eval_ALL_bonn_right1.sh      #  89 nodes, cuda:2
    eval_ALL_bonn_right2.sh      #  89 nodes, cuda:3
    eval_ALL_bonn_right3.sh      #  89 nodes, cuda:3
    eval_ALL_bonn_right4.sh      #  69 nodes, cuda:0
)

# refuse to start a second copy on top of a run that is still going
if [ -f "$LOG_DIR/pids.txt" ]; then
    for pid in $(cat "$LOG_DIR/pids.txt"); do
        if kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: a previous run is still active (PID $pid). Stop it first with:"
            echo "  kill \$(cat $LOG_DIR/pids.txt)"
            exit 1
        fi
    done
fi

# evaluation is worthless without the trained checkpoints, so fail early rather than 700 times
if [ ! -d "experiments" ]; then
    echo "ERROR: no experiments/ directory. The AddCohort checkpoints must be trained first."
    exit 1
fi

# and worthless without the staged cohort, which is what carries the recovered labels
if [ ! -d "/media/user1/MyHDataStor41/Soumyanil_EZ_Pred_project/Data/All_Hemispheres/Left_Hemis/Part_2/Node_6/Bonn_Val_Data_85" ]; then
    echo "ERROR: the Bonn cohort is not staged. Run it first with:"
    echo "  python data/prepare_bonn_cohort.py"
    exit 1
fi

: > "$LOG_DIR/pids.txt"

echo "Starting Bonn evaluation at $(date)"

for script in "${scripts[@]}"; do
    if [ ! -x "./$script" ]; then
        echo "ERROR: ./$script is missing or not executable."
        exit 1
    fi

    log_file="$LOG_DIR/${script%.sh}.log"

    # setsid detaches from the terminal session and nohup ignores SIGHUP, so the shard keeps
    # running after the SSH connection is closed
    setsid nohup ./"$script" > "$log_file" 2>&1 &
    pid=$!
    echo "$pid" >> "$LOG_DIR/pids.txt"

    echo "  launched ${script} (PID ${pid}) -> ${log_file}"
done

echo
echo "All ${#scripts[@]} shards launched. You can safely close the SSH connection."
echo "Progress:  grep -c 'Done Evaluating' $LOG_DIR/*.log"
echo "PIDs:      $LOG_DIR/pids.txt"
echo
echo "When it finishes, combine the per-node results with:"
echo "  python combine_bonn_cohort_results.py --hemisphere left"
echo "  python combine_bonn_cohort_results.py --hemisphere right"
echo "  python make_bonncohort_combination_table.py"
