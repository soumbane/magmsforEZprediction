#!/bin/bash

# Launch every root-level AddCohort evaluation shard detached, so the run survives closing the SSH
# session. Each shard keeps its own log and its own PID file under logs_add_eval/.
#
# Every node is evaluated on both subject groups with the same 3 trial checkpoints:
#   - the 17 subjects with all five sequences, on ALL 31 modality combinations
#   - the 13 subjects without DWIC, on the 15 DWIC-free combinations
#
# Run this only after training has finished, since it reads
# experiments/exp_node<N>/AddCohort/magms_trial<i>.exp/checkpoints/best_bal_accuracy.model
#
# Usage (from this directory):
#   ./run_all_add_evaluation.sh
#
# Progress:
#   grep -c "Done Evaluating" logs_add_eval/*.log     # nodes finished per shard
#   tail -f logs_add_eval/eval_ALL_add_left.log       # follow one shard
#
# Stop everything:
#   kill $(cat logs_add_eval/pids.txt)

LOG_DIR="logs_add_eval"
mkdir -p "$LOG_DIR"

# one entry per shard; the GPU assignment pairs a large shard with a small one so that each of the
# four GPUs carries a comparable number of nodes (171/184/178/178)
scripts=(
    eval_ALL_add_left.sh        # 102 nodes, cuda:0
    eval_ALL_add_left1.sh       #  89 nodes, cuda:2
    eval_ALL_add_left2.sh       #  95 nodes, cuda:1
    eval_ALL_add_right.sh       #  89 nodes, cuda:1
    eval_ALL_add_right1.sh      #  89 nodes, cuda:2
    eval_ALL_add_right2.sh      #  89 nodes, cuda:3
    eval_ALL_add_right3.sh      #  89 nodes, cuda:3
    eval_ALL_add_right4.sh      #  69 nodes, cuda:0
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
    echo "ERROR: no experiments/ directory. Train first with ./run_all_add_training.sh"
    exit 1
fi

: > "$LOG_DIR/pids.txt"

echo "Starting evaluation at $(date)"

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
echo "  python combine_add_cohort_results.py --hemisphere left  --group 17"
echo "  python combine_add_cohort_results.py --hemisphere left  --group 13"
echo "  python combine_add_cohort_results.py --hemisphere right --group 17"
echo "  python combine_add_cohort_results.py --hemisphere right --group 13"
