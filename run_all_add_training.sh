#!/bin/bash

# Launch every root-level AddCohort training shard detached, so the run survives closing the SSH
# session. Each shard keeps its own log and its own PID file under logs_add_root/.
#
# Usage (from this directory):
#   ./run_all_add_training.sh
#
# Progress:
#   grep -c "saved at" logs_add_root/*.log          # nodes finished per shard
#   tail -f logs_add_root/train_ALL_add_left.log    # follow one shard
#
# Stop everything:
#   kill $(cat logs_add_root/pids.txt)

LOG_DIR="logs_add_root"
mkdir -p "$LOG_DIR"

# one entry per shard, in the same order as the per-GPU assignment below
scripts=(
    train_ALL_add_left.sh       # 82 nodes, cuda:0
    train_ALL_add_left_1.sh     # 73 nodes, cuda:0
    train_ALL_add_left_2.sh     # 69 nodes, cuda:1
    train_ALL_add_left_3.sh     # 62 nodes, cuda:1
    train_ALL_add_right.sh      # 78 nodes, cuda:0
    train_ALL_add_right_1.sh    # 78 nodes, cuda:2
    train_ALL_add_right_2.sh    # 78 nodes, cuda:1
    train_ALL_add_right_3.sh    # 64 nodes, cuda:3
    train_ALL_add_right_4.sh    # 64 nodes, cuda:2
    train_ALL_add_right_5.sh    # 63 nodes, cuda:3
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

: > "$LOG_DIR/pids.txt"

echo "Starting training at $(date)"

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
echo "Progress:  grep -c 'saved at' $LOG_DIR/*.log"
echo "PIDs:      $LOG_DIR/pids.txt"
