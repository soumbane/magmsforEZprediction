#!/bin/bash

# Wait for the Bonn evaluation shards to finish, then combine the per-node results and build the
# publication table. Detached like the shards themselves, so the whole chain survives closing SSH:
# launch this after ./run_all_bonn_evaluation.sh and the results are ready when you come back.
#
# Usage (from this directory):
#   setsid nohup ./finish_bonn_evaluation.sh > logs_bonn_eval/finish.log 2>&1 &
#
# Check afterwards:
#   tail -30 logs_bonn_eval/finish.log
#
# If some nodes did not finish, the combined files are still written (with the missing nodes listed)
# but the publication table is deliberately skipped: a table averaged over an unknown subset of nodes
# is worse than no table.

LOG_DIR="logs_bonn_eval"
EXPECTED_NODES=711

if [ ! -f "$LOG_DIR/pids.txt" ]; then
    echo "ERROR: no $LOG_DIR/pids.txt. Start the run first with ./run_all_bonn_evaluation.sh"
    exit 1
fi

echo "Waiting for the evaluation shards to finish, started at $(date)"

# poll until no shard PID is alive
while true; do
    alive=0
    for pid in $(cat "$LOG_DIR/pids.txt"); do
        if kill -0 "$pid" 2>/dev/null; then
            alive=$((alive + 1))
        fi
    done

    [ "$alive" -eq 0 ] && break

    done_nodes=$(grep -h -c "Done Evaluating" "$LOG_DIR"/*.log 2>/dev/null | paste -sd+ | bc)
    echo "[$(date +%H:%M:%S)] ${alive} shard(s) still running, ${done_nodes}/${EXPECTED_NODES} nodes done"
    sleep 60
done

done_nodes=$(grep -h -c "Done Evaluating" "$LOG_DIR"/*.log 2>/dev/null | paste -sd+ | bc)

echo
echo "All shards exited at $(date). Nodes evaluated: ${done_nodes}/${EXPECTED_NODES}"
echo

complete=1
if [ "$done_nodes" -ne "$EXPECTED_NODES" ]; then
    echo "WARNING: ${done_nodes} of ${EXPECTED_NODES} nodes finished. Combining what exists, and"
    echo "         SKIPPING the publication table so no incomplete average is reported."
    complete=0
fi

skip_flag=""
[ "$complete" -eq 0 ] && skip_flag="--skip_missing"

echo "=== Combining left hemisphere ==="
python combine_bonn_cohort_results.py --hemisphere left $skip_flag || exit 1

echo
echo "=== Combining right hemisphere ==="
python combine_bonn_cohort_results.py --hemisphere right $skip_flag || exit 1

if [ "$complete" -eq 1 ]; then
    echo
    echo "=== Building the combination table ==="
    python make_bonncohort_combination_table.py || exit 1
else
    echo
    echo "Publication table skipped because the run is incomplete. Re-run the missing shards, then:"
    echo "  python make_bonncohort_combination_table.py"
fi

echo
echo "Finished at $(date)"
