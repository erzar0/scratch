#!/bin/bash
# Why C1 (our exact picks in the V4.1 reference) idles more than the reference in
# stage 1: timeline traces with CPU stacks of step 6 (stage 1 of a 30-step schedule)
# -> logs/trace_csa2_{ref,ours}/step6.json, read by profile_gaps.py.
# Run from experiments/csa2:
#   setsid nohup ./queue_trace_round4.sh < /dev/null > logs/queue_trace_round4.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_trace_round4.pid
HERE=$(pwd)
export TRITON_CACHE_DIR=$HOME/.triton/cache
export SWEEP_ITERS=30 SWEEP_EXT_ITERS=2 SWEEP_VAL_EVERY=0 SAVE_CHECKPOINT=0
export PROFILE_AT=6 PROFILE_CPU_AT=6
for v in ref ours; do
    mkdir -p logs/trace_csa2_$v
    PROFILE_TRACE=$HERE/logs/trace_csa2_$v PROFILE_OUT=$HERE/logs/profile_trace_csa2_$v.csv \
        b/run_b.sh exact train_csa2_${v}_profile.py > logs/trace_csa2_$v.log 2>&1
    status=$?
    echo "$(date +%H:%M) trace csa2_$v exit $status"
done
echo "$(date +%H:%M) done"
