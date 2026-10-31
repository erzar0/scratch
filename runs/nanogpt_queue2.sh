#!/bin/sh
# Throwaway queue: third seed of stripped-dense record 91 (seeds so far: 3.829, 3.573) and
# a second seed of full dense, both validating every 500 steps: is stripped dense
# seed-unstable, and does full dense also stall in stage 3? Status lines to stdout; each
# run's log to runs/logs/<name>.log.
set -e
cd /home/kyre/repos/scratch
R91=/home/kyre/repos/modded-nanogpt-r91
run() {
    name=$1
    shift
    echo "$(date +%H:%M) start $name"
    "$@" > "runs/logs/$name.log" 2>&1
}
export SWEEP_VAL_EVERY=500
export TRAIN_SEED=2
run nanogpt_dense_stripped_seed2 $R91/run_dense_stripped.sh
export TRAIN_SEED=1
run nanogpt_dense_seed1 $R91/run_r91.sh
echo "$(date +%H:%M) queue done"
