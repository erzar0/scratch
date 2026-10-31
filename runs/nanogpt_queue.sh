#!/bin/sh
# Throwaway queue: second seed of CSA2 and stripped-dense record 91 (first runs: random
# seeds, val every 2000), validating every 500 steps to see the stage-3 stall and the
# cooldown. Status lines to stdout; each run's log to runs/logs/<name>.log.
set -e
cd /home/kyre/repos/scratch
R91=/home/kyre/repos/modded-nanogpt-r91
run() {
    name=$1
    shift
    echo "$(date +%H:%M) start $name"
    "$@" > "runs/logs/$name.log" 2>&1
}
export TRAIN_SEED=1 SWEEP_VAL_EVERY=500
run nanogpt_csa2_seed1 $R91/run_csa2.sh
run nanogpt_dense_stripped_seed1 $R91/run_dense_stripped.sh
echo "$(date +%H:%M) queue done"
