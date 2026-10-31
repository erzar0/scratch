#!/bin/sh
# Throwaway queue: CSA2 and stripped-dense record 91 trained with documents cut at the
# row length (SWEEP_LONG_DOCS=1: 2048/4096/6144 tokens per stage instead of 896/2048),
# seed 1 like their normal-cut runs, val every 500; then each checkpoint's long eval
# (src/nanogpt/long.py). Status lines to stdout; logs to runs/logs/<name>.log.
set -e
cd /home/kyre/repos/scratch
R91=/home/kyre/repos/modded-nanogpt-r91
run() {
    name=$1
    script=$2
    echo "$(date +%H:%M) start $name"
    "$R91/run_$script.sh" > "runs/logs/$name.log" 2>&1
    id=$(grep -m1 -oE "^logs/[0-9a-f-]+\.txt" "runs/logs/$name.log" | sed 's|logs/||; s|\.txt||')
    echo "$(date +%H:%M) $name run id $id, long eval"
    PYTHONPATH=src $R91/.venv/bin/python -m nanogpt.long "train_$script.py" "$id" \
        > "runs/logs/${name}_long.log" 2>&1
}
export TRAIN_SEED=1 SWEEP_VAL_EVERY=500 SWEEP_LONG_DOCS=1
run nanogpt_csa2_longdocs_seed1 csa2
run nanogpt_dense_stripped_longdocs_seed1 dense_stripped
echo "$(date +%H:%M) queue done"
