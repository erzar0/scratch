#!/usr/bin/env bash
# Experiments on the CSA2 checkpoints, one GPU job at a time, after the training
# queue (runs/train/queue.sh) exits. Status lines go to stdout, each step's log to
# runs/logs/<step>.log; a failed step is reported and the queue goes on.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
while [ -e /proc/112355 ]; do sleep 60; done
mkdir -p runs/logs

step() {
    local name=$1
    shift
    echo "$(date +%T) start $name"
    if env "$@" > "runs/logs/$name.log" 2>&1; then
        echo "$(date +%T) done $name"
    else
        echo "$(date +%T) FAILED $name"
    fi
}

suite() {  # $1 = T, $2 = batch, $3 = router sizes
    local t=$1 b=$2 sizes=$3
    step "eval_csa_T$t" SEQ="$t" BATCH="$b" uv run python -m experiments.eval_csa
    step "router_T$t" SEQ="$t" uv run python -m experiments.train_router $sizes
    for mode in mean bound oracle learned mixed; do
        step "route_${mode}_T$t" SEQ="$t" BATCH="$b" \
            uv run python -m experiments.eval_route "$mode"
    done
    for exp in paired_se copy_test pick_recall; do
        step "${exp}_T$t" SEQ="$t" BATCH="$b" uv run python -m "experiments.$exp"
    done
}

suite 2048 2 "8 16"
suite 8192 1 "16 32"
# model 2: headlines chosen on model 1's sweeps
step router_T2048_s1 SEED=1 SEQ=2048 uv run python -m experiments.train_router 8 16
for exp in paired_se copy_test pick_recall; do
    step "${exp}_T2048_s1" SEED=1 SEQ=2048 BATCH=2 uv run python -m "experiments.$exp"
done
step bench_indexer uv run python -m bench.indexer
echo "$(date +%T) queue finished"
