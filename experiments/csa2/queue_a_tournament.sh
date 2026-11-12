#!/bin/bash
# A tournament: CSA2 stripped (train_csa2.py) with the indexer swapped for our
# tournament picks (a/csa2_src glue, csa2_select_fast kernel). Smoke, full run, then
# its per-position validation. Run from experiments/csa2 with setsid; logs in logs/.
set -uo pipefail
HERE=$(pwd)
R91=$HOME/repos/modded-nanogpt-r91
export TRITON_CACHE_DIR=$HOME/.triton/cache
export SCRATCH_SRC=$HERE/a/csa2_src/src CSA2_SELECT=tournament CSA2_FAST=1 TRAIN_SEED=1
SWEEP_ITERS=30 SWEEP_EXT_ITERS=2 SWEEP_VAL_EVERY=15 SAVE_CHECKPOINT=0 \
    $R91/run_csa2.sh > logs/smoke_a_tournament.log 2>&1 || exit 1
last=$(grep -oE "step:3[0-9]/[0-9]+ val_loss:[0-9.]+" logs/smoke_a_tournament.log | tail -n 1)
echo "smoke: $last"
[[ $last =~ val_loss:[0-7]\. ]] || { echo "smoke val loss too high or NaN"; exit 1; }
$R91/run_csa2.sh > logs/full_a_tournament.log 2>&1 || exit 1
run=$(grep -m1 -oE "^logs/[0-9a-f-]{36}\.txt" logs/full_a_tournament.log | cut -c6-41)
echo "run $run"
EVAL_COMPILE=1 $R91/.venv/bin/python eval/packed_val.py train_csa2.py $run 128 \
    logs/eval_$run.csv > logs/eval_a_tournament.log 2>&1
echo done
