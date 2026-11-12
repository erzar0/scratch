#!/bin/bash
# Full MQA-window tournament run with fast picks, then its per-position validation.
# Run from experiments/csa2 with setsid; logs in logs/.
set -euo pipefail
HERE=$(pwd)
export TRITON_CACHE_DIR=$HOME/.triton/cache
b/run_b.sh tournament train_tournament_kv_mqa_fast.py > logs/full_mqa_fast.log 2>&1
run=$(grep -m1 -oE "^logs/[0-9a-f-]{36}\.txt" logs/full_mqa_fast.log | cut -c6-41)
CSA2_SELECT=tournament EVAL_COMPILE=1 PYTHONPATH=$HOME/repos/scratch/src \
    $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/packed_val.py \
    $HERE/b/train_tournament_kv_mqa_fast.py $run 128 logs/eval_$run.csv \
    > logs/eval_mqa_fast.log 2>&1
echo done
