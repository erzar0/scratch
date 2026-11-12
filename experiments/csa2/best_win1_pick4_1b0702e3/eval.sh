#!/bin/bash
# Validation loss by position (1M val tokens, 128 rows of 8192) of a run trained by
# run.sh, exact picks: eval.sh [RUN_ID] -> results/per_position_<id8>.csv
set -eu
HERE=$(dirname "$(readlink -f "$0")")
RUN=${1:-1b0702e3-0cc6-43cb-8fc1-a133899b90d0}
export CSA2_SELECT=exact EVAL_COMPILE=1 SCRATCH_SRC=$HERE/src
export PYTHONPATH=$HERE/src:$HERE/eval TRITON_CACHE_DIR=$HOME/.triton/cache
exec $HOME/repos/modded-nanogpt-r91/.venv/bin/python $HERE/eval/rule_eval.py \
    $HERE/train_tournament_kv_win1_pick4_fast.py $RUN 128 \
    $HERE/results/per_position_${RUN:0:8}.csv exact
