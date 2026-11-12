#!/bin/bash
# Run 7's checkpoint under another pick rule, no training (ANALYSIS.md tests 1-3):
#   eval_rule.sh SCRIPT RULE [ARGS] -> results/rule_<tag>.csv, e.g.
#   eval_rule.sh train_tournament_kv_win1_pick4_share04_fast.py block 8 64 mean
# SCRIPT: train_tournament_kv_win1_pick4_fast.py (picks at 0, 4, 8) or
# train_tournament_kv_win1_pick4_share04_fast.py (picks at 0, 4; make_pick_share_script.py)
# RULE: exact | prefilter KEEP DIMS | block SIZE KEEP SUMMARY (mean max minmax meanstd)
set -eu
HERE=$(dirname "$(readlink -f "$0")")
RUN=1b0702e3-0cc6-43cb-8fc1-a133899b90d0
SCRIPT=$1
shift
TAG=$(basename "$SCRIPT" .py | sed 's/train_tournament_kv_win1_pick4_fast/picks048/; s/train_tournament_kv_win1_pick4_\(.*\)_fast/\1/')_$(IFS=_; echo "$*")
export CSA2_SELECT=exact EVAL_COMPILE=1 SCRATCH_SRC=$HERE/src
export PYTHONPATH=$HERE/src:$HERE/eval TRITON_CACHE_DIR=$HOME/.triton/cache
exec $HOME/repos/modded-nanogpt-r91/.venv/bin/python $HERE/eval/rule_eval.py \
    $HERE/$SCRIPT $RUN 128 $HERE/results/rule_$TAG.csv "$@"
