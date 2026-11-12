#!/bin/bash
# C3 + C4 for run 7 (1b0702e3), no training: block prefilter with picks at layers 0, 4;
# block meanstd at 512 kept entries (p512's budget); then decode timing
# (bench/decode_block.py). Run from experiments/csa2:
#   setsid nohup ./queue_block2.sh < /dev/null > logs/queue_block2.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_block2.pid
R=$HOME/repos/scratch
RUN=1b0702e3-0cc6-43cb-8fc1-a133899b90d0
export TRITON_CACHE_DIR=$HOME/.triton/cache
eval_rule() {  # tag script rule [args]
    local tag=$1 script=$2
    shift 2
    CSA2_SELECT=exact EVAL_COMPILE=1 PYTHONPATH=$R/src \
        $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/rule_eval.py \
        $R/experiments/csa2/b/$script $RUN 128 logs/rule_${RUN:0:8}_$tag.csv "$@" \
        > logs/rule_${RUN:0:8}_$tag.log 2>&1
    echo "$(date +%H:%M) $tag exit $? $(grep ',0,8192,' logs/rule_${RUN:0:8}_$tag.csv | tail -1)"
}
eval_rule share04_block_meanstd_8x32 train_tournament_kv_win1_pick4_share04_fast.py \
    block 8 32 meanstd
eval_rule block_meanstd_8x64 train_tournament_kv_win1_pick4_fast.py block 8 64 meanstd
cd $R
PYTHONPATH=src:experiments/csa2/bench $HOME/repos/modded-nanogpt/.venv/bin/python \
    experiments/csa2/bench/decode_block.py > experiments/csa2/logs/decode_block.log 2>&1
echo "$(date +%H:%M) decode_block exit $?"
echo "$(date +%H:%M) done"
