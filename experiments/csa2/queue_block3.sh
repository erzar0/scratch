#!/bin/bash
# Run 7 (1b0702e3) at eval, no training: block prefilter at 512 kept entries (the decode
# bench's 64 blocks of 8) for mean / min-max, and with picks at layers 0, 4. Waits for
# queue_block2.sh (one GPU job at a time). Run from experiments/csa2:
#   setsid nohup ./queue_block3.sh < /dev/null > logs/queue_block3.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_block3.pid
while ps -p "$(cat logs/queue_block2.pid)" > /dev/null; do sleep 15; done
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
for summary in mean minmax; do
    eval_rule block_${summary}_8x64 train_tournament_kv_win1_pick4_fast.py block 8 64 $summary
done
for summary in meanstd mean; do
    eval_rule share04_block_${summary}_8x64 train_tournament_kv_win1_pick4_share04_fast.py \
        block 8 64 $summary
done
echo "$(date +%H:%M) done"
