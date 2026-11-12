#!/bin/bash
# Run 7 (1b0702e3) at eval, no training: picks shared across groups (IndexCache;
# b/make_pick_share_script.py: picks at layer 0 only, at 0 and 4, at 0 and 8), then
# the block prefilter (csa2_select_block: 4 summaries x 128 or 256 kept entries).
# Per-position CSVs -> logs/rule_1b0702e3_<tag>.csv. Run from experiments/csa2:
#   setsid nohup ./queue_block.sh < /dev/null > logs/queue_block.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_block.pid
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
for t in 0 04 08; do
    eval_rule share$t train_tournament_kv_win1_pick4_share${t}_fast.py exact
done
for summary in mean max minmax meanstd; do
    for sk in 8:16 16:16 8:32; do
        size=${sk%%:*} keep=${sk#*:}
        eval_rule block_${summary}_${size}x$keep train_tournament_kv_win1_pick4_fast.py \
            block $size $keep $summary
    done
done
echo "$(date +%H:%M) done"
