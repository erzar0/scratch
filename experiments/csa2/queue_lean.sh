#!/bin/bash
# Round 2 phase A (plan wondrous-twirling-flurry), no training, after queue_round2.sh's
# current run (one GPU job at a time): lean script checks (tests, run 7 eval, window
# bench, smoke step times A B A B) and FP8 entry cache evals on run 7 with picks at
# layers 0, 4. Run from experiments/csa2:
#   setsid nohup ./queue_lean.sh < /dev/null > logs/queue_lean.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_lean.pid
while ps -p "$(cat logs/queue_round2.pid)" > /dev/null; do sleep 30; done
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
for t in csa2_lean csa2_flex_gqa; do
    (cd $R && PYTHONPATH=src uv run python tests/nanogpt/test_$t.py) > logs/test_$t.log 2>&1
    echo "$(date +%H:%M) test $t exit $? $(tail -1 logs/test_$t.log)"
done
eval_rule lean_exact train_tournament_kv_win1_pick4_lean_fast.py exact
(cd $R && PYTHONPATH=src uv run python experiments/csa2/bench/train_window.py) \
    > logs/train_window.log 2>&1
echo "$(date +%H:%M) train_window exit $?"
for i in 1 2; do
    for s in win1_pick4 win1_pick4_lean; do
        SWEEP_ITERS=300 SWEEP_EXT_ITERS=10 SWEEP_VAL_EVERY=0 SAVE_CHECKPOINT=0 \
            b/run_b.sh exact train_tournament_kv_${s}_fast.py > logs/smoke_${s}_$i.log 2>&1
        echo "$(date +%H:%M) smoke $s $i exit $? $(python3 stage_times.py logs/smoke_${s}_$i.log)"
    done
done
for m in picks all; do
    eval_rule share04_fp8${m}_exact train_tournament_kv_win1_pick4_share04_fp8${m}_fast.py exact
    eval_rule share04_fp8${m}_block_mean_8x64 \
        train_tournament_kv_win1_pick4_share04_fp8${m}_fast.py block 8 64 mean
done
echo "$(date +%H:%M) done"
