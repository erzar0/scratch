#!/bin/bash
# 6-head window of 32 tokens (record 91's own K and V heads): after queue_decode3.sh,
# smoke, then the full tournament run with fast picks, its per-position validation
# and rule swaps (exact, prefilter keep 512), all at window 32.
# Run from experiments/csa2 with setsid; logs in logs/.
set -uo pipefail
HERE=$(pwd)
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_decode3.pid)" 2>/dev/null; do sleep 30; done
S=train_tournament_kv_w32_fast.py
SWEEP_ITERS=30 SWEEP_EXT_ITERS=2 SWEEP_VAL_EVERY=15 SAVE_CHECKPOINT=0 \
    b/run_b.sh tournament $S > logs/smoke_w32.log 2>&1 || exit 1
last=$(grep -oE "step:3[0-9]/[0-9]+ val_loss:[0-9.]+" logs/smoke_w32.log | tail -n 1)
echo "smoke: $last"
[[ $last =~ val_loss:[0-6]\. ]] || { echo "smoke val loss too high or NaN"; exit 1; }
b/run_b.sh tournament $S > logs/full_w32.log 2>&1 || exit 1
run=$(grep -m1 -oE "^logs/[0-9a-f-]{36}\.txt" logs/full_w32.log | cut -c6-41)
echo "run $run"
for rule in tournament exact p512; do
    case $rule in
        p*) args="prefilter ${rule#p} 16" ;;
        *) args=$rule ;;
    esac
    CSA2_SELECT=tournament EVAL_COMPILE=1 EVAL_WINDOW=32 \
        PYTHONPATH=$HOME/repos/scratch/src \
        $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/rule_eval.py \
        $HERE/b/$S $run 128 logs/rule_${run:0:8}_$rule.csv $args \
        > logs/rule_${run:0:8}_$rule.log 2>&1
done
echo done
