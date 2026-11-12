#!/bin/bash
# Routed channels (MoE over attention channels, src/nanogpt/csa2_route.py) on B:
# after queue_profile.sh, tests, smoke, the full run (exact picks on q_routed, fast
# kernel), then per-position validation with its own rule and tournament picks.
# Run from experiments/csa2 with setsid; logs in logs/.
set -uo pipefail
HERE=$(pwd)
R=$HOME/repos/scratch
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_profile.pid)" 2>/dev/null; do sleep 30; done
cd $R
for t in test_csa2_route test_csa2_flex test_csa2_select; do
    uv run python -m tests.nanogpt.$t > $HERE/logs/$t.log 2>&1 || { echo "$t failed"; exit 1; }
done
echo "tests passed"
cd $HERE
S=train_tournament_kv_route_fast.py
SWEEP_ITERS=30 SWEEP_EXT_ITERS=2 SWEEP_VAL_EVERY=15 SAVE_CHECKPOINT=0 \
    b/run_b.sh exact $S > logs/smoke_route.log 2>&1 || exit 1
last=$(grep -oE "step:3[0-9]/[0-9]+ val_loss:[0-9.]+" logs/smoke_route.log | tail -n 1)
echo "smoke: $last"
[[ $last =~ val_loss:[0-6]\. ]] || { echo "smoke val loss too high or NaN"; exit 1; }
b/run_b.sh exact $S > logs/full_route.log 2>&1 || exit 1
run=$(grep -m1 -oE "^logs/[0-9a-f-]{36}\.txt" logs/full_route.log | cut -c6-41)
echo "run $run"
for rule in exact tournament; do
    CSA2_SELECT=exact EVAL_COMPILE=1 PYTHONPATH=$R/src \
        $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/rule_eval.py \
        $HERE/b/$S $run 128 logs/rule_${run:0:8}_$rule.csv $rule \
        > logs/rule_${run:0:8}_$rule.log 2>&1
done
echo done
