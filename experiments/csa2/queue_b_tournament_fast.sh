#!/bin/bash
# B tournament retrained with the fast pick kernel (csa2_select_fast; same picks as
# run f7f5027f, which used the slow kernel: 2.76 h): after queue_pick4.sh, the full
# run, then per-position validation with its own picks and exact picks.
# Run from experiments/csa2 with setsid; logs in logs/.
set -uo pipefail
HERE=$(pwd)
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_pick4.pid)" 2>/dev/null; do sleep 30; done
S=train_tournament_kv_fast.py
b/run_b.sh tournament $S > logs/full_b_tournament_fast.log 2>&1 || exit 1
run=$(grep -m1 -oE "^logs/[0-9a-f-]{36}\.txt" logs/full_b_tournament_fast.log | cut -c6-41)
echo "run $run"
for rule in tournament exact; do
    CSA2_SELECT=tournament EVAL_COMPILE=1 PYTHONPATH=$HOME/repos/scratch/src \
        $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/rule_eval.py \
        $HERE/b/$S $run 128 logs/rule_${run:0:8}_$rule.csv $rule \
        > logs/rule_${run:0:8}_$rule.log 2>&1
done
echo done
