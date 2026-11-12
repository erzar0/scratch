#!/bin/bash
# After queue_decode2.sh: eval-time window probes on the 6-head B checkpoints (own rule,
# csa2_flex window 64 / 32 / 16). Run from experiments/csa2 with setsid; logs in logs/.
set -uo pipefail
R=$HOME/repos/scratch
HERE=$(pwd)
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_decode2.pid)" 2>/dev/null; do sleep 20; done
while read -r run own; do
    for w in 64 32 16; do
        EVAL_WINDOW=$w CSA2_SELECT=$own EVAL_COMPILE=1 PYTHONPATH=$R/src \
            $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/rule_eval.py \
            $HERE/b/train_tournament_kv_fast.py $run 128 logs/window_${run:0:8}_$w.csv $own \
            > logs/window_${run:0:8}_$w.log 2>&1
    done
done <<LIST
5f0cf619-be55-44ea-ad17-e182a1b0858a exact
f7f5027f-86ca-47c3-8fc0-4b4c0447ba95 tournament
LIST
echo done
