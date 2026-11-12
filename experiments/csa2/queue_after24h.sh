#!/bin/bash
# After queue_24h.sh (no training): prefilter keep 512 at eval on the two clean
# 1-head-window runs (d3c62030 picks every 2, 1b0702e3 every 4; must equal exact),
# then the decode heat map with their decode rows. Run from experiments/csa2 with setsid.
set -uo pipefail
HERE=$(pwd)
R=$HOME/repos/scratch
export TRITON_CACHE_DIR=$HOME/.triton/cache
for pair in d3c62030-839b-46fa-9cc3-4434d250d0fd:train_tournament_kv_win1_fast.py \
            1b0702e3-0cc6-43cb-8fc1-a133899b90d0:train_tournament_kv_win1_pick4_fast.py; do
    run=${pair%%:*} S=${pair#*:}
    CSA2_SELECT=exact EVAL_COMPILE=1 PYTHONPATH=$R/src \
        $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/rule_eval.py \
        $HERE/b/$S $run 128 logs/rule_${run:0:8}_p512.csv prefilter 512 16 \
        > logs/rule_${run:0:8}_p512.log 2>&1
    echo "$(date +%H:%M) ${run:0:8} p512 exit $? $(grep ',0,8192,' logs/rule_${run:0:8}_p512.csv | tail -1)"
done
cd $R
PYTHONPATH=src:experiments/csa2/bench $HOME/repos/modded-nanogpt/.venv/bin/python \
    experiments/csa2/bench/decode_heat.py > $HERE/logs/decode_heat.log 2>&1
echo "$(date +%H:%M) heat exit $?"
