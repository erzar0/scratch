#!/bin/bash
# Round 3, after the profiles (queue_round3_c.sh): decode bench with the reference
# layout (bench/decode_block.py), then be093c56 (exact picks every 4 layers, record
# 91's own window; best loss) with FP8 entries at eval (b/make_fp8_script.py all) ->
# logs/rule_be093c56_fp8all_exact.csv; gate with eval/paired.py. Run from
# experiments/csa2:
#   setsid nohup ./queue_round3_d.sh < /dev/null > logs/queue_round3_d.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_round3_d.pid
R=$HOME/repos/scratch
export TRITON_CACHE_DIR=$HOME/.triton/cache
(cd $R && PYTHONPATH=src:experiments/csa2/bench \
    $HOME/repos/modded-nanogpt/.venv/bin/python experiments/csa2/bench/decode_block.py) \
    > logs/decode_block.log 2>&1
status=$?
echo "$(date +%H:%M) decode_block exit $status"
run=be093c56-fd99-48c5-adc6-59a8421f8d2c
out=logs/rule_${run:0:8}_fp8all_exact
CSA2_SELECT=exact EVAL_COMPILE=1 PYTHONPATH=$R/src \
    $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/rule_eval.py \
    $R/experiments/csa2/b/train_tournament_kv_pick4_fp8all_fast.py $run 128 \
    $out.csv exact > $out.log 2>&1
status=$?
echo "$(date +%H:%M) fp8all eval exit $status: $(tail -n 1 $out.csv)"
echo "$(date +%H:%M) done"
