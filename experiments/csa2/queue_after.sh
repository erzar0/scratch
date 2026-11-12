#!/bin/bash
# After queue_mqa.sh: decode tests and bench (GPU alone), then rule-swap evals of the
# B checkpoints (own rule, the other rule, the decode prefilter keeping 128 / 512).
# Run from experiments/csa2 with setsid; logs in logs/.
set -uo pipefail
HERE=$(pwd)
R=$HOME/repos/scratch
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_mqa.pid)" 2>/dev/null; do sleep 30; done
cd $R
for t in test_csa2_decode test_csa2_select_decode; do
    uv run python -m tests.nanogpt.$t > $HERE/logs/$t.log 2>&1
done
PYTHONPATH=src $HOME/repos/modded-nanogpt/.venv/bin/python \
    experiments/csa2/bench/decode_csa2.py runs/nanogpt/long_context/csa2_decode.csv \
    > $HERE/logs/decode_bench.log 2>&1
cd $HERE
Q=$(grep -m1 -oE "^logs/[0-9a-f-]{36}\.txt" logs/full_mqa_fast.log | cut -c6-41)
# run script own-rule other-rules...
while read -r run script own rules; do
    for rule in $rules; do
        case $rule in
            p*) args="prefilter ${rule#p} 16" ;;
            *) args=$rule ;;
        esac
        CSA2_SELECT=$own EVAL_COMPILE=1 PYTHONPATH=$R/src \
            $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/rule_eval.py \
            $HERE/b/$script $run 128 logs/rule_${run:0:8}_$rule.csv $args \
            > logs/rule_${run:0:8}_$rule.log 2>&1
    done
done <<LIST
5f0cf619-be55-44ea-ad17-e182a1b0858a train_tournament_kv_fast.py exact exact tournament p128 p512
f7f5027f-86ca-47c3-8fc0-4b4c0447ba95 train_tournament_kv_fast.py tournament tournament exact p128 p512
$Q train_tournament_kv_mqa_fast.py tournament exact p128 p512
LIST
echo done
