#!/bin/bash
# C3-smooth (9624b171, b/train_csa2_ours_r4_smooth.py) evals skipped in round 5, after
# queue_pf_full.sh: eval/rule_eval.py by position with its own rule (tournament) and
# block mean 8x64 (C3's decode rule), each also with FP4 caches; gates vs its own rule.
# Run from experiments/csa2:
#   setsid nohup ./queue_c3s_evals.sh < /dev/null > logs/queue_c3s_evals.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_c3s_evals.pid
R=$HOME/repos/scratch
PY=$HOME/repos/modded-nanogpt-r91/.venv/bin/python
export TRITON_CACHE_DIR=$HOME/.triton/cache
RUN=9624b171-a72f-43a2-b4c5-223e1b41a764
while kill -0 "$(cat logs/queue_pf_full.pid)" 2>/dev/null; do sleep 30; done

for fp4 in 0 1; do
    for args in tournament "block 8 64 mean"; do
        tag=$(echo $args | sed -E 's/block ([0-9]+) ([0-9]+) (\w+)/block_\3_\1x\2/')
        tag=$( ((fp4)) && echo fp4_)$tag
        out=logs/rule_${RUN:0:8}_$tag
        [ -e $out.csv ] && continue
        CSA2_SELECT=tournament CSA2_REF_FP4=$fp4 EVAL_COMPILE=1 PYTHONPATH=$R/src \
            $PY eval/rule_eval.py $R/experiments/csa2/b/train_csa2_ours_r4_smooth.py \
            $RUN 128 $out.csv $args > $out.log 2>&1
        status=$?
        echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv)"
    done
done
BASE=tournament $PY eval/paired.py ${RUN:0:8} block_mean_8x64
BASE=fp4_tournament $PY eval/paired.py ${RUN:0:8} fp4_block_mean_8x64
echo "$(date +%H:%M) done"
