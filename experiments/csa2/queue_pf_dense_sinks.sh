#!/bin/bash
# Dense nanogpt checkpoint 9497acb4 with each attention replaced by the parameter-free
# block (eval/pf_swap_eval.py) and the first 4 tokens of each document kept exact
# (sink tokens), after queue_c3s_evals.sh: m 1 top 64, m 2/4 x top 64/512; gates vs
# its own attention (eval/paired.py). Run from experiments/csa2:
#   setsid nohup ./queue_pf_dense_sinks.sh < /dev/null > logs/queue_pf_dense_sinks.log \
#       2>&1 &
set -uo pipefail
echo $$ > logs/queue_pf_dense_sinks.pid
R=$HOME/repos/scratch
PY=$HOME/repos/modded-nanogpt-r91/.venv/bin/python
export TRITON_CACHE_DIR=$HOME/.triton/cache
DENSE=9497acb4-9da5-4e9e-82f8-5a4111777428
while kill -0 "$(cat logs/queue_c3s_evals.pid)" 2>/dev/null; do sleep 30; done

for job in "1 64" "2 64" "2 512" "4 64" "4 512"; do
    read -r m k <<< "$job"
    out=logs/rule_${DENSE:0:8}_pf_m${m}_k${k}_s4
    [ -e $out.csv ] && continue
    PYTHONPATH=$R/src $PY eval/pf_swap_eval.py train_dense_stripped.py $DENSE 128 \
        $out.csv $m $k 4 > $out.log 2>&1
    status=$?
    echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv)"
done
BASE=dense $PY eval/paired.py ${DENSE:0:8} pf_m1_k64_s4 pf_m2_k64_s4 pf_m2_k512_s4 \
    pf_m4_k64_s4 pf_m4_k512_s4
echo "$(date +%H:%M) done"
