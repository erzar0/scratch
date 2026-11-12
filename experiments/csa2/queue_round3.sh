#!/bin/bash
# Round 3 (plan wondrous-twirling-flurry): the CSA2 reference baseline (DeepSeek
# V4.1-Flash-1017 model.py layers, b/train_csa2_ref.py, seed 1). Guard as
# queue_round2.sh: killed if val at step 2000 is NaN or > 5.0. Then validation by
# position with its own indexer: bf16 caches, then TensorRT-LLM's FP4 / FP8 caches
# (CSA2_REF_FP4=1). Run from experiments/csa2:
#   setsid nohup ./queue_round3.sh < /dev/null > logs/queue_round3.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_round3.pid
R=$HOME/repos/scratch
TAG=csa2_ref SCRIPT=train_csa2_ref.py
export TRITON_CACHE_DIR=$HOME/.triton/cache
log=logs/full_$TAG.log
TRAIN_SEED=1 setsid b/run_b.sh exact $SCRIPT > $log 2>&1 < /dev/null &
pid=$!
echo $pid > logs/full_$TAG.pid
echo "$(date +%H:%M) $TAG started ($SCRIPT, seed 1, pgid $pid)"
while kill -0 $pid 2>/dev/null; do
    v=$(grep -m1 -oE "^step:2000/[0-9]+ val_loss:[^ ]+" $log | cut -d: -f3)
    if [[ -n $v ]] && awk -v v="$v" 'BEGIN { exit !(v ~ /nan/ || v + 0 > 5.0) }'; then
        echo "$TAG guard: val $v at step 2000, killed"
        kill -- -$pid
        break
    fi
    sleep 60
done
wait $pid
status=$?
final=$(grep -oE "val_loss:[0-9.na]+" $log | tail -n 1 | cut -d: -f2)
echo "$(date +%H:%M) $TAG exit $status, final val $final"
[[ $status == 0 ]] || exit 1
run=$(grep -m1 -oE "^logs/[0-9a-f-]{36}\.txt" $log | cut -c6-41)
echo "$TAG run $run"
for fp4 in 0 1; do
    out=logs/rule_${run:0:8}_indexer$( ((fp4)) && echo _fp4)
    CSA2_REF_FP4=$fp4 EVAL_COMPILE=1 PYTHONPATH=$R/src \
        $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/packed_val.py \
        $R/experiments/csa2/b/$SCRIPT $run 128 $out.csv > $out.log 2>&1
    status=$?
    echo "$(date +%H:%M) $out eval exit $status: $(tail -n 1 $out.csv)"
done
echo "$(date +%H:%M) done"
