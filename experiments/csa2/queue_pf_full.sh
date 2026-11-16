#!/bin/bash
# Round 5, step 4: the parameter-free CSA2 block (src/nanogpt/csa2_pf.py) trained in
# the V4.1 layout with tournament picks, after queue_pf_sinks.sh: $SCRIPT
# (b/make_csa2_pf_script.py). Seed 1. Killed if val at step 2000 is NaN or > 5.0;
# then eval/rule_eval.py by position with its own rule, exact, block mean 8x64 and
# 8x256, each also with FP4 caches; gates vs its own rule (eval/paired.py).
# Run from experiments/csa2:
#   SCRIPT=train_csa2_pf_smooth.py setsid nohup ./queue_pf_full.sh < /dev/null \
#       > logs/queue_pf_full.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_pf_full.pid
R=$HOME/repos/scratch
export TRITON_CACHE_DIR=$HOME/.triton/cache
TAG=${SCRIPT%.py}
TAG=${TAG#train_}_tournament
LOG=logs/full_$TAG.log
while kill -0 "$(cat logs/queue_pf_sinks.pid)" 2>/dev/null; do sleep 30; done

TRAIN_SEED=1 setsid b/run_b.sh tournament $SCRIPT > $LOG 2>&1 < /dev/null &
pid=$!
echo $pid > logs/full_$TAG.pid
echo "$(date +%H:%M) $TAG started (pgid $pid)"
while kill -0 $pid 2>/dev/null; do
    v=$(grep -m1 -oE "^step:2000/[0-9]+ val_loss:[^ ]+" $LOG | cut -d: -f3)
    if [[ -n $v ]] && awk -v v="$v" 'BEGIN { exit !(v ~ /nan/ || v + 0 > 5.0) }'; then
        echo "$TAG guard: val $v at step 2000, killed"
        kill -- -$pid
        break
    fi
    sleep 60
done
wait $pid
status=$?
final=$(grep -oE "val_loss:[0-9.na]+" $LOG | tail -n 1 | cut -d: -f2)
echo "$(date +%H:%M) $TAG exit $status, final val $final"
[[ $status == 0 ]] || exit $status
RUN=$(grep -m1 -oE "^logs/[0-9a-f-]{36}\.txt" $LOG | cut -c6-41)
echo "$TAG run $RUN"
for fp4 in 0 1; do
    for args in tournament exact "block 8 64 mean" "block 8 256 mean"; do
        tag=$(echo $args | sed -E 's/block ([0-9]+) ([0-9]+) (\w+)/block_\3_\1x\2/')
        tag=$( ((fp4)) && echo fp4_)$tag
        out=logs/rule_${RUN:0:8}_$tag
        [ -e $out.csv ] && continue
        CSA2_SELECT=tournament CSA2_REF_FP4=$fp4 EVAL_COMPILE=1 PYTHONPATH=$R/src \
            $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/rule_eval.py \
            $R/experiments/csa2/b/$SCRIPT $RUN 128 $out.csv $args > $out.log 2>&1
        status=$?
        echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv)"
    done
done
PY=$HOME/repos/modded-nanogpt-r91/.venv/bin/python
BASE=tournament $PY eval/paired.py ${RUN:0:8} exact block_mean_8x64 block_mean_8x256
BASE=fp4_tournament $PY eval/paired.py ${RUN:0:8} fp4_exact fp4_block_mean_8x64 \
    fp4_block_mean_8x256
echo "$(date +%H:%M) done"
