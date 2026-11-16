#!/bin/bash
# Round 5, step 1: C3 (the V4.1 reference with our tournament picks, ratio 4 at layers
# 1-9) retrained with the stage-switch fix: b/train_csa2_ours_r4_smooth.py
# (b/make_csa2_ours_script.py r4_smooth; the learning rate ramps across stage starts,
# train loss logged every 50 steps). Seed 1. Guard and evals as queue_round4.sh: killed
# if val at step 2000 is NaN or > 5.0; then eval/rule_eval.py by position with its own
# rule, exact and block mean 8x64, each also with FP4 caches. Run from experiments/csa2:
#   setsid nohup ./queue_round5.sh < /dev/null > logs/queue_round5.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_round5.pid
R=$HOME/repos/scratch
export TRITON_CACHE_DIR=$HOME/.triton/cache
TAG=csa2_ours_r4_smooth_tournament SCRIPT=train_csa2_ours_r4_smooth.py
LOG=logs/full_$TAG.log

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
    for args in tournament exact "block 8 64 mean"; do
        tag=$(echo $args | sed -E 's/block ([0-9]+) ([0-9]+) (\w+)/block_\3_\1x\2/')
        out=logs/rule_${RUN:0:8}_$( ((fp4)) && echo fp4_)$tag
        [ -e $out.csv ] && continue
        CSA2_SELECT=tournament CSA2_REF_FP4=$fp4 EVAL_COMPILE=1 PYTHONPATH=$R/src \
            $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/rule_eval.py \
            $R/experiments/csa2/b/$SCRIPT $RUN 128 $out.csv $args > $out.log 2>&1
        status=$?
        echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv)"
    done
done
echo "$(date +%H:%M) done"
