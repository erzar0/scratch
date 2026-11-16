#!/bin/bash
# Round 6: the parameter-free block with a fixed sink ln 128 (src/nanogpt/csa2_pf_sink.py,
# b/train_csa2_pf_sink.py from b/make_csa2_pf_sink_script.py), tournament picks.
# 30-step smoke (stops the queue on failure), then the full run, seed 1, killed if
# val at step 2000 is NaN or > 5.0; then eval/rule_eval.py by position with its own
# rule, exact, block mean 8x64 and 8x256, each with bf16, FP8 (eval/fp8_eval.py) and
# FP4 caches; sink share (eval/sink_share.py); gates vs its own rule per format
# (eval/paired.py). Run from experiments/csa2:
#   setsid nohup ./queue_pf_sink.sh < /dev/null > logs/queue_pf_sink.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_pf_sink.pid
R=$HOME/repos/scratch
PY=$HOME/repos/modded-nanogpt-r91/.venv/bin/python
export TRITON_CACHE_DIR=$HOME/.triton/cache
SCRIPT=train_csa2_pf_sink.py
TAG=csa2_pf_sink_tournament
LOG=logs/full_$TAG.log

env SWEEP_ITERS=30 SWEEP_EXT_ITERS=2 SWEEP_VAL_EVERY=0 SAVE_CHECKPOINT=0 \
    b/run_b.sh tournament $SCRIPT > logs/smoke_$TAG.log 2>&1 < /dev/null
status=$?
echo "$(date +%H:%M) smoke exit $status: $(grep -oE "step:30/[0-9]+ [^ ]+ [^ ]+" \
    logs/smoke_$TAG.log | tail -n 1)"
[[ $status == 0 ]] || exit $status

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

for fmt in bf16 fp8 fp4; do
    for rule in tournament exact "block 8 64 mean" "block 8 256 mean"; do
        tag=$(echo $rule | sed -E 's/block ([0-9]+) ([0-9]+) (\w+)/block_\3_\1x\2/')
        [[ $fmt != bf16 ]] && tag=${fmt}_$tag
        out=logs/rule_${RUN:0:8}_$tag
        [ -e $out.csv ] && continue
        args=(eval/rule_eval.py $R/experiments/csa2/b/$SCRIPT $RUN 128 $out.csv $rule)
        [[ $fmt == fp8 ]] && args=(eval/fp8_eval.py "${args[@]}")
        CSA2_SELECT=tournament CSA2_REF_FP4=$([[ $fmt == fp4 ]] && echo 1 || echo 0) \
            EVAL_COMPILE=1 PYTHONPATH=$R/src $PY "${args[@]}" > $out.log 2>&1
        status=$?
        echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv 2>/dev/null)"
    done
done
CSA2_SELECT=tournament PYTHONPATH=$R/src $PY eval/sink_share.py \
    $R/experiments/csa2/b/$SCRIPT $RUN 32 > logs/sink_${RUN:0:8}.log 2>&1
echo "$(date +%H:%M) sink share exit $?"

for p in "" fp8_ fp4_; do
    BASE=${p}tournament $PY eval/paired.py ${RUN:0:8} ${p}exact ${p}block_mean_8x64 \
        ${p}block_mean_8x256
done
echo "$(date +%H:%M) done"
