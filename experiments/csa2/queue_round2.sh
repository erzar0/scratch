#!/bin/bash
# Round 2 (plan wondrous-twirling-flurry): full B exact runs, one at a time, each
# followed by per-position validation with exact picks. Guard as queue_24h.sh: a run is
# killed (its process group) if val at step 2000 is NaN or > 5.0. Args: triples
# TAG SCRIPT SEED (script in b/). Run from experiments/csa2:
#   setsid nohup ./queue_round2.sh TAG SCRIPT SEED ... < /dev/null > logs/queue_round2_<x>.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_round2.pid
R=$HOME/repos/scratch
export TRITON_CACHE_DIR=$HOME/.triton/cache

final() {  # tag -> last val loss of its log
    grep -oE "val_loss:[0-9.na]+" logs/full_$1.log | tail -n 1 | cut -d: -f2
}

full() {  # tag script seed; a tag whose run finished before is skipped (resume)
    local log=logs/full_$1.log v
    if grep -q "^step:10160/10160 val_loss" $log 2>/dev/null; then
        echo "$1 finished before, skipped"
        return 0
    fi
    TRAIN_SEED=$3 setsid b/run_b.sh exact $2 > $log 2>&1 < /dev/null &
    local pid=$!
    echo $pid > logs/full_$1.pid
    echo "$(date +%H:%M) $1 started ($2, seed $3, pgid $pid)"
    while kill -0 $pid 2>/dev/null; do
        v=$(grep -m1 -oE "^step:2000/[0-9]+ val_loss:[^ ]+" $log | cut -d: -f3)
        if [[ -n $v ]] && awk -v v="$v" 'BEGIN { exit !(v ~ /nan/ || v + 0 > 5.0) }'; then
            echo "$1 guard: val $v at step 2000, killed"
            kill -- -$pid
            break
        fi
        sleep 60
    done
    wait $pid
    local status=$?
    echo "$(date +%H:%M) $1 exit $status, final val $(final $1)"
    [[ $status == 0 ]] || return 1
    local run
    run=$(grep -m1 -oE "^logs/[0-9a-f-]{36}\.txt" $log | cut -c6-41)
    echo "$1 run $run"
    CSA2_SELECT=exact EVAL_COMPILE=1 PYTHONPATH=$R/src \
        $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/rule_eval.py \
        $R/experiments/csa2/b/$2 $run 128 logs/rule_${run:0:8}_exact.csv exact \
        > logs/rule_${run:0:8}_exact.log 2>&1
    echo "$(date +%H:%M) $1 eval exit $? $(grep ',0,8192,' logs/rule_${run:0:8}_exact.csv | tail -1)"
}

while (($# >= 3)); do
    full "$1" "$2" "$3"
    shift 3
done
echo "$(date +%H:%M) done"
