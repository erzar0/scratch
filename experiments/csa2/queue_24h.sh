#!/bin/bash
# 24 h plan, phase 2 (plan wondrous-twirling-flurry): full B exact runs with CSA2's
# K=V window (1, 2 or 6 shared vectors per window token) and picks every 4 layers /
# routed, one at a time, each followed by per-position validation with exact picks.
# Guard: a run is killed (its process group) if val at step 2000 is NaN or > 5.0
# (B exact 4.71, CSA2 4.84); the queue goes on. Run from experiments/csa2:
#   setsid nohup bash -c 'echo $$ > logs/queue_24h.pid; exec ./queue_24h.sh' \
#       > logs/queue_24h.log 2>&1 < /dev/null &
set -uo pipefail
HERE=$(pwd)
R=$HOME/repos/scratch
export TRITON_CACHE_DIR=$HOME/.triton/cache

final() {  # tag -> last val loss of its log
    grep -oE "val_loss:[0-9.na]+" logs/full_$1.log | tail -n 1 | cut -d: -f2
}

full() {  # tag script [seed]; a tag whose run finished before is skipped (resume)
    local log=logs/full_$1.log v
    if grep -q "^step:10160/10160 val_loss" $log 2>/dev/null; then
        echo "$1 finished before, skipped"
        return 0
    fi
    TRAIN_SEED=${3:-1} setsid b/run_b.sh exact $2 > $log 2>&1 < /dev/null &
    local pid=$!
    echo $pid > logs/full_$1.pid
    echo "$(date +%H:%M) $1 started ($2, seed ${3:-1}, pgid $pid)"
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
        $HERE/b/$2 $run 128 logs/rule_${run:0:8}_exact.csv exact \
        > logs/rule_${run:0:8}_exact.log 2>&1
    echo "$1 eval exit $?"
}

better() {  # the window (1 or 2) with the lower final val; 1 if neither finished
    awk -v a="$(final win1)" -v b="$(final win2)" \
        'BEGIN { print (b != "" && b !~ /nan/ && (a == "" || a ~ /nan/ || b + 0 < a + 0)) ? 2 : 1 }'
}

full win1 train_tournament_kv_win1_fast.py
full route_pick4 train_tournament_kv_route_pick4_fast.py
full win2 train_tournament_kv_win2_fast.py
full pick4 train_tournament_kv_pick4_fast.py
G=$(better)
echo "better window: $G"
full route_win${G}_pick4 train_tournament_kv_route_win${G}_pick4_fast.py
full win6 train_tournament_kv_win6_fast.py
full win${G}_pick4 train_tournament_kv_win${G}_pick4_fast.py
full route_win${G}_pick4_seed2 train_tournament_kv_route_win${G}_pick4_fast.py 2
echo "$(date +%H:%M) done"
