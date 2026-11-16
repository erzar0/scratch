#!/bin/bash
# Round 4 (plan wondrous-twirling-flurry, phase C): the V4.1 reference port trained
# with our summed-query picks instead of its learned indexer (src/nanogpt/csa2_ours.py,
# b/make_csa2_ours_script.py), seed 1, one run at a time:
#   C1  exact picks        b/train_csa2_ours.py
#   C2  tournament picks   b/train_csa2_ours.py
#   C3  the rule of C1/C2 with the lower eval loss, ratio 4 at layers 1-9
#       (b/train_csa2_ours_r4.py)
# Guard as queue_round3.sh: killed if val at step 2000 is NaN or > 5.0. After each
# run, eval/rule_eval.py by position: its own rule, exact, block mean 8x64, each also
# with TensorRT-LLM's FP4 / FP8 caches (CSA2_REF_FP4=1). Run from experiments/csa2:
#   setsid nohup ./queue_round4.sh < /dev/null > logs/queue_round4.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_round4.pid
R=$HOME/repos/scratch
export TRITON_CACHE_DIR=$HOME/.triton/cache

evals() {  # run script rule
    local run=$1 script=$2 rule=$3 fp4 tag args out status
    for fp4 in 0 1; do
        for args in "$rule" exact "block 8 64 mean"; do
            tag=$(echo $args | sed -E 's/block ([0-9]+) ([0-9]+) (\w+)/block_\3_\1x\2/')
            out=logs/rule_${run:0:8}_$( ((fp4)) && echo fp4_)$tag
            [ -e $out.csv ] && continue
            CSA2_SELECT=$rule CSA2_REF_FP4=$fp4 EVAL_COMPILE=1 PYTHONPATH=$R/src \
                $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/rule_eval.py \
                $R/experiments/csa2/b/$script $run 128 $out.csv $args > $out.log 2>&1
            status=$?
            echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv)"
        done
    done
}

train() {  # tag rule script; sets RUN
    local tag=$1 rule=$2 script=$3 log=logs/full_$1.log pid v status final
    TRAIN_SEED=1 setsid b/run_b.sh $rule $script > $log 2>&1 < /dev/null &
    pid=$!
    echo $pid > logs/full_$tag.pid
    echo "$(date +%H:%M) $tag started ($script, $rule, seed 1, pgid $pid)"
    while kill -0 $pid 2>/dev/null; do
        v=$(grep -m1 -oE "^step:2000/[0-9]+ val_loss:[^ ]+" $log | cut -d: -f3)
        if [[ -n $v ]] && awk -v v="$v" 'BEGIN { exit !(v ~ /nan/ || v + 0 > 5.0) }'
        then
            echo "$tag guard: val $v at step 2000, killed"
            kill -- -$pid
            break
        fi
        sleep 60
    done
    wait $pid
    status=$?
    final=$(grep -oE "val_loss:[0-9.na]+" $log | tail -n 1 | cut -d: -f2)
    echo "$(date +%H:%M) $tag exit $status, final val $final"
    RUN=$(grep -m1 -oE "^logs/[0-9a-f-]{36}\.txt" $log | cut -c6-41)
    echo "$tag run $RUN"
    [[ $status == 0 ]] && evals $RUN $script $rule
    return $status
}

loss() {  # eval loss over all positions of a run's rule
    awk -F, '$2 == 0 && $3 == 8192 { print $5 }' logs/rule_${1:0:8}_$2.csv 2>/dev/null
}

train csa2_ours_exact exact train_csa2_ours.py
c1=$RUN
train csa2_ours_tournament tournament train_csa2_ours.py
c2=$RUN
a=$(loss $c1 exact) b=$(loss $c2 tournament)
best=exact
[[ -n $b ]] && { [[ -z $a ]] || awk -v a="$a" -v b="$b" 'BEGIN { exit !(b < a) }'; } \
    && best=tournament
echo "$(date +%H:%M) C1 exact $a, C2 tournament $b: C3 uses $best"
train csa2_ours_r4_$best $best train_csa2_ours_r4.py
echo "$(date +%H:%M) done"
