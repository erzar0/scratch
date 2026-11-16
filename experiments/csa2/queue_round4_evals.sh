#!/bin/bash
# Round 4, after queue_round4_after.sh: the tiled exact-pick kernel's test and timing
# (tests/nanogpt/test_csa2_select_tiled.py), then decode rules with more candidates
# than block mean 8x64 (which fails the gate on C1 and C2) on C2 (tournament, the
# reference layout) and C3 (tournament, ratio 4): blocks of 16 and 32 entries, 128
# and 256 blocks. eval/rule_eval.py by position -> logs/rule_<id8>_<tag>.csv; gate
# with eval/paired.py (BASE=tournament). Run from experiments/csa2:
#   setsid nohup ./queue_round4_evals.sh < /dev/null > logs/queue_round4_evals.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_round4_evals.pid
R=$HOME/repos/scratch
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_round4_after.pid)" 2>/dev/null; do sleep 30; done
(cd $R && uv run python -m tests.nanogpt.test_csa2_select_tiled) \
    > logs/test_csa2_select_tiled.log 2>&1
status=$?
echo "$(date +%H:%M) test_csa2_select_tiled exit $status"
C2=7f71fcf2-d7fc-49e7-9fc8-559c821b4a58
C3=$(grep -m1 -oE "^logs/[0-9a-f-]{36}\.txt" logs/full_csa2_ours_r4_tournament.log | cut -c6-41)
for job in "$C2 train_csa2_ours.py" "$C3 train_csa2_ours_r4.py"; do
    read -r run script <<< "$job"
    for args in "block 16 64 mean" "block 32 64 mean" "block 8 128 mean" \
        "block 8 256 mean"; do
        tag=$(echo $args | sed -E 's/block ([0-9]+) ([0-9]+) (\w+)/block_\3_\1x\2/')
        out=logs/rule_${run:0:8}_$tag
        [ -e $out.csv ] && continue
        CSA2_SELECT=tournament EVAL_COMPILE=1 PYTHONPATH=$R/src \
            $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/rule_eval.py \
            $R/experiments/csa2/b/$script $run 128 $out.csv $args > $out.log 2>&1
        status=$?
        echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv)"
    done
done
echo "$(date +%H:%M) done"
