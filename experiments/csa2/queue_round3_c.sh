#!/bin/bash
# Round 3 phase C, after queue_round3.sh (the CSA2 reference baseline's run and
# evals): decode bench with the reference layout (bench/decode_block.py), then where
# training time goes (make_profile_script.py; steps 6, 16, 26 of a 30-step schedule,
# one per stage) for the reference baseline and run 7 -> logs/profile_<version>.csv.
# Run from experiments/csa2:
#   setsid nohup ./queue_round3_c.sh < /dev/null > logs/queue_round3_c.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_round3_c.pid
HERE=$(pwd)
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_round3.pid)" 2>/dev/null; do sleep 30; done
(cd $HOME/repos/scratch && PYTHONPATH=src:experiments/csa2/bench \
    $HOME/repos/modded-nanogpt/.venv/bin/python experiments/csa2/bench/decode_block.py) \
    > logs/decode_block.log 2>&1
status=$?
echo "$(date +%H:%M) decode_block exit $status"
export SWEEP_ITERS=30 SWEEP_EXT_ITERS=2 SWEEP_VAL_EVERY=0 SAVE_CHECKPOINT=0
export PROFILE_AT=6,16,26
for v in "csa2_ref train_csa2_ref_profile.py" \
    "b_win1_pick4 train_tournament_kv_win1_pick4_fast_profile.py"; do
    set -- $v
    PROFILE_OUT=$HERE/logs/profile_$1.csv b/run_b.sh exact $2 > logs/profile_$1.log 2>&1
    status=$?
    echo "$(date +%H:%M) profile $1 exit $status"
done
echo "$(date +%H:%M) done"
