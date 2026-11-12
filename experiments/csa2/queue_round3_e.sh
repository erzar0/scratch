#!/bin/bash
# Round 3: decode bench (bench/decode_block.py, after the window kernel's dtype fix),
# then the reference baseline's profile with one host sync per indexer call
# (csa2_ref._picks) -> logs/profile_csa2_ref_sync.csv. Run from experiments/csa2:
#   setsid nohup ./queue_round3_e.sh < /dev/null > logs/queue_round3_e.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_round3_e.pid
HERE=$(pwd)
export TRITON_CACHE_DIR=$HOME/.triton/cache
(cd $HOME/repos/scratch && PYTHONPATH=src:experiments/csa2/bench \
    $HOME/repos/modded-nanogpt/.venv/bin/python experiments/csa2/bench/decode_block.py) \
    > logs/decode_block.log 2>&1
status=$?
echo "$(date +%H:%M) decode_block exit $status"
export SWEEP_ITERS=30 SWEEP_EXT_ITERS=2 SWEEP_VAL_EVERY=0 SAVE_CHECKPOINT=0
export PROFILE_AT=6,16,26
PROFILE_OUT=$HERE/logs/profile_csa2_ref_sync.csv b/run_b.sh exact \
    train_csa2_ref_profile.py > logs/profile_csa2_ref_sync.log 2>&1
status=$?
echo "$(date +%H:%M) profile csa2_ref_sync exit $status"
echo "$(date +%H:%M) done"
