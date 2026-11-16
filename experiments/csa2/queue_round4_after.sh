#!/bin/bash
# Round 4 phase B, after queue_round4.sh (the 3 training runs and their evals):
# the fused pick kernel's test (bf16 and FP8 caches), the decode bench with the
# reference layout (bench/decode_ref_picks.py: one-launch block-mean picks, FP8 rows,
# kernels per token), then where C1's training time goes (make_profile_script.py;
# steps 6, 16, 26 of a 30-step schedule) -> logs/profile_csa2_ours.csv.
# Run from experiments/csa2:
#   setsid nohup ./queue_round4_after.sh < /dev/null > logs/queue_round4_after.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_round4_after.pid
HERE=$(pwd) R=$HOME/repos/scratch
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_round4.pid)" 2>/dev/null; do sleep 30; done
(cd $R && uv run python -m tests.nanogpt.test_csa2_pick_fused) \
    > logs/test_csa2_pick_fused.log 2>&1
status=$?
echo "$(date +%H:%M) test_csa2_pick_fused exit $status"
(cd $R && PYTHONPATH=src:experiments/csa2/bench \
    $HOME/repos/modded-nanogpt/.venv/bin/python experiments/csa2/bench/decode_ref_picks.py) \
    > logs/decode_ref_picks.log 2>&1
status=$?
echo "$(date +%H:%M) decode_ref_picks exit $status"
export SWEEP_ITERS=30 SWEEP_EXT_ITERS=2 SWEEP_VAL_EVERY=0 SAVE_CHECKPOINT=0
export PROFILE_AT=6,16,26
PROFILE_OUT=$HERE/logs/profile_csa2_ours.csv b/run_b.sh exact \
    train_csa2_ours_profile.py > logs/profile_csa2_ours.log 2>&1
status=$?
echo "$(date +%H:%M) profile csa2_ours exit $status"
echo "$(date +%H:%M) done"
