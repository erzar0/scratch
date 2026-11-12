#!/bin/bash
# Where training time goes, by kernel and phase (make_profile_script.py): after
# queue_decode5.sh, one profiled step per stage (6, 16, 26 of a 30-step schedule:
# same batch and sequence shapes as the full run) for CSA2, A tournament, B exact,
# B tournament and B routed channels -> logs/profile_<version>.csv.
# Run from experiments/csa2 with setsid; logs in logs/.
set -uo pipefail
HERE=$(pwd)
R91=$HOME/repos/modded-nanogpt-r91
export TRITON_CACHE_DIR=$HOME/.triton/cache
export SWEEP_ITERS=30 SWEEP_EXT_ITERS=2 SWEEP_VAL_EVERY=0 SAVE_CHECKPOINT=0
export PROFILE_AT=6,16,26
while kill -0 "$(cat logs/queue_decode5.pid)" 2>/dev/null; do sleep 30; done
csa2() {  # run_csa2.sh's environment on the profiled copy of train_csa2.py
    (cd $R91 && DATA_PATH=$HOME/repos/modded-nanogpt \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SWEEP_BATCH_DIV=8 \
        SWEEP_VAL_BATCH=65536 SWEEP_BIGRAM_VOCAB=47160 DISABLE_FP8=1 TRAIN_SEED=1 \
        PYTHONPATH=$R91 SCRATCH_SRC=$HERE/a/csa2_src/src PROFILE_OUT=$HERE/logs/profile_$1.csv \
        .venv/bin/torchrun --standalone --nproc_per_node=1 $HERE/b/train_csa2_profile.py)
}
csa2 csa2 > logs/profile_csa2.log 2>&1
echo "csa2 exit $?"
CSA2_SELECT=tournament CSA2_FAST=1 csa2 a_tournament > logs/profile_a_tournament.log 2>&1
echo "a_tournament exit $?"
for v in exact tournament; do
    PROFILE_OUT=$HERE/logs/profile_b_$v.csv b/run_b.sh $v train_tournament_kv_fast_profile.py \
        > logs/profile_b_$v.log 2>&1
    echo "b_$v exit $?"
done
PROFILE_OUT=$HERE/logs/profile_b_route.csv b/run_b.sh exact \
    train_tournament_kv_route_fast_profile.py > logs/profile_b_route.log 2>&1
echo "b_route exit $?"
echo done
