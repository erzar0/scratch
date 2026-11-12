#!/bin/bash
# Retrains run d3c62030 from this folder (b/run_b.sh's env, exact picks, seed 1).
# Needs ~/repos/modded-nanogpt (venv, data) and ~/repos/modded-nanogpt-r91 (cwd: the
# run's log and checkpoint land in its logs/). Smoke: export SWEEP_ITERS=30 first.
set -eu
HERE=$(dirname "$(readlink -f "$0")")
cd $HOME/repos/modded-nanogpt-r91
FORK=$HOME/repos/modded-nanogpt
export DATA_PATH=$FORK PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SWEEP_BATCH_DIV=8 SWEEP_VAL_BATCH=65536 SWEEP_BIGRAM_VOCAB=47160 DISABLE_FP8=1
export SWEEP_ITERS=${SWEEP_ITERS:-10000} SWEEP_EXT_ITERS=${SWEEP_EXT_ITERS:-160}
export SWEEP_VAL_EVERY=${SWEEP_VAL_EVERY:-2000} SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-1}
export CSA2_SELECT=exact TRAIN_SEED=${TRAIN_SEED:-1}
export SCRATCH_SRC=$HERE/src PYTHONPATH=$HERE:$HOME/repos/modded-nanogpt-r91
export TRITON_CACHE_DIR=$HOME/.triton/cache
exec $FORK/.venv/bin/torchrun --standalone --nproc_per_node=1 \
    $HERE/train_tournament_kv_win1_fast.py
