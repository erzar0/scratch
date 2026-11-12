#!/bin/bash
# B: r91 tournament recipe (run_tournament.sh's env) on train_tournament_kv.py.
# Usage: run_b.sh exact|tournament [SCRIPT]  (smoke: export SWEEP_ITERS etc. first)
set -eu
HERE=$(dirname "$(readlink -f "$0")")
cd $HOME/repos/modded-nanogpt-r91
FORK=$HOME/repos/modded-nanogpt
export DATA_PATH=$FORK PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SWEEP_BATCH_DIV=8 SWEEP_VAL_BATCH=65536 SWEEP_BIGRAM_VOCAB=47160 DISABLE_FP8=1
export SWEEP_ITERS=${SWEEP_ITERS:-10000} SWEEP_EXT_ITERS=${SWEEP_EXT_ITERS:-160}
export SWEEP_VAL_EVERY=${SWEEP_VAL_EVERY:-2000} SAVE_CHECKPOINT=${SAVE_CHECKPOINT:-1}
export PYTHONPATH=$HOME/repos/modded-nanogpt-r91 CSA2_SELECT=$1 TRAIN_SEED=${TRAIN_SEED:-1}
export TRITON_CACHE_DIR=$HOME/.triton/cache
exec $FORK/.venv/bin/torchrun --standalone --nproc_per_node=1 $HERE/${2:-train_tournament_kv.py}
