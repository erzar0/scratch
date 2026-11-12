#!/bin/bash
# Where the GPU idles (make_profile_script.py timeline mode, profile_gaps.py): after
# queue_route.sh, Chrome traces of steps 6, 7 (stage 1) and 26, 27 (stage 3) of a
# 30-step schedule; 6 and 26 with CPU ops and Python stacks, 7 and 27 GPU only.
# Dense stripped record 91, B exact, B tournament and CSA2 -> logs/trace_<version>/,
# summary in logs/profile_gaps.log. Run from experiments/csa2 with setsid.
set -uo pipefail
HERE=$(pwd)
R91=$HOME/repos/modded-nanogpt-r91
export TRITON_CACHE_DIR=$HOME/.triton/cache
export SWEEP_ITERS=30 SWEEP_EXT_ITERS=2 SWEEP_VAL_EVERY=0 SAVE_CHECKPOINT=0
export PROFILE_AT=6,7,26,27 PROFILE_CPU_AT=6,26 PROFILE_OUT=/dev/null
while kill -0 "$(cat logs/queue_route.pid)" 2>/dev/null; do sleep 30; done
trace() {  # version, rule, script
    mkdir -p logs/trace_$1
    PROFILE_TRACE=$HERE/logs/trace_$1 b/run_b.sh $2 $3 > logs/trace_$1.log 2>&1
    echo "$1 exit $?"
}
trace dense_stripped exact train_dense_stripped_profile.py
trace b_exact exact train_tournament_kv_fast_profile.py
trace b_tournament tournament train_tournament_kv_fast_profile.py
mkdir -p logs/trace_csa2
(cd $R91 && DATA_PATH=$HOME/repos/modded-nanogpt \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SWEEP_BATCH_DIV=8 \
    SWEEP_VAL_BATCH=65536 SWEEP_BIGRAM_VOCAB=47160 DISABLE_FP8=1 TRAIN_SEED=1 \
    PYTHONPATH=$R91 SCRATCH_SRC=$HERE/a/csa2_src/src PROFILE_TRACE=$HERE/logs/trace_csa2 \
    .venv/bin/torchrun --standalone --nproc_per_node=1 $HERE/b/train_csa2_profile.py) \
    > logs/trace_csa2.log 2>&1
echo "csa2 exit $?"
uv run python profile_gaps.py logs/trace_dense_stripped logs/trace_b_exact \
    logs/trace_b_tournament logs/trace_csa2 > logs/profile_gaps.log 2>&1
echo done
