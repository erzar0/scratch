#!/bin/bash
# After queue_mqa2.sh: fused-merge decode test, per-kernel times (8K, 512K), one-program
# tensor-core attention variants, parts sweep. Run from experiments/csa2 with setsid.
set -uo pipefail
HERE=$(pwd)
R=$HOME/repos/scratch
PY=$HOME/repos/modded-nanogpt/.venv/bin/python
SP=/tmp/claude-1000/-home-kyre-repos-scratch/cf6eb213-4baa-40a1-b9e3-9da952735db5/scratchpad
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_mqa2.pid)" 2>/dev/null; do sleep 30; done
cd $R
uv run python -m tests.nanogpt.test_csa2_decode > $HERE/logs/test_csa2_decode.log 2>&1 \
    || { echo "test failed"; exit 1; }
for n in 8192 524288; do
    PYTHONPATH=src $PY $SP/kernel_times.py $n > $HERE/logs/kernel_times_$n.log 2>&1
done
PYTHONPATH=src $PY $SP/attend_mma.py > $HERE/logs/attend_mma.log 2>&1
PYTHONPATH=src $PY experiments/csa2/bench/decode_parts.py > $HERE/logs/decode_parts.log 2>&1
echo done
