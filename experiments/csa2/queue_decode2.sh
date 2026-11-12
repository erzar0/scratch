#!/bin/bash
# After queue_window.sh: decode test and bench again (two-level merge). Run from
# experiments/csa2 with setsid; logs in logs/.
set -uo pipefail
HERE=$(pwd)
R=$HOME/repos/scratch
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_window.pid)" 2>/dev/null; do sleep 20; done
cd $R
uv run python -m tests.nanogpt.test_csa2_decode > $HERE/logs/test_csa2_decode.log 2>&1 \
    || exit 1
PYTHONPATH=src $HOME/repos/modded-nanogpt/.venv/bin/python \
    experiments/csa2/bench/decode_csa2.py runs/nanogpt/long_context/csa2_decode.csv \
    > $HERE/logs/decode_bench.log 2>&1
echo done
