#!/bin/bash
# After the window-32 training (pid $1): decode attention test, one-kernel hist
# select (scratchpad, correctness + timing), parts sweep, then A tournament (CSA2's
# model and window, the indexer swapped for our tournament picks).
# Run from experiments/csa2 with setsid; logs in logs/.
set -uo pipefail
HERE=$(pwd)
R=$HOME/repos/scratch
PY=$HOME/repos/modded-nanogpt/.venv/bin/python
SP=/tmp/claude-1000/-home-kyre-repos-scratch/cf6eb213-4baa-40a1-b9e3-9da952735db5/scratchpad
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$1" 2>/dev/null; do sleep 30; done
cd $R
uv run python -m tests.nanogpt.test_csa2_decode > $HERE/logs/test_csa2_decode.log 2>&1
echo "test exit $?"
PYTHONPATH=src timeout 1200 $PY $SP/select_hist.py > $HERE/logs/select_hist.log 2>&1
echo "select exit $?"
PYTHONPATH=src $PY experiments/csa2/bench/decode_parts.py > $HERE/logs/decode_parts3.log 2>&1
echo "parts exit $?"
cd $HERE
./queue_a_tournament.sh
echo done
