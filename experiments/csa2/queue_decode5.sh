#!/bin/bash
# After queue_a_tournament.sh: one-kernel select v2 (correctness + timing), then the
# decode bench's check (decode picks = training picks). Run from experiments/csa2.
set -uo pipefail
HERE=$(pwd)
PY=$HOME/repos/modded-nanogpt/.venv/bin/python
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_a_tournament.pid)" 2>/dev/null; do sleep 30; done
cd $HOME/repos/scratch
PYTHONPATH=src timeout 1200 $PY experiments/csa2/bench/select_one_kernel.py \
    > $HERE/logs/select_one_kernel.log 2>&1
echo "select exit $?"
PYTHONPATH=src $PY experiments/csa2/bench/decode_csa2.py check > $HERE/logs/decode_check.log 2>&1
echo "check exit $?"
echo done
