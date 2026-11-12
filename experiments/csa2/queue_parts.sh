#!/bin/bash
# After queue_window.sh: decode time by part. Run from experiments/csa2 with setsid.
set -uo pipefail
R=$HOME/repos/scratch
HERE=$(pwd)
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_window.pid)" 2>/dev/null; do sleep 20; done
cd $R
PYTHONPATH=src $HOME/repos/modded-nanogpt/.venv/bin/python \
    experiments/csa2/bench/decode_parts.py > $HERE/logs/decode_parts.log 2>&1
echo done
