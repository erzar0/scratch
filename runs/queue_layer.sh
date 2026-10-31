#!/usr/bin/env bash
# The layer bench (bench/layer.py) after the eval queue (runs/queue_eval.sh) exits.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
while [ -e /proc/127692 ]; do sleep 60; done
echo "$(date +%T) start bench_layer"
if uv run python -m bench.layer > runs/logs/bench_layer.log 2>&1; then
    echo "$(date +%T) done bench_layer"
else
    echo "$(date +%T) FAILED bench_layer"
fi
