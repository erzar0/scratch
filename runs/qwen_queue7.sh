#!/bin/sh
# Throwaway queue, part 7 (part 6 ran the probe's base variant; its "all" variant
# ran out of memory, now one backward per layer): the "all" variant, then part 3's
# remaining Qwen dense-control steps.
# Status lines to stdout; each step's log to runs/logs/<step>.log.
set -e
cd /home/kyre/repos/scratch
R91=/home/kyre/repos/modded-nanogpt-r91
CKPT=$R91/logs/ec348eea-fe95-43b8-b0d4-98b06c489b76/state_step010160.pt
run() {
    name=$1
    shift
    echo "$(date +%H:%M) start $name"
    "$@" > "runs/logs/$name.log" 2>&1
}
run nanogpt_probe_all env PYTHONPATH=src $R91/.venv/bin/python -m nanogpt.probe "$CKPT" all
run qwen_short_dense uv run python -m qwen.evaluate short dense
run qwen_bench_layer uv run python -m qwen.bench layer
run qwen_bench_model uv run python -m qwen.bench model
echo "$(date +%H:%M) queue done"
