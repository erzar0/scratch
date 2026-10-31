#!/bin/sh
# Throwaway queue, part 5 (replaces part 4 mid-run): wait for record 91 pretraining
# (torchrun PID 849575, started by part 4), the nanogpt CSA2 layer probe on its
# checkpoint, then part 3's remaining Qwen dense-control steps.
# Status lines to stdout; each step's log to runs/logs/<step>.log.
set -e
cd /home/kyre/repos/scratch
R91=/home/kyre/repos/modded-nanogpt-r91
run() {
    name=$1
    shift
    echo "$(date +%H:%M) start $name"
    "$@" > "runs/logs/$name.log" 2>&1
}
while kill -0 849575 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M) nanogpt_r91 finished"
CKPT=$(ls $R91/logs/ec348eea-fe95-43b8-b0d4-98b06c489b76/state_step*.pt | tail -1)
run nanogpt_probe env PYTHONPATH=src $R91/.venv/bin/python -m nanogpt.probe "$CKPT"
run qwen_short_dense uv run python -m qwen.evaluate short dense
run qwen_bench_layer uv run python -m qwen.bench layer
run qwen_bench_model uv run python -m qwen.bench model
echo "$(date +%H:%M) queue done"
