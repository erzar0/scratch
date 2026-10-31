#!/bin/sh
# Throwaway queue, part 4 (replaces part 3; its layers 3/7 probe is done): record 91
# dense pretraining (../modded-nanogpt-r91), then part 3's remaining dense-control steps.
# Status lines to stdout; each step's log to runs/logs/<step>.log.
set -e
cd /home/kyre/repos/scratch
run() {
    name=$1
    shift
    echo "$(date +%H:%M) start $name"
    "$@" > "runs/logs/$name.log" 2>&1
}
run nanogpt_r91 sh /home/kyre/repos/modded-nanogpt-r91/run_r91.sh
run qwen_short_dense uv run python -m qwen.evaluate short dense
run qwen_bench_layer uv run python -m qwen.bench layer
run qwen_bench_model uv run python -m qwen.bench model
echo "$(date +%H:%M) queue done"
