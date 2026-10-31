#!/bin/sh
# Throwaway Qwen queue, part 2 (after the probe): the dense control's remaining
# evaluations and the speed benchmarks. Dropped: long_reduced. Deferred to the
# fixed CSA2-Qwen: route, lm_route.
# Status lines to stdout; each step's log to runs/logs/qwen_<step>.log.
set -e
cd /home/kyre/repos/scratch
run() {
    name=$1
    shift
    echo "$(date +%H:%M) start $name"
    "$@" > "runs/logs/qwen_$name.log" 2>&1
}
run long_dense uv run python -m qwen.evaluate long dense
run short_dense uv run python -m qwen.evaluate short dense
run bench_layer uv run python -m qwen.bench layer
run bench_model uv run python -m qwen.bench model
echo "$(date +%H:%M) queue done"
