#!/bin/sh
# Throwaway Qwen queue, part 3 (replaces part 2 mid-run): wait for long_dense
# (python PID 812879, started by part 2), probe CSA2 layers 3 and 7 trained alone
# as Full layers, then the rest of part 2.
# Status lines to stdout; each step's log to runs/logs/qwen_<step>.log.
set -e
cd /home/kyre/repos/scratch
run() {
    name=$1
    shift
    echo "$(date +%H:%M) start $name"
    "$@" > "runs/logs/qwen_$name.log" 2>&1
}
while kill -0 812879 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M) long_dense finished"
run probe_full_3_7 env LAYERS=3,7 uv run python -m qwen.probe layer base
run short_dense uv run python -m qwen.evaluate short dense
run bench_layer uv run python -m qwen.bench layer
run bench_model uv run python -m qwen.bench model
echo "$(date +%H:%M) queue done"
