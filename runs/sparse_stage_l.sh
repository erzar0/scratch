#!/bin/sh
# Sparse decode attentions, stage L: SparQ with wider top-k, for loss against
# time beside the tournament (plots/sparse_pareto.py). Logs to
# runs/logs/sparse_<row>.log.
set -e
cd /home/kyre/repos/scratch
for row in sparq-32-256 sparq-32-512 sparq-64-256 sparq-64-512; do
    echo "$(date +%H:%M) start $row"
    uv run python -m experiments.sparse_qwen pg19 "$row" > "runs/logs/sparse_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
