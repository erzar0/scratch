#!/bin/sh
# Sparse decode attentions, stage K: global halving reading K in fixed groups of
# 8 dims (groups-*, src/sparse/group_order.py), groups ordered by |q|; rotary
# pairs vs dims as stored. Logs to runs/logs/sparse_<row>.log.
set -e
cd /home/kyre/repos/scratch
for row in groups-pairs-8-drop-512 groups-contiguous-8-drop-512; do
    echo "$(date +%H:%M) start $row"
    uv run python -m experiments.sparse_qwen pg19 "$row" > "runs/logs/sparse_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
