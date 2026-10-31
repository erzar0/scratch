#!/bin/sh
# Sparse decode attentions, stage F part 2: the first 1-4 rounds halve inside blocks
# of 1024 keys, the rest globally (0 rounds = global, 6 = blocks-1024-512, both run).
# Same books as stage A. Logs to runs/logs/sparse_<row>.log.
set -e
cd /home/kyre/repos/scratch
for rounds in 1 2 3 4; do
    row=hybrid-1024-$rounds-512
    echo "$(date +%H:%M) start $row"
    uv run python -m experiments.sparse_qwen pg19 "$row" > "runs/logs/sparse_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
