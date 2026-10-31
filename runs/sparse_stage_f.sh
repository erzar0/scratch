#!/bin/sh
# Sparse decode attentions, stage F: halving inside fixed blocks of keys (one GPU
# program per block, no global sync), against global halving at keep 512.
# Same books as stage A. Logs to runs/logs/sparse_<row>.log.
set -e
cd /home/kyre/repos/scratch
for row in blocks-1024-256 blocks-4096-256 blocks-1024-512 blocks-4096-512 \
    tournament-query-halving-drop-512; do
    echo "$(date +%H:%M) start $row"
    uv run python -m experiments.sparse_qwen pg19 "$row" > "runs/logs/sparse_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
