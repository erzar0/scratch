#!/bin/sh
# Sparse decode attentions, stage N: tournaments on keys stored in a random
# order (src/sparse/shuffled.py): halving in blocks of 1024 and 4096, and the
# bracket, beside the bracket in position order at the same keep. Logs to
# runs/logs/sparse_<row>.log.
set -e
cd /home/kyre/repos/scratch
for row in shuffled-blocks-1024-512 shuffled-blocks-4096-512 \
    shuffled-tournament-query-bracket-drop-512 tournament-query-bracket-drop-512; do
    echo "$(date +%H:%M) start $row"
    uv run python -m experiments.sparse_qwen pg19 "$row" > "runs/logs/sparse_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
