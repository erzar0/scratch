#!/bin/sh
# Sparse decode attentions, stage D: SparQ and the main tournament rows again, now
# counting K sectors per layout for both (attend.k_sectors). Same books as stage A.
# Starts when stage C (PID $1) exits.
set -e
cd /home/kyre/repos/scratch
while kill -0 "$1" 2>/dev/null; do sleep 30; done
for row in sparq-16-128 sparq-32-128 sparq-64-128 sparq-32-64 sparq-16-256 \
    tournament-query-halving-drop-64 tournament-query-halving-drop-128 \
    tournament-query-halving-drop-256; do
    echo "$(date +%H:%M) start $row"
    uv run python -m experiments.sparse_qwen pg19 "$row" > "runs/logs/sparse_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
