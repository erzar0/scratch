#!/bin/sh
# Sparse decode attentions, stage B part 2: the best tournament (|q| order, halving,
# losers dropped) at keep 64 and 256, with 16 dimensions per round (as many reads as
# SparQ with 32), and with the original loser rule. Starts when part 1 (PID $1) exits.
set -e
cd /home/kyre/repos/scratch
while kill -0 "$1" 2>/dev/null; do sleep 30; done
for row in tournament-query-halving-drop-64 tournament-query-halving-drop-256 \
    tournament-query-halving-drop-128-16 tournament-query-halving-approx-128; do
    echo "$(date +%H:%M) start $row"
    uv run python -m experiments.sparse_qwen pg19 "$row" > "runs/logs/sparse_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
