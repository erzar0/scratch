#!/bin/sh
# Sparse decode attentions, stage A: references, then the tournament ablation ladder
# (keep 128, 8 dimensions per round), 16 PG-19 books at 32k, last 1024 positions.
# Status lines to stdout; each row's log to runs/logs/sparse_<row>.log.
set -e
cd /home/kyre/repos/scratch
for row in dense exact-128 sparq-32-128 loki-pre-32-128 loki-post-32-128 \
    tournament-random-bracket-approx-128 \
    tournament-random-bracket-drop-128 \
    tournament-random-bracket-mean-128 \
    tournament-random-halving-drop-128 tournament-random-halving-mean-128 \
    tournament-hadamard-halving-drop-128 tournament-hadamard-halving-mean-128 \
    tournament-query-halving-drop-128 tournament-query-halving-mean-128 \
    tournament-pca-halving-drop-128 tournament-pca-halving-mean-128 \
    tournament-query-bracket-drop-128; do
    echo "$(date +%H:%M) start $row"
    uv run python -m experiments.sparse_qwen pg19 "$row" > "runs/logs/sparse_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
