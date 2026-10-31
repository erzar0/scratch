#!/bin/sh
# Sparse decode attentions, stage B part 1: the tournament with SparQ's local window,
# the PCA rows again on the 32k-calibrated basis, and the baselines' budgets.
# Same books as stage A. Status lines to stdout; logs to runs/logs/sparse_<row>.log.
set -e
cd /home/kyre/repos/scratch
for row in tournament-query-halving-drop-128-8-16 \
    tournament-pca-halving-drop-128 tournament-pca-halving-mean-128 \
    sparq-16-128 sparq-64-128 sparq-32-64 sparq-16-256 \
    exact-64 exact-256 \
    loki-post-16-128 loki-post-64-128; do
    echo "$(date +%H:%M) start $row"
    uv run python -m experiments.sparse_qwen pg19 "$row" > "runs/logs/sparse_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
