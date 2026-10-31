#!/bin/sh
# Sparse decode attentions, stage I: the user's sort-based variant. Global halving
# with a 16-dim first round (first-16-*), and losers kept with approximate scores
# (approx) vs dropped, keeps 128/256/512; first-8 rows already exist as
# tournament-query-halving-*. Then RULER at 50 samples for the keep-512 rows of
# both new ideas. Logs to runs/logs/sparse_<row>.log and sparse_ruler50_<row>.log.
set -e
cd /home/kyre/repos/scratch
for row in tournament-query-halving-approx-256 tournament-query-halving-approx-512 \
    first-16-drop-128 first-16-drop-256 first-16-drop-512 \
    first-16-approx-128 first-16-approx-256 first-16-approx-512; do
    echo "$(date +%H:%M) start $row"
    uv run python -m experiments.sparse_qwen pg19 "$row" > "runs/logs/sparse_$row.log" 2>&1
done
for row in tournament-query-halving-approx-512 first-16-drop-512; do
    echo "$(date +%H:%M) start RULER $row"
    TASKS=niah_single_2,niah_multikey_2,niah_multivalue,ruler_vt LIMIT=50 SPARSE_SCORED=128 \
        uv run python -m qwen.evaluate long "qwen:$row" > "runs/logs/sparse_ruler50_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
