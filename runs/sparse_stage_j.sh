#!/bin/sh
# Sparse decode attentions, stage J: global halving with sampled-median cutoffs
# (sampled-*, against the exact rows tournament-query-halving-drop-512 and
# first-16-*-512), and losers whose weight goes to their mean value (meanv).
# Then RULER at 50 samples for first-16 mean-V and approximate losers.
# Logs to runs/logs/sparse_<row>.log and sparse_ruler50_<row>.log.
set -e
cd /home/kyre/repos/scratch
for row in sampled-8-drop-512 first-8-meanv-512 first-16-meanv-512 first-16-meanv-128 \
    sampled-16-drop-512 sampled-16-approx-512; do
    echo "$(date +%H:%M) start $row"
    uv run python -m experiments.sparse_qwen pg19 "$row" > "runs/logs/sparse_$row.log" 2>&1
done
for row in first-16-meanv-512 first-16-approx-512; do
    echo "$(date +%H:%M) start RULER $row"
    TASKS=niah_single_2,niah_multikey_2,niah_multivalue,ruler_vt LIMIT=50 SPARSE_SCORED=128 \
        uv run python -m qwen.evaluate long "qwen:$row" > "runs/logs/sparse_ruler50_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
