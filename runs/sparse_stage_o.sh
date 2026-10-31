#!/bin/sh
# Sparse decode attentions, stage O: RULER at 50 samples for the tournament
# halving inside blocks of 1024 keys stored in random slots (equal to global
# halving on PG-19). Logs to runs/logs/sparse_ruler50_<row>.log.
set -e
cd /home/kyre/repos/scratch
for row in shuffled-blocks-1024-512; do
    echo "$(date +%H:%M) start RULER $row"
    TASKS=niah_single_2,niah_multikey_2,niah_multivalue,ruler_vt LIMIT=50 SPARSE_SCORED=128 \
        uv run python -m qwen.evaluate long "qwen:$row" > "runs/logs/sparse_ruler50_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
