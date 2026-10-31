#!/bin/sh
# Sparse decode attentions, stage E: RULER at equal reads. SparQ with 16 dimensions
# (the tournament's reads at keep 128) and the tournament at keep 256 (which beat
# SparQ with 32 dimensions on PG-19). Starts when PID $1 (stage D) exits.
set -e
cd /home/kyre/repos/scratch
while kill -0 "$1" 2>/dev/null; do sleep 30; done
export TASKS=niah_single_2,niah_multikey_2,niah_multivalue,ruler_vt LIMIT=25 SPARSE_SCORED=128
for row in sparq-16-128 tournament-query-halving-drop-256; do
    echo "$(date +%H:%M) start $row"
    uv run python -m qwen.evaluate long "qwen:$row" > "runs/logs/sparse_ruler_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
