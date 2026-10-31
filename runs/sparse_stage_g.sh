#!/bin/sh
# Sparse decode attentions, stage G: RULER at 50 samples per task and length for
# SparQ with 32 dimensions and the tournament at keep 256 (their 25-sample tie hid
# a 16K gap). Replaces their 25-sample rows. Starts when PID $1 (stage F) exits.
set -e
cd /home/kyre/repos/scratch
while kill -0 "$1" 2>/dev/null; do sleep 30; done
export TASKS=niah_single_2,niah_multikey_2,niah_multivalue,ruler_vt LIMIT=50 SPARSE_SCORED=128
for row in sparq-32-128 tournament-query-halving-drop-256; do
    echo "$(date +%H:%M) start $row"
    uv run python -m qwen.evaluate long "qwen:$row" > "runs/logs/sparse_ruler50_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
