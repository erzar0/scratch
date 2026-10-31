#!/bin/sh
# Sparse decode attentions, stage G part 2: RULER at 50 samples for the best stage F
# row under SparQ 32's reads (global halving, keep 512). Starts when PID $1 exits.
set -e
cd /home/kyre/repos/scratch
while kill -0 "$1" 2>/dev/null; do sleep 30; done
export TASKS=niah_single_2,niah_multikey_2,niah_multivalue,ruler_vt LIMIT=50 SPARSE_SCORED=128
row=tournament-query-halving-drop-512
echo "$(date +%H:%M) start $row"
uv run python -m qwen.evaluate long "qwen:$row" > "runs/logs/sparse_ruler50_$row.log" 2>&1
echo "$(date +%H:%M) queue done"
