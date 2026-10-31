#!/bin/sh
# Sparse decode attentions, stage G part 3: RULER at 50 samples for the hybrid
# tournament (first 3 rounds inside blocks of 1024, then global, keep 512).
set -e
cd /home/kyre/repos/scratch
export TASKS=niah_single_2,niah_multikey_2,niah_multivalue,ruler_vt LIMIT=50 SPARSE_SCORED=128
row=hybrid-1024-3-512
echo "$(date +%H:%M) start $row"
uv run python -m qwen.evaluate long "qwen:$row" > "runs/logs/sparse_ruler50_$row.log" 2>&1
echo "$(date +%H:%M) queue done"
