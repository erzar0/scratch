#!/bin/sh
# Sparse decode attentions, stage C: RULER (4 tasks, first 25 samples per length),
# the sparse method on the last 128 queries. Dense = the existing qwen rows of
# runs/qwen/longctx.csv. Starts when stage B part 2 (PID $1) exits.
set -e
cd /home/kyre/repos/scratch
while kill -0 "$1" 2>/dev/null; do sleep 30; done
export TASKS=niah_single_2,niah_multikey_2,niah_multivalue,ruler_vt LIMIT=25 SPARSE_SCORED=128
for row in exact-128 sparq-32-128 loki-post-32-128 \
    tournament-random-bracket-approx-128 tournament-query-halving-drop-128 \
    tournament-query-halving-drop-128-16; do
    echo "$(date +%H:%M) start $row"
    uv run python -m qwen.evaluate long "qwen:$row" > "runs/logs/sparse_ruler_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
