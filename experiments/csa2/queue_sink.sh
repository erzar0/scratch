#!/bin/bash
# Sink share by layer and position (eval/sink_share.py) for the three runs with a
# learned sink; waits for queue_variants.sh (one GPU job at a time). From
# experiments/csa2:
#   setsid nohup ./queue_sink.sh < /dev/null > logs/queue_sink.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_sink.pid
R=$HOME/repos/scratch
PY=$HOME/repos/modded-nanogpt-r91/.venv/bin/python
export TRITON_CACHE_DIR=$HOME/.triton/cache
while kill -0 "$(cat logs/queue_variants.pid)" 2>/dev/null; do sleep 10; done
for job in "65614cd9-0fdb-4adc-b897-1294ab8caf74 train_csa2_ref.py" \
    "7f71fcf2-d7fc-49e7-9fc8-559c821b4a58 train_csa2_ours.py" \
    "9624b171-a72f-43a2-b4c5-223e1b41a764 train_csa2_ours_r4_smooth.py"; do
    read -r run script <<< "$job"
    out=logs/sink_${run:0:8}
    [ -e $out.csv ] && continue
    CSA2_SELECT=tournament PYTHONPATH=$R/src $PY eval/sink_share.py \
        $R/experiments/csa2/b/$script $run 32 > $out.log 2>&1
    status=$?
    echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.log)"
done
echo "$(date +%H:%M) done"
