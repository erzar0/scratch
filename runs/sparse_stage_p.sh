#!/bin/sh
# Sparse decode attentions, stage P: bound bandit (src/sparse/bound_bandit.py),
# |q| channel order, keep 512. Exact bound with the radius scaled by 1/4, 1/8
# and 1/16; Gaussian bound at 2, 1 and 1/2 standard deviations. Book 0 smoke:
# the full exact bound picks the same keys as 1/4 of it, so it is left out; 1/16
# already loses 0.21 nats on book 0, so 1/64 is left out. Waits for a running
# row given as $1 (a PID); launch with setsid nohup. Logs to runs/logs/sparse_<row>.log.
set -e
cd /home/kyre/repos/scratch
while [ -n "$1" ] && kill -0 "$1" 2>/dev/null; do sleep 30; done
for row in bandit-exact-0.125-512 bandit-gauss-1-512 bandit-gauss-0.5-512 \
    bandit-exact-0.25-512 bandit-gauss-2-512 bandit-exact-0.0625-512; do
    echo "$(date +%H:%M) start $row"
    uv run python -m experiments.sparse_qwen pg19 "$row" > "runs/logs/sparse_$row.log" 2>&1
done
echo "$(date +%H:%M) queue done"
