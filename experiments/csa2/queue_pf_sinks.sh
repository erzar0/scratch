#!/bin/bash
# Qwen3-0.6B-Base under csa2_pf.drop_in with sink tokens (each row's first 4 tokens
# kept exact), after queue_pf_checks.sh: m 1/4 x top 64/512 (eval/qwen_swap_eval.py),
# gated against its own attention. Run from experiments/csa2:
#   setsid nohup ./queue_pf_sinks.sh < /dev/null > logs/queue_pf_sinks.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_pf_sinks.pid
R=$HOME/repos/scratch
while kill -0 "$(cat logs/queue_pf_checks.pid)" 2>/dev/null; do sleep 15; done
for job in "1 64" "1 512" "4 64" "4 512"; do
    read -r m k <<< "$job"
    out=logs/rule_qwen06b_pf_m${m}_k${k}_s4
    [ -e $out.csv ] && continue
    (cd $R && HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run python \
        experiments/csa2/eval/qwen_swap_eval.py 32 experiments/csa2/$out.csv $m $k 4) \
        > $out.log 2>&1
    status=$?
    echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv)"
done
BASE=own $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/paired.py qwen06b \
    pf_m1_k64_s4 pf_m1_k512_s4 pf_m4_k64_s4 pf_m4_k512_s4
echo "$(date +%H:%M) done"
