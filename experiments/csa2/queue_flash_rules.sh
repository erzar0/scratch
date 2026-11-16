#!/bin/bash
# Flash-structure eval rules on the parameter-free checkpoint (eval only, lm_head
# clipped as in training, bf16 caches) -> logs/rule_fa3176f4_hclip_<tag>.csv, then
# each rule's change vs the trained tournament picks (and the reuse rules vs the
# block rule they reuse). A job whose csv exists is skipped. Run from experiments/csa2:
#   setsid nohup ./queue_flash_rules.sh < /dev/null > logs/queue_flash_rules.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_flash_rules.pid
R=$HOME/repos/scratch
PY=$HOME/repos/modded-nanogpt-r91/.venv/bin/python
export TRITON_CACHE_DIR=$HOME/.triton/cache
RUN=fa3176f4-4264-4efe-accb-b49d7c9d7d9c
SCRIPT=$R/experiments/csa2/b/train_csa2_pf_smooth.py
# tag | rule
JOBS="reuse9_block_mean_8x256|reuse 8 256 block
flash_structure|reuse 8 256 exact
head_max|head_max
stale8_block_mean_8x256|stale 8 256 8"
while IFS="|" read -r tag rule; do
    out=logs/rule_${RUN:0:8}_hclip_$tag
    [ -e $out.csv ] && continue
    HEAD_CLIP=1 CSA2_SELECT=tournament CSA2_REF_FP4=0 EVAL_COMPILE=1 \
        PYTHONPATH=$R/src $PY eval/variant_eval.py eval/rule_eval.py $SCRIPT $RUN \
        128 $out.csv $rule > $out.log 2>&1
    status=$?
    echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv 2>/dev/null)"
done <<< "$JOBS"
echo "change vs tournament picks (trained rule):"
while IFS="|" read -r tag rule; do
    BASE=hclip_tournament $PY eval/paired.py ${RUN:0:8} hclip_$tag
done <<< "$JOBS"
echo "change vs best 256 blocks of 8 at every pick layer:"
for tag in reuse9_block_mean_8x256 stale8_block_mean_8x256; do
    BASE=hclip_block_mean_8x256 $PY eval/paired.py ${RUN:0:8} hclip_$tag
done
echo "$(date +%H:%M) done"
