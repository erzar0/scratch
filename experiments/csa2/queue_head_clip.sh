#!/bin/bash
# Every scoreboard model's own-rule eval again with the lm_head weights clipped to
# the training head's FP8 range (eval/variant_eval.py HEAD_CLIP; eval only): bf16 and
# FP4 caches -> logs/rule_<id8>_[fp4_]hclip_<rule>.csv; then each run's change vs
# its unclipped eval (eval/paired.py). A job whose csv exists is skipped. Run from
# experiments/csa2:
#   setsid nohup ./queue_head_clip.sh < /dev/null > logs/queue_head_clip.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_head_clip.pid
R=$HOME/repos/scratch
PY=$HOME/repos/modded-nanogpt-r91/.venv/bin/python
export TRITON_CACHE_DIR=$HOME/.triton/cache
# run | training script | eval script | rule (csv tag) | rule args
JOBS="65614cd9-0fdb-4adc-b897-1294ab8caf74|train_csa2_ref.py|ref_rule_eval.py|indexer|own
8ee7cd76-cdd7-4e9e-8fc5-356b9a9cc5ec|train_csa2_ours.py|rule_eval.py|exact|exact
7f71fcf2-d7fc-49e7-9fc8-559c821b4a58|train_csa2_ours.py|rule_eval.py|tournament|tournament
87d9531c-3007-4568-ac42-855982849471|train_csa2_ours_r4.py|rule_eval.py|tournament|tournament
9624b171-a72f-43a2-b4c5-223e1b41a764|train_csa2_ours_r4_smooth.py|rule_eval.py|tournament|tournament
fa3176f4-4264-4efe-accb-b49d7c9d7d9c|train_csa2_pf_smooth.py|rule_eval.py|tournament|tournament
5be6bb81-dfbf-4690-a598-e1422bbbe1e1|train_csa2_pf_sink.py|rule_eval.py|tournament|tournament"
for fmt in bf16 fp4; do
    while IFS="|" read -r run script ev tag rule; do
        p=$([[ $fmt == fp4 ]] && echo fp4_ || echo "")
        out=logs/rule_${run:0:8}_${p}hclip_$tag
        [ -e $out.csv ] && continue
        HEAD_CLIP=1 CSA2_SELECT=tournament CSA2_REF_FP4=$([[ $fmt == fp4 ]] && echo 1 || echo 0) \
            EVAL_COMPILE=1 PYTHONPATH=$R/src $PY eval/variant_eval.py eval/$ev \
            $R/experiments/csa2/b/$script $run 128 $out.csv $rule > $out.log 2>&1
        status=$?
        echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv 2>/dev/null)"
    done <<< "$JOBS"
done
while IFS="|" read -r run script ev tag rule; do
    BASE=$tag $PY eval/paired.py ${run:0:8} hclip_$tag
    base=fp4_$tag  # the reference's FP4 eval is tagged indexer_fp4
    [ -e logs/rule_${run:0:8}_$base.csv ] || base=${tag}_fp4
    BASE=$base $PY eval/paired.py ${run:0:8} fp4_hclip_$tag
done <<< "$JOBS"
echo "$(date +%H:%M) done"
