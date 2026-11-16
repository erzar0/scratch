#!/bin/bash
# After queue_head_clip.sh: every other rule_table.py row with the lm_head clipped as
# in training (eval/variant_eval.py HEAD_CLIP; eval only), same rules and knobs as
# the unclipped evals, bf16 / FP8 / FP4 caches -> logs/rule_<id8>_[fp8_|fp4_]hclip_
# <tag>.csv. A job whose csv exists is skipped. Run from experiments/csa2:
#   setsid nohup ./queue_head_clip_rules.sh < /dev/null > logs/queue_head_clip_rules.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_head_clip_rules.pid
R=$HOME/repos/scratch
PY=$HOME/repos/modded-nanogpt-r91/.venv/bin/python
export TRITON_CACHE_DIR=$HOME/.triton/cache
REF="65614cd9-0fdb-4adc-b897-1294ab8caf74|train_csa2_ref.py|ref_rule_eval.py"
C2="7f71fcf2-d7fc-49e7-9fc8-559c821b4a58|train_csa2_ours.py|rule_eval.py"
C3S="9624b171-a72f-43a2-b4c5-223e1b41a764|train_csa2_ours_r4_smooth.py|rule_eval.py"
PF="fa3176f4-4264-4efe-accb-b49d7c9d7d9c|train_csa2_pf_smooth.py|rule_eval.py"
SINK="5be6bb81-dfbf-4690-a598-e1422bbbe1e1|train_csa2_pf_sink.py|rule_eval.py"
B=block_mean_8x
# run | training script | eval script | tag | knobs (FP8/FP4 extra after +) | rule
JOBS="$REF|indexer||own
$C2|tournament||tournament
$C2|exact||exact
$C2|${B}256||block 8 256 mean
$C2|${B}64||block 8 64 mean
$C3S|tournament||tournament
$C3S|exact||exact
$C3S|${B}64||block 8 64 mean
$PF|tournament||tournament
$PF|exact||exact
$PF|${B}256||block 8 256 mean
$PF|${B}64||block 8 64 mean
$PF|share9_${B}256|SHARE9=1+MEANS_QUANT=1|block 8 256 mean
$SINK|tournament||tournament
$SINK|exact||exact
$SINK|${B}256||block 8 256 mean
$SINK|${B}64||block 8 64 mean"
for fmt in bf16 fp8 fp4; do
    while IFS="|" read -r run script ev tag knobs rule; do
        p=$([[ $fmt == bf16 ]] && echo "" || echo ${fmt}_)
        out=logs/rule_${run:0:8}_${p}hclip_$tag
        [ -e $out.csv ] && continue
        k=${knobs%%+*}
        [[ $fmt != bf16 && $knobs == *+* ]] && k="$k ${knobs#*+}"
        args=(eval/variant_eval.py eval/$ev $R/experiments/csa2/b/$script $run 128
              $out.csv $rule)
        [[ $fmt == fp8 ]] && args=(eval/fp8_eval.py "${args[@]}")
        env $k HEAD_CLIP=1 CSA2_SELECT=tournament \
            CSA2_REF_FP4=$([[ $fmt == fp4 ]] && echo 1 || echo 0) \
            EVAL_COMPILE=1 PYTHONPATH=$R/src $PY "${args[@]}" > $out.log 2>&1
        status=$?
        echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv 2>/dev/null)"
    done <<< "$JOBS"
done
echo "$(date +%H:%M) done"
