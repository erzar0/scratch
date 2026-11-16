#!/bin/bash
# V4.1 reference checkpoint 65614cd9 (b/train_csa2_ref.py) under eval rules
# (eval/ref_rule_eval.py; eval only): its own indexer or our summed-query picks
# (exact; block-mean pool 1 = 64 blocks of 8, pool 4 = 256 = V4.1's candidate
# blocks, pool 128 = every block, must equal exact), with the env knobs YARN,
# YARN_TEMP, CSA2_REF_FP4, SHARE, TOPK. A job whose csv exists is skipped.
# Run from experiments/csa2:
#   setsid nohup ./queue_ref_rules.sh < /dev/null > logs/queue_ref_rules.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_ref_rules.pid
R=$HOME/repos/scratch
RUN=65614cd9-0fdb-4adc-b897-1294ab8caf74
export TRITON_CACHE_DIR=$HOME/.triton/cache
# tag | env | rule
JOBS="summed_exact||exact
summed_block1||block 1
summed_block4||block 4
summed_block128||block 128
indexer_yarn4|YARN=4|own
indexer_yarn4t|YARN=4 YARN_TEMP=1|own
summed_exact_yarn4|YARN=4|exact
summed_exact_yarn4t|YARN=4 YARN_TEMP=1|exact
summed_exact_fp4|CSA2_REF_FP4=1|exact
summed_block1_fp4|CSA2_REF_FP4=1|block 1
summed_exact_share|SHARE=1|exact
summed_block1_share|SHARE=1|block 1
summed_exact_top128|TOPK=128|exact"
while IFS="|" read -r tag vars rule; do
    out=logs/rule_${RUN:0:8}_$tag
    [ -e $out.csv ] && continue
    env $vars EVAL_COMPILE=1 PYTHONPATH=$R/src \
        $HOME/repos/modded-nanogpt-r91/.venv/bin/python eval/ref_rule_eval.py \
        $R/experiments/csa2/b/train_csa2_ref.py $RUN 128 $out.csv $rule > $out.log 2>&1
    status=$?
    echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv 2>/dev/null)"
done <<< "$JOBS"
echo "$(date +%H:%M) done"
