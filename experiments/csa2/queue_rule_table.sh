#!/bin/bash
# Missing cells of the model x decode-rule table (rule_table.py), no training: the
# decode bench (bench/decode_ref_picks.py, FP8 and two-launch rows), C3-smooth exact
# (bf16, FP4), C2 FP4 best 256 blocks, every rule with FP8 caches (eval/fp8_eval.py);
# then gates vs each run's own rule in the same cache format (eval/paired.py). A job
# whose csv exists is skipped. Run from experiments/csa2:
#   setsid nohup ./queue_rule_table.sh < /dev/null > logs/queue_rule_table.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_rule_table.pid
R=$HOME/repos/scratch
PY=$HOME/repos/modded-nanogpt-r91/.venv/bin/python
export TRITON_CACHE_DIR=$HOME/.triton/cache

(cd $R && PYTHONPATH=src:experiments/csa2/bench \
    $HOME/repos/modded-nanogpt/.venv/bin/python experiments/csa2/bench/decode_ref_picks.py \
    > experiments/csa2/logs/decode_ref_picks.log 2>&1)
status=$?
echo "$(date +%H:%M) decode bench exit $status: $(tail -n 1 logs/decode_ref_picks.log)"

REF=65614cd9-0fdb-4adc-b897-1294ab8caf74
C2=7f71fcf2-d7fc-49e7-9fc8-559c821b4a58
C3S=9624b171-a72f-43a2-b4c5-223e1b41a764
PF=fa3176f4-4264-4efe-accb-b49d7c9d7d9c
# tag | run | training script | format (bf16, fp4, fp8) | rule
JOBS="indexer_fp8|$REF|train_csa2_ref.py|fp8|own
exact|$C3S|train_csa2_ours_r4_smooth.py|bf16|exact
fp4_exact|$C3S|train_csa2_ours_r4_smooth.py|fp4|exact
fp4_block_mean_8x256|$C2|train_csa2_ours.py|fp4|block 8 256 mean"
for job in "$C2 train_csa2_ours.py" "$C3S train_csa2_ours_r4_smooth.py" \
    "$PF train_csa2_pf_smooth.py"; do
    read -r run script <<< "$job"
    for rule in tournament exact "block 8 256 mean" "block 8 64 mean"; do
        [[ $run == "$C3S" && $rule == "block 8 256 mean" ]] && continue
        tag=fp8_$(echo $rule | sed -E 's/block ([0-9]+) ([0-9]+) (\w+)/block_\3_\1x\2/')
        JOBS+=$'\n'"$tag|$run|$script|fp8|$rule"
    done
done
while IFS="|" read -r tag run script fmt rule; do
    out=logs/rule_${run:0:8}_$tag
    [ -e $out.csv ] && continue
    if [[ $rule == own ]]; then
        args=(eval/packed_val.py $R/experiments/csa2/b/$script $run 128 $out.csv)
    else
        args=(eval/rule_eval.py $R/experiments/csa2/b/$script $run 128 $out.csv $rule)
    fi
    [[ $fmt == fp8 ]] && args=(eval/fp8_eval.py "${args[@]}")
    CSA2_SELECT=tournament CSA2_REF_FP4=$([[ $fmt == fp4 ]] && echo 1 || echo 0) \
        EVAL_COMPILE=1 PYTHONPATH=$R/src $PY "${args[@]}" > $out.log 2>&1
    status=$?
    echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv 2>/dev/null)"
done <<< "$JOBS"

BASE=tournament $PY eval/paired.py ${C3S:0:8} exact
BASE=fp4_tournament $PY eval/paired.py ${C3S:0:8} fp4_exact
BASE=fp4_tournament $PY eval/paired.py ${C2:0:8} fp4_block_mean_8x256
for run in $C2 $C3S $PF; do
    tags="fp8_exact fp8_block_mean_8x64"
    [[ $run != "$C3S" ]] && tags+=" fp8_block_mean_8x256"
    BASE=fp8_tournament $PY eval/paired.py ${run:0:8} $tags
done
echo "$(date +%H:%M) done"
