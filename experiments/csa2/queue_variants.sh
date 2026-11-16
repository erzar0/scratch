#!/bin/bash
# Decode variants of trained runs, no training (eval/variant_eval.py knobs):
# the tournament-picks run without in-window picks (WINDOW_PICKS); the parameter-free
# run with layer 9 reusing layer 5's picks (SHARE9) under tournament and best 256
# blocks, bf16 / FP8 / FP4 caches; block means rounded as the entry cache
# (MEANS_QUANT, FP8 and FP4). Then gates vs each run's own rule in the same cache
# format (eval/paired.py), then the decode bench (bench/decode_pf_share.py). A job
# whose csv exists is skipped. Run from experiments/csa2:
#   setsid nohup ./queue_variants.sh < /dev/null > logs/queue_variants.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_variants.pid
R=$HOME/repos/scratch
PY=$HOME/repos/modded-nanogpt-r91/.venv/bin/python
export TRITON_CACHE_DIR=$HOME/.triton/cache

C2=7f71fcf2-d7fc-49e7-9fc8-559c821b4a58
PF=fa3176f4-4264-4efe-accb-b49d7c9d7d9c
B=block_mean_8x256
# tag | run | training script | format (bf16, fp4, fp8) | knobs | rule
JOBS="window_tournament|$C2|train_csa2_ours.py|bf16|WINDOW_PICKS=1|tournament
share9_tournament|$PF|train_csa2_pf_smooth.py|bf16|SHARE9=1|tournament
share9_$B|$PF|train_csa2_pf_smooth.py|bf16|SHARE9=1|block 8 256 mean
fp8_share9_tournament|$PF|train_csa2_pf_smooth.py|fp8|SHARE9=1|tournament
fp8_share9_$B|$PF|train_csa2_pf_smooth.py|fp8|SHARE9=1 MEANS_QUANT=1|block 8 256 mean
fp4_share9_tournament|$PF|train_csa2_pf_smooth.py|fp4|SHARE9=1|tournament
fp4_share9_$B|$PF|train_csa2_pf_smooth.py|fp4|SHARE9=1 MEANS_QUANT=1|block 8 256 mean
fp8_${B}_qmeans|$PF|train_csa2_pf_smooth.py|fp8|MEANS_QUANT=1|block 8 256 mean
fp4_${B}_qmeans|$PF|train_csa2_pf_smooth.py|fp4|MEANS_QUANT=1|block 8 256 mean"
while IFS="|" read -r tag run script fmt knobs rule; do
    out=logs/rule_${run:0:8}_$tag
    [ -e $out.csv ] && continue
    args=(eval/variant_eval.py eval/rule_eval.py $R/experiments/csa2/b/$script $run 128
          $out.csv $rule)
    [[ $fmt == fp8 ]] && args=(eval/fp8_eval.py "${args[@]}")
    env $knobs CSA2_SELECT=tournament \
        CSA2_REF_FP4=$([[ $fmt == fp4 ]] && echo 1 || echo 0) \
        EVAL_COMPILE=1 PYTHONPATH=$R/src $PY "${args[@]}" > $out.log 2>&1
    status=$?
    echo "$(date +%H:%M) $out exit $status: $(tail -n 1 $out.csv 2>/dev/null)"
done <<< "$JOBS"

BASE=tournament $PY eval/paired.py ${C2:0:8} window_tournament
BASE=tournament $PY eval/paired.py ${PF:0:8} share9_tournament share9_$B
BASE=fp8_tournament $PY eval/paired.py ${PF:0:8} fp8_share9_tournament fp8_share9_$B \
    fp8_${B}_qmeans
BASE=fp4_tournament $PY eval/paired.py ${PF:0:8} fp4_share9_tournament fp4_share9_$B \
    fp4_${B}_qmeans

(cd $R && PYTHONPATH=src:experiments/csa2/bench \
    $HOME/repos/modded-nanogpt/.venv/bin/python experiments/csa2/bench/decode_pf_share.py \
    > experiments/csa2/logs/decode_pf_share.log 2>&1)
status=$?
echo "$(date +%H:%M) decode bench exit $status: $(tail -n 1 logs/decode_pf_share.log)"
echo "$(date +%H:%M) done"
