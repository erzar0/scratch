#!/bin/bash
# Parameter-free CSA2 (src/nanogpt/csa2_pf.py) checks, after C3-smooth's training:
# 1 unit tests; 2 the V4.1 reference checkpoint 65614cd9 with its compressor, indexer
# and sink dropped (eval/swap_eval.py into b/train_csa2_pf.py): exact picks in bf16
# and FP4, block mean 8x64; 3 the dense checkpoint 9497acb4 with each attention
# replaced by the block (eval/pf_swap_eval.py): its own attention, then m 1/2/4 x
# top 64/256/512; 4 Qwen3-0.6B-Base (eval/qwen_swap_eval.py): its own, m 1 top 8192
# (= exact attention), m 1/4 x top 64/512; 5 synthetic suite row
# deepseek-v4.1/csa2-parameter-free (latency, fidelity); 6 decode bench
# (bench/decode_ref_picks.py, PF rows); 7 a 30-step smoke of b/train_csa2_pf_smooth.py.
# Gates: eval/paired.py (BASE = the model's own attention). Run from experiments/csa2:
#   setsid nohup ./queue_pf_checks.sh < /dev/null > logs/queue_pf_checks.log 2>&1 &
set -uo pipefail
echo $$ > logs/queue_pf_checks.pid
R=$HOME/repos/scratch
PY=$HOME/repos/modded-nanogpt-r91/.venv/bin/python
export TRITON_CACHE_DIR=$HOME/.triton/cache
REF=65614cd9-0fdb-4adc-b897-1294ab8caf74
DENSE=9497acb4-9da5-4e9e-82f8-5a4111777428
while kill -0 "$(cat logs/full_csa2_ours_r4_smooth_tournament.pid)" 2>/dev/null; do
    sleep 30
done

run() {  # run NAME COMMAND...: logs/NAME.log, one status line
    local name=$1
    shift
    "$@" > logs/$name.log 2>&1
    local status=$?
    echo "$(date +%H:%M) $name exit $status: $(tail -n 1 logs/$name.log)"
}

# 1 unit tests
run test_csa2_pf bash -c "cd $R && uv run python -m tests.nanogpt.test_csa2_pf"
run test_pf_registry bash -c "cd $R && uv run pytest -q tests/attention/test_registry.py \
    tests/attention/test_correctness.py -k parameter-free"

# 2 reference checkpoint, compressor + indexer + sink dropped
for job in "0 exact pf_exact" "1 exact pf_exact_fp4" "0 block_8_64_mean pf_block_mean_8x64"; do
    read -r fp4 rule tag <<< "$job"
    out=logs/rule_${REF:0:8}_$tag
    [ -e $out.csv ] && continue
    run rule_${REF:0:8}_$tag env CSA2_SELECT=exact CSA2_REF_FP4=$fp4 EVAL_COMPILE=1 \
        PYTHONPATH=$R/src $PY eval/swap_eval.py $R/experiments/csa2/b/train_csa2_pf.py \
        $REF 128 $out.csv ${rule//_/ }
done

# 3 dense checkpoint, each attention replaced (eager: the block loops over chunks)
out=logs/rule_${DENSE:0:8}_dense
[ -e $out.csv ] || run rule_${DENSE:0:8}_dense env PYTHONPATH=$R/src \
    $PY eval/packed_val.py train_dense_stripped.py $DENSE 128 $out.csv
for m in 1 2 4; do
    for k in 64 256 512; do
        out=logs/rule_${DENSE:0:8}_pf_m${m}_k$k
        [ -e $out.csv ] && continue
        run rule_${DENSE:0:8}_pf_m${m}_k$k env PYTHONPATH=$R/src $PY eval/pf_swap_eval.py \
            train_dense_stripped.py $DENSE 128 $out.csv $m $k
    done
done
BASE=dense $PY eval/paired.py ${DENSE:0:8} \
    $(for m in 1 2 4; do for k in 64 256 512; do echo pf_m${m}_k$k; done; done) \
    > logs/paired_${DENSE:0:8}_pf.txt 2>&1

# 4 Qwen3-0.6B-Base
for job in "0 0 32" "1 8192 4" "1 64 32" "1 512 32" "4 64 32" "4 512 32"; do
    read -r m k rows <<< "$job"
    out=logs/rule_qwen06b_$( ((m)) && echo pf_m${m}_k$k || echo own)
    [ -e $out.csv ] && continue
    run $(basename $out) bash -c "cd $R && HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
        uv run python experiments/csa2/eval/qwen_swap_eval.py $rows \
        experiments/csa2/$out.csv $m $k"
done
BASE=own $PY eval/paired.py qwen06b pf_m1_k64 pf_m1_k512 pf_m4_k64 pf_m4_k512 \
    > logs/paired_qwen06b_pf.txt 2>&1

# 5 synthetic suite, 6 decode bench
run suite_pf bash -c "cd $R && uv run python -m benchmarks.run \
    src/benchmarks/suites/standard.toml --only latency fidelity \
    --versions deepseek-v4.1/csa2-parameter-free"
run decode_ref_picks bash -c "cd $R && PYTHONPATH=src:experiments/csa2/bench \
    $HOME/repos/modded-nanogpt/.venv/bin/python experiments/csa2/bench/decode_ref_picks.py"

# 7 smoke
run smoke_csa2_pf_smooth env SWEEP_ITERS=30 SWEEP_EXT_ITERS=2 SWEEP_VAL_EVERY=0 \
    SAVE_CHECKPOINT=0 b/run_b.sh tournament train_csa2_pf_smooth.py
echo "$(date +%H:%M) done"
