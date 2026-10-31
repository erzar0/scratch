#!/bin/sh
# Throwaway Qwen queue, strictly sequential (no process-name waits): retrain
# CSA2-Qwen on the mix, quick books check, then every evaluation.
# Status lines to stdout; each step's log to runs/logs/qwen_<step>.log.
set -e
cd /home/kyre/repos/scratch
run() {
    name=$1
    shift
    echo "$(date +%H:%M) start $name"
    "$@" > "runs/logs/qwen_$name.log" 2>&1
}
REDUCED=niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,niah_multikey_2
REDUCED=$REDUCED,niah_multikey_3,niah_multivalue,niah_multiquery
REDUCED=$REDUCED,icl_trec_coarse,icl_trec_fine,icl_banking77,icl_clinic150
run csa2_transfer env SEQ=4096 BATCH=2 STAGE=transfer STEPS=1500 PEAK_LR=1e-3 \
    uv run python -m qwen.train csa2
run csa2_e2e env SEQ=4096 BATCH=2 STAGE=e2e STEPS=2000 PEAK_LR=3e-4 \
    uv run python -m qwen.train csa2
run quick_csa2 env LIMIT=8 uv run python -m qwen.evaluate lm csa2
run long_qwen uv run python -m qwen.evaluate long qwen
run short_qwen uv run python -m qwen.evaluate short qwen
run lm_csa2 uv run python -m qwen.evaluate lm csa2 csa2:window csa2:random \
    csa2:oracle csa2:mean csa2:bound
run long_csa2 uv run python -m qwen.evaluate long csa2
run short_csa2 uv run python -m qwen.evaluate short csa2
run dense_e2e env SEQ=4096 BATCH=2 STAGE=e2e STEPS=2000 PEAK_LR=3e-5 \
    uv run python -m qwen.train dense
run lm_dense uv run python -m qwen.evaluate lm dense
run check_dense uv run python -m qwen.evaluate check dense
run long_dense uv run python -m qwen.evaluate long dense
run short_dense uv run python -m qwen.evaluate short dense
run route uv run python -m qwen.route
run lm_route uv run python -m qwen.evaluate lm csa2:learned csa2:mixed
run long_reduced env TASKS=$REDUCED LIMIT=10 uv run python -m qwen.evaluate long \
    csa2:window csa2:random csa2:oracle csa2:mean csa2:bound csa2:learned csa2:mixed
run bench_layer uv run python -m qwen.bench layer
run bench_model uv run python -m qwen.bench model
echo "$(date +%H:%M) queue done"
