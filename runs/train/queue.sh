#!/usr/bin/env bash
# CSA2 language models, one GPU job at a time.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONUNBUFFERED=1
SEQ=2048 BATCH=2 STEPS=24000 uv run python -m model.train csa2 > runs/train/csa2_seq2048.log 2>&1
SEQ=2048 BATCH=2 STEPS=24000 SEED=1 uv run python -m model.train csa2 > runs/train/csa2_seq2048_s1.log 2>&1
SEQ=8192 BATCH=1 STEPS=12000 uv run python -m model.train csa2 > runs/train/csa2_seq8192.log 2>&1
