#!/usr/bin/env bash
# Turnkey script to test DeepSeek-V4.1-Flash-1017 IQ1_M on a single NVIDIA H200 (Brev environment).
# Supports extensive logging, artifact generation (JSON/CSV), and optional Weights & Biases tracking.
set -euo pipefail

MODEL_REPO="ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF"
MODEL_DIR="./model_iq1m"
ENGRAM_FILE="DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
SHARD6_LINK="DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00006-of-00006.gguf"
OUTPUT_DIR="./runs/benchmarks/iq1m_h200"

echo "=== Step 1: Checking GPU & Hardware Environment ==="
nvidia-smi --query-gpu=name,memory.total,driver_version,utilization.memory --format=csv,noheader || true
df -h .

echo "=== Step 2: Setting up Environment & Installing Dependencies ==="
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if command -v uv &> /dev/null; then
    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment .venv with uv..."
        uv venv .venv
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    echo "Installing dependencies via uv pip..."
    uv pip install -U huggingface_hub hf_transfer gguf torch polars orjson wandb datasets
elif command -v pip &> /dev/null; then
    pip install -U huggingface_hub hf_transfer gguf torch polars orjson wandb datasets
elif python3 -m pip --version &> /dev/null; then
    python3 -m pip install -U huggingface_hub hf_transfer gguf torch polars orjson wandb datasets
else
    echo "Neither uv nor pip found; installing python3-pip..."
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip python3-venv
    python3 -m pip install -U huggingface_hub hf_transfer gguf torch polars orjson wandb datasets
fi

PYTHON_BIN="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

# Prompt for Weights & Biases key if --wandb is passed and not already exported
if [[ "$*" == *"--wandb"* ]] && [ -z "${WANDB_API_KEY:-}" ]; then
    if [ -t 0 ]; then
        echo ""
        echo "=================================================="
        echo "Weights & Biases Authentication"
        echo "Get your API key at: https://wandb.ai/authorize"
        read -r -s -p "Enter W&B API Key (or press Enter to run offline): " input_key
        echo ""
        echo "=================================================="
        clean_key=$(echo "$input_key" | tr -d '[:space:]')
        if [ -n "$clean_key" ]; then
            export WANDB_API_KEY="$clean_key"
            echo "W&B API key configured for this session."
        else
            unset WANDB_API_KEY || true
            echo "No key entered; proceeding in offline mode."
        fi
    fi
fi

echo "=== Step 3: Fast Parallel Download of IQ1_M GGUF Shards ==="
# Enable high performance multi-stream download
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
mkdir -p "$MODEL_DIR"

"$PYTHON_BIN" - <<EOF
from huggingface_hub import snapshot_download

print("Downloading DeepSeek IQ1_M shards via huggingface_hub (hf_transfer enabled)...")
snapshot_download(
    repo_id="$MODEL_REPO",
    allow_patterns=["*IQ1_M*", "$ENGRAM_FILE"],
    local_dir="$MODEL_DIR",
)
print("Shards successfully downloaded to $MODEL_DIR")
EOF

echo "=== Step 4: Linking Shared Engram Table as Shard 6 ==="
cd "$MODEL_DIR"
if [ ! -e "$SHARD6_LINK" ]; then
    ln -s "$ENGRAM_FILE" "$SHARD6_LINK"
    echo "Symlinked $ENGRAM_FILE -> $SHARD6_LINK"
else
    echo "Shard 6 link already present."
fi
cd - > /dev/null

echo "=== Step 5: Running Attention Comparison with Extensive Logging & Artifacts ==="
# Forward any CLI flags passed to this script (e.g. --wandb, --verbose)
"$PYTHON_BIN" test_iq1m_h200.py \
    --model-dir "$MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo "=== Step 6: SOTA Downstream Benchmarks & llama.cpp Reasoning ==="
RUN_DOWNSTREAM=false
if [[ "$*" == *"--llamacpp"* ]] || [[ "$*" == *"--reasoning"* ]] || [[ "$*" == *"--downstream"* ]]; then
    RUN_DOWNSTREAM=true
fi

if [ "$RUN_DOWNSTREAM" = true ]; then
    LLAMA_DIR="./llama.cpp"
    if ! command -v llama-bench &> /dev/null && [ ! -f "$LLAMA_DIR/build/bin/llama-bench" ]; then
        echo "Building llama.cpp with CUDA support for H200..."
        if [ ! -d "$LLAMA_DIR" ]; then
            git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
        fi
        cmake -B "$LLAMA_DIR/build" -S "$LLAMA_DIR" -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
        cmake --build "$LLAMA_DIR/build" --config Release -j"$(nproc)"
    fi

    BENCH_BIN="llama-bench"
    CLI_BIN="llama-cli"
    if [ -f "$LLAMA_DIR/build/bin/llama-bench" ]; then
        BENCH_BIN="$LLAMA_DIR/build/bin/llama-bench"
        CLI_BIN="$LLAMA_DIR/build/bin/llama-cli"
    fi

    echo "Running llama-bench on NVIDIA H200 (100% offload, -ngl 99)..."
    "$BENCH_BIN" -m "$MODEL_DIR/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf" \
        -ngl 99 -p 512,2048 -n 128 -b 1

    echo "Running SOTA downstream reasoning evaluations (AIME25, GPQA-D, MATH-500, Passkey, Perplexity)..."
    "$PYTHON_BIN" eval_downstream_h200.py \
        --model-dir "$MODEL_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --llama-dir "$LLAMA_DIR" \
        "$@"
else
    echo "Pass --reasoning or --llamacpp to run SOTA downstream tasks (AIME25, GPQA-D, MATH-500, Perplexity, Passkey)."
fi

echo "=== Benchmark Complete ==="
echo "Logs and exported artifacts are available in: $OUTPUT_DIR"
