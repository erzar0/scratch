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
    uv pip install -q -U "huggingface_hub[cli]" hf_transfer gguf torch polars orjson wandb
elif command -v pip &> /dev/null; then
    pip install -q -U "huggingface_hub[cli]" hf_transfer gguf torch polars orjson wandb
elif python3 -m pip --version &> /dev/null; then
    python3 -m pip install -q -U "huggingface_hub[cli]" hf_transfer gguf torch polars orjson wandb
else
    echo "Neither uv nor pip found; installing python3-pip..."
    sudo apt-get update -qq && sudo apt-get install -y -qq python3-pip python3-venv
    python3 -m pip install -q -U "huggingface_hub[cli]" hf_transfer gguf torch polars orjson wandb
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
        if [ -n "$input_key" ]; then
            export WANDB_API_KEY="$input_key"
            echo "W&B API key configured for this session."
        else
            echo "No key entered; proceeding in offline mode."
        fi
    fi
fi

echo "=== Step 3: Fast Parallel Download of IQ1_M GGUF Shards ==="
# hf_transfer maximizes download speeds on Brev 10Gbps+ networking
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$MODEL_DIR"

HF_CLI="huggingface-cli"
if [ -f ".venv/bin/huggingface-cli" ]; then
    HF_CLI=".venv/bin/huggingface-cli"
fi

# Download 5 backbone shards + shared Engram table
"$HF_CLI" download "$MODEL_REPO" \
    --include "*IQ1_M*" "$ENGRAM_FILE" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False

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
PYTHON_BIN="python3"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

# Forward any CLI flags passed to this script (e.g. --wandb, --verbose)
"$PYTHON_BIN" test_iq1m_h200.py \
    --model-dir "$MODEL_DIR" \
    --output-dir "$OUTPUT_DIR" \
    "$@"

echo "=== Step 6: (Optional) Llama.cpp End-to-End Benchmark ==="
if command -v llama-bench &> /dev/null; then
    echo "Running llama-bench with full VRAM offloading (-ngl 99 on H200)..."
    llama-bench -m "$MODEL_DIR/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf" \
        -ngl 99 -p 512,2048 -n 128 -b 1
else
    echo "llama-bench binary not in PATH. To run full end-to-end inference:"
    echo "  git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp"
    echo "  cmake -B build -DGGML_CUDA=ON && cmake --build build --config Release -j"
    echo "  ./build/bin/llama-cli -m $MODEL_DIR/DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ1_M-00001-of-00006.gguf -ngl 99 -p 'Explain quantum computing' -n 128"
fi

echo "=== Benchmark Complete ==="
echo "Logs and exported artifacts are available in: $OUTPUT_DIR"
