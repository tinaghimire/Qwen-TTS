#!/bin/bash

# ============================================================================
# UV-based Single GPU Training Launch Script for Qwen3-TTS
# ============================================================================
#
# This script launches training on a single GPU using uv run.
#
# Usage:
#   ./uv_train.sh
#
# For custom GPU selection:
#   CUDA_VISIBLE_DEVICES=0 ./uv_train.sh
#
# ============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# GPU Configuration (use specific GPU, e.g., "0" or "2,3" for multiple)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# UV run command
UV_CMD="uv run"

# Mixed precision mode: bf16 (recommended), fp16, no
MIXED_PRECISION=${MIXED_PRECISION:-"bf16"}

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Environment Variables
# -----------------------------------------------------------------------------

# OpenMP settings
export OMP_NUM_THREADS=4

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Check Prerequisites
# -----------------------------------------------------------------------------

echo "=========================================="
echo "UV Single GPU Training Launch Script"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "   Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
    while IFS=, read -r idx name total free; do
        printf "   GPU %s: %s, Total: %s, Free: %s\n" "$idx" "$name" "$total" "$free"
    done
    echo ""
    echo "📊 Using GPU(s): $CUDA_VISIBLE_DEVICES"
else
    echo "⚠ Warning: nvidia-smi not found, cannot check GPU status"
fi

echo ""

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Launch Training
# -----------------------------------------------------------------------------

echo "🚀 Launching training with uv run..."
echo ""
echo "Configuration:"
echo "   Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "   Mixed precision: $MIXED_PRECISION"
echo ""

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠ Warning: .env file not found in current directory"
    echo "   Looking in parent directory..."
    if [ -f "../.env" ]; then
        echo "   Found .env in parent directory"
        cd ..
    else
        echo "❌ Error: .env file not found"
        echo "   Please create a .env file with training configuration"
        exit 1
    fi
fi

# Launch training with uv run
$UV_CMD train.py

# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "✅ Training completed"
echo "=========================================="