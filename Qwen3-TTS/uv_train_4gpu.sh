#!/bin/bash

# ============================================================================
# UV-based Multi-GPU Training Launch Script for Qwen3-TTS
# ============================================================================
#
# This script launches distributed training across 4 GPUs using uv run with accelerate.
#
# Usage:
#   ./uv_train_4gpu.sh
#
# For custom GPU selection:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./uv_train_4gpu.sh
#
# ============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Number of GPUs to use
NUM_GPUS=4

# Number of machines (for multi-node training)
NUM_MACHINES=1

# Mixed precision mode: bf16 (recommended), fp16, no
MIXED_PRECISION="bf16"

# UV run command
UV_CMD="uv run"

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Environment Variables for Distributed Training
# -----------------------------------------------------------------------------

# NCCL (NVIDIA Collective Communications Library) settings
export NCCL_DEBUG=INFO              # Enable NCCL debugging output
# export NCCL_IB_DISABLE=1          # Disable InfiniBand if not available
# export NCCL_SOCKET_IFNAME=eth0    # Set network interface for NCCL

# OpenMP settings
export OMP_NUM_THREADS=4            # Number of OpenMP threads per process

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Check Prerequisites
# -----------------------------------------------------------------------------

echo "=========================================="
echo "UV Multi-GPU Training Launch Script"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "   Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if accelerate is available with uv
if ! $UV_CMD -c "import accelerate" 2>/dev/null; then
    echo "❌ Error: accelerate is not installed"
    echo "   Install it with: uv pip install accelerate"
    exit 1
fi

# Check number of visible GPUs
VISIBLE_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "📊 GPU Configuration:"
echo "   Visible GPUs: $VISIBLE_GPUS"
echo "   CUDA Devices: $CUDA_VISIBLE_DEVICES"

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🔍 GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | \
    while IFS=, read -r idx name total free; do
        printf "   GPU %s: %s, Total: %s, Free: %s\n" "$idx" "$name" "$total" "$free"
    done
else
    echo "⚠ Warning: nvidia-smi not found, cannot check GPU status"
fi

echo ""

# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Launch Training
# -----------------------------------------------------------------------------

echo "🚀 Launching distributed training with uv run..."
echo ""
echo "Configuration:"
echo "   Number of processes: $NUM_GPUS"
echo "   Number of machines: $NUM_MACHINES"
echo "   Mixed precision: $MIXED_PRECISION"
echo "   Data directory: $(pwd)/data"
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

# Launch training with uv run and accelerate
$UV_CMD accelerate launch \
    --num_processes=$NUM_GPUS \
    --num_machines=$NUM_MACHINES \
    --mixed_precision=$MIXED_PRECISION \
    --dynamo_backend="no" \
    train.py

# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "✅ Training completed"
echo "=========================================="