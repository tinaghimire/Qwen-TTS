# CUDA and Flash-Attention Setup Guide

This guide explains how the project handles CUDA and flash-attn dependencies using `uv sync`.

## Overview

The project is configured to automatically handle:
- **PyTorch with CUDA 13.0** - For GPU acceleration
- **flash-attn** - For optimized attention computation (2-4x faster)
- **Build dependencies** - Required for compiling flash-attn from source

## Configuration

### pyproject.toml

The `pyproject.toml` file contains all necessary configurations:

```toml
[project]
dependencies = [
    # ... other dependencies ...
    "torch>=2.5.0",
    "torchaudio>=2.5.0",
    "flash-attn>=2.6.0",
]

[build-system]
requires = ["setuptools>=70.0.0", "wheel", "ninja"]
build-backend = "setuptools.build_meta"

[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv]
extra-build-dependencies = [
    "setuptools>=70.0.0",
    "wheel",
    "ninja",
]
```

### Key Components

1. **PyTorch Index**: The `[[tool.uv.index]]` section tells `uv` to use PyTorch's CUDA 13.0 wheel repository
2. **flash-attn**: Added to dependencies for automatic installation
3. **Build Dependencies**: Required for compiling flash-attn from source
4. **Build System**: Configured to use setuptools with ninja for faster builds

## Installation

### Quick Start

```bash
# Simply run uv sync - it handles everything!
uv sync
```

This command will:
1. Create a virtual environment (if needed)
2. Install PyTorch with CUDA 13.0 support
3. Install flash-attn (compiles from source, takes 10-30 minutes)
4. Install all other dependencies

### What Happens During Installation

#### Phase 1: PyTorch Installation
- Downloads PyTorch wheels from PyTorch's CUDA 13.0 repository
- Installs CUDA runtime libraries (cublas, cudnn, etc.)
- Sets up CUDA bindings

#### Phase 2: flash-attn Compilation
- Downloads flash-attn source code
- Compiles CUDA kernels (this takes time!)
- Installs optimized attention operators

**Expected time**: 10-30 minutes depending on your CPU

## Verification

After installation, verify everything is working:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Check flash-attn
python -c "import flash_attn; print('flash-attn installed successfully!')"
```

Expected output:
```
PyTorch: 2.10.0+cu130
CUDA available: True
CUDA version: 13.0
flash-attn installed successfully!
```

## Troubleshooting

### Issue: CUDA Version Mismatch

**Symptom**: Error about CUDA version mismatch during flash-attn compilation

**Solution**: Ensure your system CUDA matches PyTorch's CUDA version
```bash
# Check system CUDA
nvidia-smi

# Should show CUDA Version: 13.0
```

### Issue: flash-attn Build Fails

**Symptom**: Compilation errors during flash-attn installation

**Solutions**:

1. **Install build dependencies manually**:
```bash
uv pip install setuptools wheel ninja
uv pip install flash-attn --no-build-isolation
```

2. **Skip flash-attn** (not recommended, but works):
```bash
# Remove flash-attn from pyproject.toml dependencies
# Then run:
uv sync
```

3. **Use pre-built wheel** (if available):
```bash
uv pip install flash-attn --find-links https://github.com/Dao-AILab/flash-attention/releases
```

### Issue: Out of Memory During Build

**Symptom**: Build process runs out of memory

**Solution**: Reduce parallel compilation jobs
```bash
export MAX_JOBS=2
uv sync
```

## Performance Benefits

### With flash-attn
- **2-4x faster** attention computation
- **Lower memory usage** - enables longer sequences
- **Better GPU utilization** - especially important for RTX 5070 Ti

### Without flash-attn
- Slower inference (but still functional)
- Higher memory usage
- May limit sequence length

## GPU Requirements

### Minimum
- NVIDIA GPU with compute capability 7.0+
- 8GB VRAM
- CUDA 13.0

### Recommended
- NVIDIA RTX 4090 or RTX 5070 Ti
- 16GB+ VRAM
- CUDA 13.0

## System Requirements

### Linux
- Ubuntu 20.04+ or similar
- CUDA 13.0 toolkit
- GCC 9+ for compilation

### Build Tools
- ninja (for faster builds)
- setuptools >= 70.0.0
- wheel

## Clean Installation

If you need to start fresh:

```bash
# Remove virtual environment
rm -rf .venv

# Remove uv lock file
rm uv.lock

# Reinstall everything
uv sync
```

## Advanced Configuration

### Using Different CUDA Version

To use a different CUDA version, modify `pyproject.toml`:

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu121"  # Change to cu121 for CUDA 12.1
explicit = true
```

Then update torch dependencies:
```toml
dependencies = [
    "torch>=2.5.0",
    "torchaudio>=2.5.0",
]
```

And reinstall:
```bash
rm -rf .venv uv.lock
uv sync
```

### Disabling flash-attn

If you want to skip flash-attn installation:

1. Remove from `pyproject.toml`:
```toml
dependencies = [
    # ... other dependencies ...
    # "flash-attn>=2.6.0",  # Comment out or remove
]
```

2. Reinstall:
```bash
uv sync
```

## Summary

The project is configured to handle all CUDA and flash-attn dependencies automatically through `uv sync`. Simply run:

```bash
uv sync
```

And everything will be installed correctly with CUDA 13.0 support and optimized flash-attn for faster inference.
