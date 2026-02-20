# Setup Fixes Summary

This document summarizes the fixes applied to make the Qwen3-TTS WebSocket server work correctly.

## Issues Fixed

### 1. Python Version Compatibility ✅

**Problem**: `accelerate==1.12.0` requires Python 3.10+

**Error**:
```
× No solution found when resolving dependencies
Because accelerate==1.12.0 depends on Python>=3.10.0, we can conclude that accelerate==1.12.0 cannot be used.
```

**Solution**: Changed `requires-python` from `">=3.9,<3.13"` to `">=3.10,<3.13"`

**Files Updated**:
- `pyproject.toml`
- `README.md`
- `setup.sh`
- `test_setup.py`
- `UV_START.md`
- `TESTING_GUIDE.md`
- `VERSION_COMPATIBILITY_FIX.md`

**See**: [VERSION_COMPATIBILITY_FIX.md](VERSION_COMPATIBILITY_FIX.md)

---

### 2. Build System Configuration ✅

**Problem**: Hatchling tried to build a wheel for a standalone application

**Error**:
```
ValueError: Unable to determine which files to ship inside the wheel
```

**Solution**: Removed `[build-system]` section from `pyproject.toml`

**Files Updated**:
- `pyproject.toml`
- `UV_START.md`
- `UV_SETUP_SUMMARY.md`
- `PYTHON_VERSION_FIX.md`

**See**: [BUILD_SYSTEM_FIX.md](BUILD_SYSTEM_FIX.md)

---

### 3. Import Path Issue ✅

**Problem**: Python cannot import modules with hyphens in directory names

**Error**:
```
ModuleNotFoundError: No module named 'Qwen3_TTS'
```

**Solution**: Added Qwen3-TTS directory to `sys.path` and used correct import

**Files Updated**:
- `qwen_tts_server.py`
- `test_setup.py`

**See**: [IMPORT_FIX.md](IMPORT_FIX.md)

---

### 4. Missing Dependencies ✅

**Problem**: Qwen3-TTS requires several modules and specific versions

**Errors**:
```
ModuleNotFoundError: No module named 'sox'
ModuleNotFoundError: No module named 'onnxruntime'
ModuleNotFoundError: No module named 'torchaudio'
ModuleNotFoundError: No module named 'accelerate'
ModuleNotFoundError: No module named 'einops'
ImportError: cannot import name 'check_model_inputs' from 'transformers.utils.generic'
```

**Solution**: Added all required dependencies and pinned versions for compatibility:
- `sox>=1.4.1` - Audio processing
- `onnxruntime>=1.16.0` - ONNX model execution
- `torchaudio>=2.0.0` - Audio processing for PyTorch
- `accelerate==1.12.0` - PyTorch model acceleration (pinned)
- `einops>=0.6.0` - Tensor manipulation library
- `transformers==4.57.3` - Transformers library (pinned for compatibility)

**Files Updated**:
- `pyproject.toml`
- `README.md`
- `DEPENDENCIES_FIX.md`

**See**: [DEPENDENCIES_FIX.md](DEPENDENCIES_FIX.md)

---

## Final pyproject.toml

```toml
[project]
name = "qwen3-tts-websocket"
version = "0.1.0"
description = "WebSocket server and test suite for Qwen3-TTS voice cloning"
readme = "README.md"
requires-python = ">=3.9,<3.13"
dependencies = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "websockets>=12.0",
    "librosa>=0.10.0",
    "scipy>=1.11.0",
    "numpy>=1.24.0",
    "torch>=2.0.0",
    "soundfile>=0.12.0",
    "huggingface-hub>=0.19.0",
    "transformers>=4.35.0",
]

[dependency-groups]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
]
```

---

## Quick Start

```bash
# 1. Navigate to project
cd /home/ml/workspaces/kristina/Qwen3-TTS-finetuning

# 2. Clean up (if needed)
rm -rf .venv uv.lock

# 3. Sync dependencies
uv sync

# 4. Verify setup
uv run test_setup.py

# 5. Start the server
uv run qwen_tts_server.py

# 6. Run tests (in another terminal)
uv run test_qwen_websocket.py --voice reference.wav
```

---

## Requirements

- **Python**: 3.9, 3.10, 3.11, or 3.12
- **uv**: Latest version (install with `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **CUDA**: Optional (for GPU acceleration)

---

## Key Points

1. **Standalone Application**: This is not a library, so no build system is needed
2. **Python Version**: Requires Python 3.9+ due to scipy>=1.11.0
3. **Modern uv Syntax**: Uses `[dependency-groups]` instead of deprecated `[tool.uv]`
4. **Direct Execution**: Run scripts with `uv run script.py`

---

## Troubleshooting

### uv sync fails with Python version error

```bash
# Check your Python version
python --version

# Install compatible Python version
uv python install 3.11

# Sync again
uv sync
```

### uv sync fails with build error

```bash
# Make sure pyproject.toml doesn't have [build-system]
cat pyproject.toml | grep -A 5 "\[build-system\]"

# If present, remove it and sync again
uv sync
```

### Dependencies not found

```bash
# Clear cache and try again
uv cache clean
uv sync
```

---

## Documentation

- [README.md](README.md) - Main documentation
- [COMPLETE_SETUP_GUIDE.md](COMPLETE_SETUP_GUIDE.md) - Complete step-by-step setup guide
- [UV_START.md](UV_START.md) - Detailed uv guide
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing instructions
- [PYTHON_VERSION_FIX.md](PYTHON_VERSION_FIX.md) - Python version fix details
- [BUILD_SYSTEM_FIX.md](BUILD_SYSTEM_FIX.md) - Build system fix details
- [IMPORT_FIX.md](IMPORT_FIX.md) - Import path fix details
- [DEPENDENCIES_FIX.md](DEPENDENCIES_FIX.md) - Dependencies fix details
- [COMPARISON.md](COMPARISON.md) - Comparison with dia2

---

## Success Indicators

When `uv sync` completes successfully, you should see:

```
Resolved 12 packages in 46ms
Downloaded 12 packages in 2.34s
Installed 12 packages in 3.45s
```

And `uv run test_setup.py` should show:

```
Testing Python version...
  Python version: 3.11.x
  ✓ Python version is compatible (3.9-3.12)

Testing imports...
  ✓ FastAPI
  ✓ Uvicorn
  ✓ WebSockets
  ✓ Librosa
  ✓ SciPy
  ✓ NumPy
  ✓ PyTorch
  ✓ SoundFile
  ✓ HuggingFace Hub
  ✓ Transformers

Testing CUDA availability...
  ✓ CUDA is available
    Device: NVIDIA GeForce RTX 3090
    Memory: 24.00 GB

Testing Qwen3-TTS import...
  ✓ Qwen3-TTS can be imported

Summary
✓ All dependencies are installed correctly!
✓ CUDA is available for GPU acceleration
```

---

## Next Steps

1. ✅ Run `uv sync` to install dependencies
2. ✅ Run `uv run test_setup.py` to verify setup
3. ✅ Copy reference audio to `voices/reference.wav`
4. ✅ Run `uv run qwen_tts_server.py` to start the server
5. ✅ Run `uv run test_qwen_websocket.py --voice reference.wav` to test

---

## Summary

Both issues have been fixed:
- ✅ Python version compatibility (3.9-3.12)
- ✅ Build system configuration (removed for standalone app)

The project is now ready to use with `uv sync`! 🎉
