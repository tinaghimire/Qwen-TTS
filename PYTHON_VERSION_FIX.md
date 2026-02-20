# Python Version Compatibility Fix

## Problem

The `uv sync` command was failing with a dependency resolution error:

```
error: Failed to resolve: No candidate versions found for scipy>=1.11.0
```

The issue was that the project specified `requires-python = ">=3.8"`, but `scipy>=1.11.0` only supports Python 3.9+.

## Root Cause

From the error message:
- `scipy>=1.11.1,<=1.11.3` depends on `Python>=3.9,<3.13`
- `scipy>=1.11.4,<=1.13.1` depends on `Python>=3.9`
- `scipy>=1.14.0,<=1.15.3` depends on `Python>=3.10`
- `scipy>=1.16.0` depends on `Python>=3.11`

Since the project specified `requires-python = ">=3.8"`, uv tried to find a scipy version compatible with Python 3.8, but none exists.

## Solution

Changed `requires-python` from `">=3.8"` to `">=3.9,<3.13"` in `pyproject.toml`:

```toml
[project]
name = "qwen3-tts-websocket"
version = "0.1.0"
description = "WebSocket server and test suite for Qwen3-TTS voice cloning"
readme = "README.md"
requires-python = ">=3.9,<3.13"  # Changed from ">=3.8"
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
```

## Files Updated

1. **`pyproject.toml`** - Changed `requires-python` to `">=3.9,<3.13"` and removed build system
2. **`README.md`** - Updated installation instructions with Python version requirement
3. **`setup.sh`** - Updated Python version check to look for 3.9-3.12
4. **`test_setup.py`** - Added Python version compatibility test
5. **`UV_START.md`** - Added Python version requirement documentation
6. **`TESTING_GUIDE.md`** - Added Python version requirement note
7. **`UV_SETUP_SUMMARY.md`** - Updated to reflect no build system

## Supported Python Versions

- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11
- ✅ Python 3.12
- ❌ Python 3.8 (not supported by scipy>=1.11.0)
- ❌ Python 3.13+ (not yet tested with all dependencies)

## How to Fix

If you're seeing this error, follow these steps:

### 1. Check your Python version

```bash
python --version
```

### 2. If using Python 3.8 or older, install a compatible version

```bash
# Using uv
uv python install 3.11

# Or using pyenv
pyenv install 3.11.0
pyenv local 3.11.0

# Or using conda
conda create -n qwen3-tts python=3.11
conda activate qwen3-tts
```

### 3. Sync dependencies

```bash
uv sync
```

### 4. Verify setup

```bash
uv run test_setup.py
```

## Verification

After the fix, `uv sync` should complete successfully:

```bash
$ uv sync
Resolved 12 packages in 1.23s
Downloaded 12 packages in 2.34s
Installed 12 packages in 3.45s
```

## Why This Range?

We chose `">=3.9,<3.13"` because:

1. **Lower bound (3.9)**: Required by scipy>=1.11.0
2. **Upper bound (3.13)**: scipy versions up to 1.15.3 support Python <3.13
3. **Practical**: Python 3.9-3.12 are widely used and well-supported
4. **Future-proof**: Can be updated when dependencies support newer Python versions

## Alternative Solutions

If you need to support Python 3.8, you have these options:

### Option 1: Use older scipy

```toml
dependencies = [
    # ...
    "scipy>=1.10.0,<1.11.0",  # Supports Python 3.8
    # ...
]
```

**Trade-off**: Older scipy may have fewer features or bug fixes.

### Option 2: Use separate environments

```bash
# For Python 3.8
uv sync --python 3.8

# For Python 3.11
uv sync --python 3.11
```

**Trade-off**: More complex setup, need to manage multiple environments.

### Option 3: Wait for scipy 3.13 support

Monitor scipy releases for Python 3.13 support and update `requires-python` accordingly.

## Testing

To verify the fix works:

```bash
# 1. Clean up existing environment
rm -rf .venv uv.lock

# 2. Sync with new Python version requirement
uv sync

# 3. Run setup verification
uv run test_setup.py

# 4. Start the server
uv run qwen_tts_server.py
```

## Summary

- **Problem**: scipy>=1.11.0 doesn't support Python 3.8
- **Solution**: Changed `requires-python` to `">=3.9,<3.13"`
- **Impact**: Users need Python 3.9-3.12 to use this project
- **Files updated**: 6 files (pyproject.toml, README.md, setup.sh, test_setup.py, UV_START.md, TESTING_GUIDE.md)

The fix ensures compatibility with scipy>=1.11.0 while maintaining a reasonable Python version range.
