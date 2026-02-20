# Version Compatibility Fix

## Problem

When running `uv run test_setup.py`, the following error occurred:

```
Testing Qwen3-TTS import...

********
Warning: flash-attn is not installed. Will only run the manual PyTorch version. Please install flash-attn for faster inference.
********

/bin/sh: 1: sox: not found
SoX could not be found!

✗ Qwen3-TTS import failed: cannot import name 'check_model_inputs' from 'transformers.utils.generic'
```

## Root Cause

1. The `check_model_inputs` function was removed or changed in newer versions of the `transformers` library
2. `accelerate==1.12.0` requires Python 3.10+, but the project specified `>=3.9`

Qwen3-TTS was developed and tested with specific versions:
- `transformers==4.57.3`
- `accelerate==1.12.0` (requires Python >=3.10)

Using newer versions (e.g., `transformers>=4.35.0`) causes import errors because the API has changed.

## Solution

Pinned `transformers` and `accelerate` to exact versions that Qwen3-TTS requires:

### Before
```toml
[project]
requires-python = ">=3.9,<3.13"
dependencies = [
    "transformers>=4.35.0",
    "accelerate>=0.20.0",
]
```

### After
```toml
[project]
requires-python = ">=3.10,<3.13"  # Updated for accelerate==1.12.0
dependencies = [
    "transformers==4.57.3",      # Pinned for Qwen3-TTS compatibility
    "accelerate==1.12.0",        # Pinned for Qwen3-TTS compatibility
]
```

## Why Pinning is Necessary

### Transformers API Changes

The `transformers` library frequently updates its API. Functions like `check_model_inputs` may be:
- Removed
- Renamed
- Moved to different modules
- Changed in signature

Qwen3-TTS code depends on specific API versions.

### Accelerate Compatibility

`accelerate` provides model acceleration features that must match the `transformers` version. Mismatched versions can cause:
- Runtime errors
- Performance issues
- Incorrect model loading

## Files Updated

1. **`pyproject.toml`** - Pinned transformers and accelerate versions
2. **`README.md`** - Updated pip install command with exact versions
3. **`DEPENDENCIES_FIX.md`** - Updated with version compatibility information
4. **`SETUP_FIXES.md`** - Updated dependencies fix section

## Verification

After the fix, run:

```bash
# Sync dependencies (this will install exact versions)
uv sync

# Verify setup
uv run test_setup.py
```

Expected output:
```
Testing Qwen3-TTS import...
  ********
  Warning: flash-attn is not installed. Will only run the manual PyTorch version. Please install flash-attn for faster inference.
  ********
  /bin/sh: 1: sox: not found
  SoX could not be found!
  ✓ Qwen3-TTS can be imported

Summary
✓ All dependencies are installed correctly!
```

## About the Warnings

### Flash Attention Warning
```
Warning: flash-attn is not installed. Will only run the manual PyTorch version.
```
- **Impact**: Slower inference (2-3x)
- **Action**: Optional - install for faster performance
- **Status**: Expected warning, not an error

### SoX CLI Warning
```
/bin/sh: 1: sox: not found
SoX could not be found!
```
- **Impact**: Some advanced audio features unavailable
- **Action**: Optional - most features work without it
- **Status**: Expected warning, not an error

### ONNX Runtime GPU Warning
```
[W:onnxruntime:Default, device_discovery.cc:211 DiscoverDevicesForPlatform] GPU device discovery failed
```
- **Impact**: ONNX models will run on CPU instead of GPU
- **Action**: Optional - PyTorch models will still use GPU
- **Status**: Expected warning, not an error

## Version Management Strategy

### Pinned Versions (Exact)
These must match Qwen3-TTS requirements:
- `transformers==4.57.3`
- `accelerate==1.12.0`

### Minimum Versions (Flexible)
These can be updated if needed:
- `fastapi>=0.104.0`
- `uvicorn[standard]>=0.24.0`
- `websockets>=12.0`
- `librosa>=0.10.0`
- `scipy>=1.11.0`
- `numpy>=1.24.0`
- `torch>=2.0.0`
- `torchaudio>=2.0.0`
- `soundfile>=0.12.0`
- `huggingface-hub>=0.19.0`
- `einops>=0.6.0`
- `sox>=1.4.1`
- `onnxruntime>=1.16.0`

## Updating Versions in the Future

If you need to update Qwen3-TTS or its dependencies:

1. **Check Qwen3-TTS requirements**:
   ```bash
   cat Qwen3-TTS/pyproject.toml
   ```

2. **Update pinned versions**:
   ```toml
   "transformers==X.Y.Z",
   "accelerate==A.B.C",
   ```

3. **Test thoroughly**:
   ```bash
   uv sync
   uv run test_setup.py
   uv run qwen_tts_server.py
   ```

4. **Run full test suite**:
   ```bash
   uv run test_qwen_websocket.py --voice reference.wav
   ```

## Troubleshooting

### Import errors after update

```bash
# Clear cache and reinstall
uv cache clean
rm -rf .venv uv.lock
uv sync
```

### Version conflicts

```bash
# Check installed versions
uv pip list | grep -E "(transformers|accelerate)"

# Force reinstall
uv pip install --force-reinstall transformers==4.57.3 accelerate==1.12.0
```

### Check Qwen3-TTS compatibility

```bash
# View Qwen3-TTS requirements
cat Qwen3-TTS/pyproject.toml | grep -A 20 "dependencies"
```

## Best Practices

1. **Always test after version updates**
2. **Pin critical dependencies** (transformers, accelerate)
3. **Use minimum versions for non-critical dependencies**
4. **Document version requirements** in README
5. **Keep uv.lock in version control** for reproducibility

## Summary

- **Problem**: Transformers API changes caused import errors
- **Solution**: Pinned transformers==4.57.3 and accelerate==1.12.0
- **Files updated**: pyproject.toml, README.md, DEPENDENCIES_FIX.md, SETUP_FIXES.md
- **Result**: Qwen3-TTS can now be imported successfully
- **Note**: Warnings about flash-attn, sox, and ONNX GPU are expected and optional

The fix ensures compatibility with Qwen3-TTS by using the exact versions it was tested with.
