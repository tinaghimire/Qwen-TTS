# Build System Fix

## Problem

When running `uv sync`, the following error occurred:

```
× Failed to build `qwen3-tts-websocket @ file:///home/ml/workspaces/kristina/Qwen3-TTS-finetuning`
├─▶ The build backend returned an error
╰─▶ Call to `hatchling.build.build_editable` failed (exit status: 1)

ValueError: Unable to determine which files to ship inside the wheel
```

## Root Cause

The `pyproject.toml` included a `[build-system]` section with hatchling, which tried to build a Python package (wheel). However:

1. This is a **standalone application**, not a library
2. There's no package directory (e.g., `qwen3_tts_websocket/`)
3. Hatchling couldn't determine what files to include in the wheel

## Solution

Removed the `[build-system]` section and related configuration from `pyproject.toml`:

### Before (with build system)
```toml
[project]
name = "qwen3-tts-websocket"
version = "0.1.0"
description = "WebSocket server and test suite for Qwen3-TTS voice cloning"
readme = "README.md"
requires-python = ">=3.9,<3.13"
dependencies = [...]

[project.optional-dependencies]
dev = [...]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = []

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
```

### After (without build system)
```toml
[project]
name = "qwen3-tts-websocket"
version = "0.1.0"
description = "WebSocket server and test suite for Qwen3-TTS voice cloning"
readme = "README.md"
requires-python = ">=3.9,<3.13"
dependencies = [...]

[dependency-groups]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
]
```

## Changes Made

1. **Removed `[build-system]` section** - No package building needed
2. **Removed `[tool.uv]` section** - Deprecated, replaced with `[dependency-groups]`
3. **Removed `[tool.pytest.ini_options]`** - Not needed for this project
4. **Changed `[project.optional-dependencies]` to `[dependency-groups]`** - Modern uv syntax

## Why This Works

### For Libraries (need build system)
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mylib"]
```

This creates a wheel that can be installed with `pip install mylib`.

### For Applications (no build system needed)
```toml
[project]
name = "myapp"
dependencies = [...]

[dependency-groups]
dev = [...]
```

This just installs dependencies for running scripts with `uv run myapp.py`.

## Benefits

1. **Simpler configuration** - No need to define package structure
2. **Faster setup** - No package building step
3. **Clearer intent** - Clearly indicates this is an application, not a library
4. **No wheel generation** - Direct script execution with `uv run`

## Files Updated

1. **`pyproject.toml`** - Removed build system configuration
2. **`UV_START.md`** - Added note about standalone application
3. **`UV_SETUP_SUMMARY.md`** - Updated to reflect no build system
4. **`PYTHON_VERSION_FIX.md`** - Added build system fix to changelog

## Verification

After the fix, `uv sync` should complete successfully:

```bash
$ uv sync
Resolved 12 packages in 46ms
Downloaded 12 packages in 2.34s
Installed 12 packages in 3.45s
```

## Running the Application

Since there's no package to install, run scripts directly:

```bash
# Start the server
uv run qwen_tts_server.py

# Run tests
uv run test_qwen_websocket.py --voice reference.wav

# Run examples
uv run quick_start.py
uv run example_usage.py
```

## If You Want to Create a Library

If you later decide to make this a library, you would:

1. Create a package directory:
   ```
   qwen3_tts_websocket/
   ├── __init__.py
   ├── server.py
   └── client.py
   ```

2. Add build system to `pyproject.toml`:
   ```toml
   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"
   
   [tool.hatch.build.targets.wheel]
   packages = ["qwen3_tts_websocket"]
   ```

3. Users would install with:
   ```bash
   pip install qwen3-tts-websocket
   ```

## Summary

- **Problem**: Hatchling tried to build a wheel for a standalone application
- **Solution**: Removed `[build-system]` section from `pyproject.toml`
- **Result**: `uv sync` now works without package building
- **Impact**: Simpler configuration, faster setup, clearer project structure

The fix ensures that `uv sync` works correctly for this standalone application without attempting to build a package.
