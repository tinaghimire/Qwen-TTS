# uv Setup Summary

This document summarizes the uv package management setup for Qwen3-TTS WebSocket testing.

## What Changed

### 1. Created `pyproject.toml`
Replaced `requirements.txt` with a modern `pyproject.toml` file that defines:
- Project metadata (name, version, description)
- Python version requirement (>=3.9,<3.13)
- All dependencies
- Dev dependencies (using dependency-groups)
- **No build system** (standalone application, not a library)

### 2. Updated `setup.sh`
Modified to use `uv sync` instead of `pip install -r requirements.txt`:
- Checks for uv installation
- Creates virtual environment automatically
- Syncs dependencies from `pyproject.toml`
- Runs setup verification

### 3. Updated Documentation
All documentation now uses `uv run` prefix:
- `README.md`
- `TESTING_GUIDE.md`
- `quick_start.py`
- `example_usage.py`

### 4. Created Additional Files
- **`.gitignore`**: Ignores `.venv/`, `uv.lock`, and other generated files
- **`UV_START.md`**: Detailed uv quick start guide
- **`test_setup.py`**: Setup verification script

## How to Use

### Initial Setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Navigate to project
cd /home/ml/workspaces/kristina/Qwen3-TTS-finetuning

# 3. Sync dependencies
uv sync

# 4. Verify setup
uv run test_setup.py
```

### Running the Server

```bash
# Start server
uv run qwen_tts_server.py

# With options
uv run qwen_tts_server.py --port 8001 --model-size 0.6B
```

### Running Tests

```bash
# All tests
uv run test_qwen_websocket.py --voice reference.wav

# Specific test
uv run test_qwen_websocket.py --voice reference.wav --test basic
```

### Running Examples

```bash
# Quick start
uv run quick_start.py

# All examples
uv run example_usage.py
```

## File Structure

```
Qwen3-TTS-finetuning/
├── pyproject.toml          # Project configuration and dependencies
├── uv.lock                 # Locked dependencies (auto-generated)
├── .venv/                  # Virtual environment (auto-generated)
├── .gitignore              # Git ignore rules
├── setup.sh                # Setup script (uses uv)
├── test_setup.py           # Setup verification
├── qwen_tts_server.py      # WebSocket server
├── test_qwen_websocket.py  # Test suite
├── quick_start.py          # Quick start example
├── example_usage.py        # Usage examples
├── README.md               # Main documentation
├── UV_START.md             # uv quick start guide
├── TESTING_GUIDE.md        # Testing guide
├── COMPARISON.md           # Comparison with dia2
└── voices/                 # Reference audio files
    └── reference.wav
```

## Dependencies

All dependencies are now managed in `pyproject.toml`:

```toml
[project]
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

## Common uv Commands

```bash
# Sync dependencies (install/update)
uv sync

# Add a new package
uv add package-name

# Add dev dependency
uv add --dev pytest

# Remove a package
uv remove package-name

# Run a script
uv run script.py

# Run Python REPL
uv run python

# List installed packages
uv pip list

# Update all packages
uv sync --upgrade

# Clear cache
uv cache clean
```

## Benefits of Using uv

1. **Speed**: 10-100x faster than pip
2. **Reliability**: Better dependency resolution
3. **Simplicity**: Single command for setup (`uv sync`)
4. **Compatibility**: Drop-in replacement for pip
5. **Modern**: Written in Rust, actively maintained
6. **Locking**: `uv.lock` ensures reproducible builds

## Migration from pip

### Before (pip)
```bash
pip install -r requirements.txt
python qwen_tts_server.py
python test_qwen_websocket.py --voice reference.wav
```

### After (uv)
```bash
uv sync
uv run qwen_tts_server.py
uv run test_qwen_websocket.py --voice reference.wav
```

## Troubleshooting

### uv command not found
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify installation
uv --version
```

### Virtual environment issues
```bash
# Remove and recreate
rm -rf .venv
uv sync
```

### Dependency conflicts
```bash
# Force resolution
uv sync --resolution=highest

# Or clear cache
uv cache clean
```

### Setup verification
```bash
# Run verification script
uv run test_setup.py
```

## Comparison: requirements.txt vs pyproject.toml

### requirements.txt (Old)
```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
websockets>=12.0
...
```

**Issues:**
- No project metadata
- No version locking
- No dev dependencies
- Manual management

### pyproject.toml (New)
```toml
[project]
name = "qwen3-tts-websocket"
version = "0.1.0"
description = "WebSocket server for Qwen3-TTS"
requires-python = ">=3.8"
dependencies = [
    "fastapi>=0.104.0",
    ...
]

[project.optional-dependencies]
dev = ["pytest>=7.4.0"]
```

**Benefits:**
- Complete project metadata
- Version locking via `uv.lock`
- Separate dev dependencies
- Standard Python packaging format

## Next Steps

1. **Run setup**:
   ```bash
   ./setup.sh
   # or
   uv sync
   ```

2. **Verify setup**:
   ```bash
   uv run test_setup.py
   ```

3. **Start server**:
   ```bash
   uv run qwen_tts_server.py
   ```

4. **Run tests**:
   ```bash
   uv run test_qwen_websocket.py --voice reference.wav
   ```

5. **Read documentation**:
   - [UV_START.md](UV_START.md) - Detailed uv guide
   - [README.md](README.md) - Main documentation
   - [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing instructions

## Support

- **uv issues**: https://github.com/astral-sh/uv/issues
- **Qwen3-TTS issues**: Check troubleshooting in README.md
- **Setup issues**: Run `uv run test_setup.py` for diagnostics
