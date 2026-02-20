# Quick Start with uv

This guide shows how to set up and run the Qwen3-TTS WebSocket server using `uv`.

## What is uv?

`uv` is a fast Python package installer and resolver, written in Rust. It's a drop-in replacement for `pip` and `pip-tools` that's much faster.

## Installation

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or visit: https://github.com/astral-sh/uv

### Check Python Version

This project requires **Python 3.10, 3.11, or 3.12** (accelerate==1.12.0 requires Python >=3.10).

```bash
# Check your Python version
python --version

# If needed, install a compatible Python version
uv python install 3.11
```

## Setup

### 1. Clone or navigate to the project

```bash
cd /home/ml/workspaces/kristina/Qwen3-TTS-finetuning
```

### 2. Sync dependencies

```bash
uv sync
```

This will:
- Create a virtual environment in `.venv/`
- Install all dependencies from `pyproject.toml`
- Lock dependencies in `uv.lock`
- Use Python 3.9-3.12 (automatically selected)

> **Note**: This is a standalone application, not a library, so no package building is required.

### 3. Prepare reference audio

```bash
mkdir -p voices
cp /path/to/your/reference.wav voices/reference.wav
```

## Running the Server

### Start the server

```bash
uv run qwen_tts_server.py
```

### With custom options

```bash
uv run qwen_tts_server.py --host 0.0.0.0 --port 8001 --model-size 0.6B
```

## Running Tests

### Run all tests

```bash
uv run test_qwen_websocket.py --voice reference.wav
```

### Run specific tests

```bash
# Basic test
uv run test_qwen_websocket.py --voice reference.wav --test basic

# Streaming test
uv run test_qwen_websocket.py --voice reference.wav --test streaming

# Similarity test
uv run test_qwen_websocket.py --voice reference.wav --test similarity

# ElevenLabs-compatible test
uv run test_qwen_websocket.py --voice reference.wav --test elevenlabs
```

## Running Examples

### Quick start example

```bash
uv run quick_start.py
```

### All examples

```bash
uv run example_usage.py
```

## Using the Virtual Environment Directly

If you want to activate the virtual environment:

```bash
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

Then run commands without `uv run`:

```bash
python qwen_tts_server.py
python test_qwen_websocket.py --voice reference.wav
```

## Adding Dependencies

### Add a new dependency

```bash
uv add package-name
```

Example:
```bash
uv add requests
```

### Add a dev dependency

```bash
uv add --dev pytest
```

### Remove a dependency

```bash
uv remove package-name
```

## Common uv Commands

```bash
# Sync dependencies (install/update)
uv sync

# Add a package
uv add package-name

# Remove a package
uv remove package-name

# List installed packages
uv pip list

# Update all packages
uv sync --upgrade

# Run a script
uv run script.py

# Run Python REPL
uv run python

# Run with specific Python version
uv run --python 3.10 script.py
```

## Troubleshooting

### uv command not found

Make sure you've installed uv and added it to your PATH:

```bash
# Check if uv is installed
uv --version

# If not, install it
curl -LsSf https://astral.sh/uv/install.sh | sh
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

## Comparison: uv vs pip

| Command | pip | uv |
|---------|-----|-----|
| Install dependencies | `pip install -r requirements.txt` | `uv sync` |
| Add package | `pip install package` | `uv add package` |
| Remove package | `pip uninstall package` | `uv remove package` |
| Run script | `python script.py` | `uv run script.py` |
| Speed | Slow | Fast (10-100x) |

## Benefits of Using uv

1. **Speed**: 10-100x faster than pip
2. **Reliability**: Better dependency resolution
3. **Simplicity**: Single command for setup (`uv sync`)
4. **Compatibility**: Drop-in replacement for pip
5. **Modern**: Written in Rust, actively maintained

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [TESTING_GUIDE.md](TESTING_GUIDE.md) for testing instructions
3. See [COMPARISON.md](COMPARISON.md) for comparison with dia2

## Support

For uv issues: https://github.com/astral-sh/uv/issues
For Qwen3-TTS issues: Check the troubleshooting section in README.md
