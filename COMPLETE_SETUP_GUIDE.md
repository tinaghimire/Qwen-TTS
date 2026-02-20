# Complete Setup Guide for Qwen3-TTS WebSocket Server

This guide provides step-by-step instructions to set up and run the Qwen3-TTS WebSocket server.

## Prerequisites

- **Python**: 3.9, 3.10, 3.11, or 3.12
- **uv**: Latest version (for package management)
- **CUDA**: Optional (for GPU acceleration)
- **SoX CLI**: Optional (for advanced audio processing)

## Installation

### Step 1: Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Navigate to Project

```bash
cd /home/ml/workspaces/kristina/Qwen3-TTS-finetuning
```

### Step 3: Sync Dependencies

```bash
uv sync
```

This will:
- Create a virtual environment in `.venv/`
- Install all required dependencies
- Lock dependencies in `uv.lock`

### Step 4: Verify Setup

```bash
uv run test_setup.py
```

Expected output:
```
============================================================
Qwen3-TTS Setup Verification
============================================================

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
  ✓ Sox
  ✓ ONNX Runtime

Testing CUDA availability...
  ✓ CUDA is available
    Device: NVIDIA GeForce RTX 3090
    Memory: 24.00 GB

Testing Qwen3-TTS import...
  ********
  Warning: flash-attn is not installed. Will only run the manual PyTorch version. Please install flash-attn for faster inference.
  ********
  /bin/sh: 1: sox: not found
  SoX could not be found!
  ✓ Qwen3-TTS can be imported

============================================================
Summary
============================================================
✓ All dependencies are installed correctly!
✓ CUDA is available for GPU acceleration
```

## Understanding the Warnings

### Flash Attention Warning
```
Warning: flash-attn is not installed. Will only run the manual PyTorch version.
```
- **Impact**: Slower inference (2-3x)
- **Action**: Optional - install for faster performance
- **How to install**: `pip install flash-attn --no-build-isolation`

### SoX CLI Warning
```
/bin/sh: 1: sox: not found
SoX could not be found!
```
- **Impact**: Some advanced audio features unavailable
- **Action**: Optional - most features work without it
- **How to install**: `sudo apt-get install sox` (Ubuntu/Debian)

## Running the Server

### Start the Server

```bash
uv run qwen_tts_server.py
```

### With Custom Options

```bash
uv run qwen_tts_server.py --host 0.0.0.0 --port 8001 --model-size 0.6B
```

### Expected Output

```
Loading Qwen3-TTS Base model...
Starting Qwen3-TTS server on http://0.0.0.0:8000
Model size: 1.7B
Voices directory: /home/ml/workspaces/kristina/Qwen3-TTS-finetuning/voices

WebSocket endpoints:
  - ws://0.0.0.0:8000/ws/voice-clone/{voice_id}
  - ws://0.0.0.0:8000/v1/text-to-speech/{voice_id}/stream-input

HTTP endpoint:
  - POST http://0.0.0.0:8000/voice-clone/{voice_id}
```

## Preparing Reference Audio

### Create Voices Directory

```bash
mkdir -p voices
```

### Add Reference Audio

```bash
cp /path/to/your/reference.wav voices/reference.wav
```

**Requirements**:
- Format: WAV
- Channels: Mono
- Duration: 5-10 seconds recommended
- Quality: Clear, minimal background noise

## Running Tests

### Run All Tests

```bash
uv run test_qwen_websocket.py --voice reference.wav
```

### Run Specific Tests

```bash
# Basic connection test
uv run test_qwen_websocket.py --voice reference.wav --test basic

# Streaming test
uv run test_qwen_websocket.py --voice reference.wav --test streaming

# Audio similarity test
uv run test_qwen_websocket.py --voice reference.wav --test similarity

# ElevenLabs-compatible test
uv run test_qwen_websocket.py --voice reference.wav --test elevenlabs
```

## Quick Start Example

```bash
# Run the quick start script
uv run quick_start.py
```

## All Examples

```bash
# Run all usage examples
uv run example_usage.py
```

## Troubleshooting

### uv sync fails

```bash
# Clear cache and try again
uv cache clean
uv sync
```

### Python version error

```bash
# Check Python version
python --version

# Install compatible Python
uv python install 3.11
```

### CUDA not available

```bash
# Check CUDA
nvidia-smi

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Import errors

```bash
# Verify all dependencies
uv run test_setup.py

# Re-sync dependencies
uv sync
```

### Port already in use

```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use a different port
uv run qwen_tts_server.py --port 8001
```

## Optional: Install Flash Attention

For faster inference (2-3x speedup):

```bash
# Install flash-attn
pip install flash-attn --no-build-isolation

# Verify installation
python -c "import flash_attn; print('Flash Attention installed')"
```

**Requirements**:
- NVIDIA GPU with compute capability 7.0+
- CUDA 11.8 or 12.x
- 16GB+ GPU memory for compilation

## Optional: Install SoX CLI

For advanced audio processing:

```bash
# Ubuntu/Debian
sudo apt-get install sox

# macOS
brew install sox

# Fedora/CentOS
sudo dnf install sox

# Verify installation
sox --version
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## WebSocket Endpoints

### Standard WebSocket
```
ws://localhost:8000/ws/voice-clone/{voice_id}
```

### ElevenLabs-Compatible
```
ws://localhost:8000/v1/text-to-speech/{voice_id}/stream-input
```

### HTTP Endpoint
```
POST http://localhost:8000/voice-clone/{voice_id}
```

## Performance Tips

1. **Use GPU**: Ensure CUDA is available for faster inference
2. **Install flash-attn**: 2-3x speedup for attention operations
3. **Use x-vector mode**: Faster than ICL mode (lower quality)
4. **Use smaller model**: 0.6B is faster than 1.7B
5. **Batch requests**: Process multiple texts together

## Documentation

- [README.md](README.md) - Main documentation
- [UV_START.md](UV_START.md) - Detailed uv guide
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - Testing instructions
- [SETUP_FIXES.md](SETUP_FIXES.md) - All fixes applied
- [PYTHON_VERSION_FIX.md](PYTHON_VERSION_FIX.md) - Python version fix
- [BUILD_SYSTEM_FIX.md](BUILD_SYSTEM_FIX.md) - Build system fix
- [IMPORT_FIX.md](IMPORT_FIX.md) - Import path fix
- [DEPENDENCIES_FIX.md](DEPENDENCIES_FIX.md) - Dependencies fix
- [COMPARISON.md](COMPARISON.md) - Comparison with dia2

## Support

For issues:
1. Check the troubleshooting section above
2. Run `uv run test_setup.py` for diagnostics
3. Review the fix documentation in SETUP_FIXES.md
4. Check terminal output for specific error messages

## Summary

1. ✅ Install uv
2. ✅ Run `uv sync` to install dependencies
3. ✅ Run `uv run test_setup.py` to verify setup
4. ✅ Add reference audio to `voices/` directory
5. ✅ Run `uv run qwen_tts_server.py` to start the server
6. ✅ Run tests with `uv run test_qwen_websocket.py --voice reference.wav`

The server is now ready to use! 🎉
