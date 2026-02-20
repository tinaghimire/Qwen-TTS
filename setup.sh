#!/bin/bash

# Setup script for Qwen3-TTS WebSocket testing

set -e

echo "=========================================="
echo "Qwen3-TTS WebSocket Testing Setup"
echo "=========================================="
echo ""

# Check if uv is installed
echo "Checking for uv..."
if ! command -v uv &> /dev/null; then
    echo "✗ uv is not installed"
    echo ""
    echo "Install uv with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo ""
    echo "Or visit: https://github.com/astral-sh/uv"
    exit 1
fi
echo "✓ uv is installed: $(uv --version)"

# Check Python version
echo ""
echo "Checking Python version..."
python_version=$(uv python list | grep "3\.\(10\|11\|12\)" | head -1 | awk '{print $2}')
if [ -z "$python_version" ]; then
    echo "✗ No compatible Python version found (requires 3.10-3.12)"
    echo "  Install Python with: uv python install 3.11"
    exit 1
fi
echo "✓ Python version: $python_version"

# Create virtual environment and sync dependencies
echo ""
echo "Creating virtual environment and syncing dependencies..."
uv sync
echo "✓ Virtual environment created and dependencies installed"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p voices
mkdir -p test_output
echo "✓ Directories created"

# Verify setup
echo ""
echo "Verifying setup..."
uv run test_setup.py
if [ $? -eq 0 ]; then
    echo "✓ Setup verification passed"
else
    echo "⚠ Setup verification failed (some dependencies may be missing)"
fi

# Check for reference audio
echo ""
echo "Checking for reference audio..."
if [ -f "voices/reference.wav" ]; then
    echo "✓ Found reference audio: voices/reference.wav"
else
    echo "⚠ No reference audio found in voices/"
    echo "  Please copy a reference audio file to voices/reference.wav"
    echo "  Example: cp /path/to/your/audio.wav voices/reference.wav"
fi

# Print summary
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. (If needed) Copy a reference audio file:"
echo "   cp /path/to/your/audio.wav voices/reference.wav"
echo ""
echo "2. Start the server:"
echo "   uv run qwen_tts_server.py"
echo ""
echo "3. Run tests:"
echo "   uv run test_qwen_websocket.py --voice reference.wav"
echo ""
echo "4. Or try the quick start:"
echo "   uv run quick_start.py"
echo ""
echo "5. Check out examples:"
echo "   uv run example_usage.py"
echo ""
echo "For more information, see README.md"
echo ""
