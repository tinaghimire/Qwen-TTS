# Qwen3-TTS WebSocket Testing Setup

This directory contains a FastAPI WebSocket server and comprehensive test suite for the Qwen3-TTS Base model (voice cloning), following the testing pattern from dia2.

## Features

1. **WebSocket Server** (`qwen_tts_server.py`)
   - Real-time voice cloning via WebSocket
   - ElevenLabs-compatible endpoint
   - HTTP endpoint for batch processing
   - Time analysis for audio generation

2. **Test Suite** (`test_qwen_websocket.py`)
   - Basic WebSocket connection testing
   - Streaming support verification
   - Audio similarity testing (MFCC and spectral)
   - ElevenLabs-compatible endpoint testing
   - Comprehensive timing statistics

## Installation

### Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **CUDA**: 13.0 (for GPU acceleration)
- **GPU**: NVIDIA GPU with compute capability 7.0+ (e.g., RTX 5070 Ti, RTX 4090, etc.)

Check your CUDA version:
```bash
nvidia-smi  # Look for "CUDA Version"
```

### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (requires Python 3.10-3.12)
# This will automatically install:
# - PyTorch with CUDA 13.0 support
# - flash-attn for optimized attention computation
# - All other required dependencies
uv sync
```

> **Note**: See [UV_START.md](UV_START.md) for a detailed uv quick start guide.
> 
> **Requirements**: Python 3.10, 3.11, or 3.12 (accelerate==1.12.0 requires Python >=3.10)

### Using pip

```bash
# Requires Python 3.10+
# Install PyTorch with CUDA 13.0 support
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchaudio

# Install other dependencies
pip install fastapi uvicorn websockets librosa scipy numpy soundfile huggingface_hub transformers==4.57.3 accelerate==1.12.0 einops sox onnxruntime

# Install flash-attn (optional, for faster inference)
pip install flash-attn --no-build-isolation
```

> **Note**: `flash-attn` and `sox` command-line tool are optional. The system will work without them, but with slower inference.

## Setup

1. **Prepare reference audio files:**
   
   **Option A: Manual setup**
   ```bash
   mkdir -p voices
   # Create a directory for each voice (voice_id is the directory name)
   mkdir -p voices/reference
   # Copy your reference audio file to the voice directory
   cp /path/to/reference.wav voices/reference/
   ```
   
   **Option B: Using the helper script (recommended)**
   ```bash
   # Add a new voice (moves the file)
   python add_voice.py --voice-id alice --audio-file /path/to/alice.wav
   
   # Add a new voice (copies the file)
   python add_voice.py --voice-id bob --audio-file /path/to/bob.mp3 --copy
   
   # List all available voices
   python add_voice.py --list
   ```

   **Voice directory structure:**
   ```
   voices/
       ├── reference/
       │   └── reference.wav
       ├── alice/
       │   └── alice.mp3
       └── bob/
           └── bob.flac
   ```

   The voice_id is the directory name (filename without extension). Supported audio formats: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`.

2. **Start the server:**
   ```bash
   # Default settings (port 8000, model size 1.7B)
   uv run qwen_tts_server.py
   
   # Custom settings
   uv run qwen_tts_server.py --host 0.0.0.0 --port 8001 --model-size 0.6B
   ```

## Server Endpoints

### WebSocket Endpoints

#### 1. Standard WebSocket Endpoint
```
ws://localhost:8000/ws/voice-clone/{voice_id}
```

Where `{voice_id}` is the voice directory name (e.g., `reference`, `alice`, `bob`).

**Protocol:**
```json
// Client sends config (optional)
{"type": "config", "language": "Auto", "use_xvector_only": false, "model_size": "1.7B"}

// Client sends text to synthesize
{"type": "text", "text": "Hello world", "ref_text": "optional reference text"}

// Client sends end signal
{"type": "end"}

// Server responds with audio
{
  "type": "audio",
  "audio": "<base64_wav>",
  "sample_rate": 24000,
  "generation_time": 1.23,
  "audio_duration": 2.45,
  "real_time_factor": 1.99
}
```

#### 2. ElevenLabs-Compatible Endpoint
```
ws://localhost:8000/v1/text-to-speech/{voice_id}/stream-input
```

Where `{voice_id}` is the voice directory name (e.g., `reference`, `alice`, `bob`).

**Protocol:**
```json
// Initialization
{"text": " ", "generation_config": {"language": "Auto", "use_xvector_only": true}}

// Stream text
{"text": "Hello world."}

// Close
{"text": ""}

// Server response
{"audio": "<base64_pcm_wav>", "isFinal": false}
// Final
{"isFinal": true}
```

### HTTP Endpoint

```
POST /voice-clone/{voice_id}
Content-Type: application/json

{
  "target_text": "Hello world",
  "ref_text": "optional reference text",
  "language": "Auto",
  "use_xvector_only": false,
  "model_size": "1.7B"
}
```

Where `{voice_id}` is the voice directory name (e.g., `reference`, `alice`, `bob`).

**Response:** WAV audio file with timing headers:
- `X-Generation-Time`: Time taken to generate audio (seconds)
- `X-Audio-Duration`: Duration of generated audio (seconds)
- `X-Sample-Rate`: Sample rate (Hz)

### List Voices Endpoint

```
GET /voices
```

**Response:** List of all available voices:
```json
{
  "voices": [
    {"id": "reference", "audio_file": "reference/reference.wav"},
    {"id": "alice", "audio_file": "alice/alice.mp3"}
  ],
  "count": 2,
  "voices_dir": "/path/to/voices"
}
```

## Running Tests

### Run All Tests
```bash
uv run test_qwen_websocket.py --voice reference
```

### Run Specific Tests
```bash
# Basic connection test
uv run test_qwen_websocket.py --voice reference --test basic

# Streaming test
uv run test_qwen_websocket.py --voice reference --test streaming

# Audio similarity test
uv run test_qwen_websocket.py --voice reference --test similarity --ref-audio /path/to/ref.wav

# ElevenLabs-compatible test
uv run test_qwen_websocket.py --voice reference --test elevenlabs
```

### Test Options
```bash
uv run test_qwen_websocket.py \
    --voice reference \
    --host localhost \
    --port 8000 \
    --ref-audio /path/to/reference.wav \
    --test all
```

**Note:** The `--voice` parameter now expects the voice directory name (voice_id), not the full filename.

## Test Output

All test outputs are saved to the `test_output/` directory:

- `test1_basic_output.wav` - Basic test output
- `test2_chunk_00.wav`, `test2_chunk_01.wav`, ... - Streaming test chunks
- `test3_generated.wav` - Audio similarity test output
- `test4_el_chunk_00.wav`, ... - ElevenLabs-compatible test chunks
- `test_results.json` - Comprehensive test results with timing statistics

## Understanding Test Results

### Timing Statistics

The test suite provides detailed timing analysis:

- **Generation Time**: Time taken to generate audio
- **Audio Duration**: Length of generated audio
- **Real-Time Factor (RTF)**: `audio_duration / generation_time`
  - RTF > 1.0: Faster than real-time (good)
  - RTF < 1.0: Slower than real-time

Example output:
```
Timing Statistics:
  Number of chunks: 3
  Generation time - Mean: 1.234s, Std: 0.123s
  Audio duration - Mean: 2.456s, Std: 0.234s
  Real-time factor - Mean: 1.990x, Std: 0.123
```

### Audio Similarity Scores

The similarity test computes two metrics:

1. **MFCC Similarity**: Based on Mel-frequency cepstral coefficients
2. **Spectral Similarity**: Based on chroma spectral features

Both scores range from 0 (dissimilar) to 1 (identical).

Quality interpretation:
- **Excellent**: > 0.7
- **Good**: 0.5 - 0.7
- **Fair**: 0.3 - 0.5
- **Poor**: < 0.3

Example output:
```
Similarity Scores:
  MFCC similarity: 0.7234 (0-1 scale)
  Spectral similarity: 0.6891 (0-1 scale)
  Average similarity: 0.7063

Voice cloning quality: Excellent
```

## Streaming Support

Qwen3-TTS supports two modes:

### 1. ICL Mode (In-Context Learning)
- Requires reference text
- Higher quality voice cloning
- Use `use_xvector_only: false`

### 2. X-Vector Only Mode
- No reference text needed
- Faster generation
- Suitable for streaming
- Use `use_xvector_only: true`

## Comparison with dia2

This setup follows the dia2 testing pattern:

| Feature | dia2 | Qwen3-TTS |
|---------|------|-----------|
| WebSocket | ✓ | ✓ |
| Streaming | ✓ | ✓ |
| Time Analysis | ✓ | ✓ |
| Audio Similarity | ✓ | ✓ |
| Voice Cloning | ✓ | ✓ |
| Model | Dia2-2B | Qwen3-TTS-Base |

## Troubleshooting

### Server won't start
- Check if port is already in use: `lsof -i :8000`
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

### WebSocket connection fails
- Verify server is running: `curl http://localhost:8000/health`
- Check firewall settings
- Ensure voice directory exists in `voices/` directory (e.g., `voices/reference/`)
- Ensure audio file exists in the voice directory
- List available voices: `curl http://localhost:8000/voices`

### Audio generation errors
- Check reference audio format (should be WAV, mono)
- Verify reference text matches audio content (for ICL mode)
- Check GPU memory: `nvidia-smi`

### Similarity scores are low
- Ensure reference audio is clear and has good quality
- Use longer reference audio (5-10 seconds recommended)
- Check that reference text accurately transcribes the audio

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

This code follows the same license as Qwen3-TTS (Apache-2.0).
