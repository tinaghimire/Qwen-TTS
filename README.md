# Qwen3-TTS WebSocket Server

A WebSocket server for real-time voice cloning using Qwen3-TTS models.

All scripts read configuration from `.env` file.

## Features

- Real-time voice cloning via WebSocket
- Streaming support with chunked audio output
- Multiple voice profiles management
- Performance analytics

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running Scripts](#running-scripts)
- [WebSocket Endpoint](#websocket-endpoint)
- [WebSocket Protocol](#websocket-protocol)
- [Testing](#testing)

## Installation

### Prerequisites

- Python 3.10, 3.11, or 3.12
- CUDA 13.0
- NVIDIA GPU with compute capability 7.0+

### Setup

```bash
git clone https://github.com/NG-Joseph/ultravox-llama-serving.git -b tts_websocket
cd ultravox-llama-serving/

uv sync

uv run test_setup.py
```

### Environment Configuration

```bash
cp .env.websocket.example .env
```

## Quick Start

### Configure Environment

Edit `.env`:

```env
HOST=0.0.0.0
PORT=8000
MODEL_SIZE=1.7B
DEFAULT_LANGUAGE=Auto
DEVICE=cuda
VOICES_DIR=voices
CHUNK_SIZE=512
TEST_VOICE_ID=english_voice
TEST_OUTPUT_DIR=test_output
TEST_TYPE=all
```

### Start Server

```bash
uv run qwen_tts_server.py
```

### Test Server

```bash
uv run test_qwen_websocket.py
```

## Configuration

### Environment Variables

All configuration is in `.env`. See `.env.websocket.example` for all options.

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `MODEL_SIZE` | `1.7B` | Model size |
| `DEFAULT_LANGUAGE` | `Auto` | Default language |
| `DEVICE` | `cuda` | Device |
| `VOICES_DIR` | `voices` | Voices directory |
| `CHUNK_SIZE` | `512` | Audio chunk size |
| `TEST_VOICE_ID` | `english_voice` | Test voice ID |
| `TEST_OUTPUT_DIR` | `test_output` | Test output directory |
| `TEST_TYPE` | `all` | Test type |

### Model Sizes

- 0.6B: Faster, lower quality
- 1.7B: Slower, higher quality

## Running Scripts

All scripts read configuration from `.env` only.

```bash
uv run qwen_tts_server.py
uv run test_qwen_websocket.py
```

## WebSocket Endpoint

### URL Format

```
ws://{host}:{port}/ws/voice-clone/{voice_id}
```

### Examples

```bash
ws://localhost:8000/ws/voice-clone/english_voice
ws://192.168.1.100:8000/ws/voice-clone/my_voice
```

### Parameters

| Parameter | Type | Required |
|-----------|------|----------|
| `{host}` | string | Yes |
| `{port}` | number | Yes |
| `{voice_id}` | string | Yes |

## WebSocket Protocol

### Request Format

1. **Configuration**
```json
{
  "type": "config",
  "language": "Auto",
  "use_xvector_only": false,
  "model_size": "1.7B",
  "chunk_size": 512
}
```

2. **Text to Synthesize**
```json
{
  "type": "text",
  "text": "Hello world",
  "ref_text": "Hello world"
}
```

3. **End Connection**
```json
{
  "type": "end"
}
```

### Response Format

1. **Config Acknowledgment**
```json
{
  "type": "config_ack"
}
```

2. **Audio Response**
```json
{
  "type": "audio",
  "audio": "<base64>",
  "sample_rate": 24000,
  "generation_time": 1.234,
  "audio_duration": 2.451,
  "real_time_factor": 1.987
}
```

3. **Error Response**
```json
{
  "type": "error",
  "message": "Error"
}
```

4. **End Acknowledgment**
```json
{
  "type": "end_ack"
}
```

## Testing

### Run All Tests

```bash
uv run test_qwen_websocket.py
```

Test outputs are saved to `test_output/`:

- `test1_basic_output.wav`
- `test2_chunk_*.wav`
- `test3_generated.wav`
- `test_results.json`

## Cloud Deployment

#### Timing Statistics

- **Generation Time**: Time taken to generate
- **Real-Time Factor (RTF)**: `audio_duration / generation_time`
  - RTF > 1.0: Good performance
  - RTF < 1.0: Slower than real-time

#### Audio Similarity Scores

Scores range from 0 to 1:

- **Excellent**: > 0.7
- **Good**: 0.5 - 0.7
- **Fair**: 0.3 - 0.5
- **Poor**: < 0.3

## Troubleshooting

### Server won't start

```bash
lsof -i :8000
uv run test_setup.py
```

### WebSocket connection fails

```bash
curl http://localhost:8000/health
curl http://localhost:8000/voices
```

### Voice not found

```bash
ls voices/english_voice/
```

### CUDA out of memory

Edit `.env`:

```env
MODEL_SIZE=0.6B
```

## Additional Documentation

- **QUICKSTART.md** - Quick start guide
- **SETUP_GUIDE.md** - Complete setup guide
- **.env.websocket.example** - Environment variables

## Summary

1. Copy `.env.websocket.example` to `.env`
2. Configure `.env` with your settings
3. Run scripts without arguments

All configuration is from `.env` only.