# Qwen3-TTS WebSocket Testing Guide

This guide explains how to test the Qwen3-TTS Base model for voice cloning using WebSocket, following the dia2 testing pattern.

## What Was Created

### 1. Server Implementation (`qwen_tts_server.py`)
A FastAPI WebSocket server with:
- **Standard WebSocket endpoint**: `/ws/voice-clone/{voice_id}`
- **ElevenLabs-compatible endpoint**: `/v1/text-to-speech/{voice_id}/stream-input`
- **HTTP endpoint**: `/voice-clone/{voice_id}`
- Time analysis for audio generation
- Support for both ICL and x-vector modes

### 2. Test Suite (`test_qwen_websocket.py`)
Comprehensive testing with:
- Basic WebSocket connection testing
- Streaming support verification
- Audio similarity testing (MFCC and spectral)
- ElevenLabs-compatible endpoint testing
- Detailed timing statistics

### 3. Helper Scripts
- `quick_start.py`: Simple example to get started
- `example_usage.py`: Various usage examples
- `setup.sh`: Automated setup script

### 4. Documentation
- `README.md`: Complete documentation
- `COMPARISON.md`: Comparison with dia2
- `TESTING_GUIDE.md`: This file

## Quick Start

### Step 1: Install Dependencies
```bash
cd /home/ml/workspaces/kristina/Qwen3-TTS-finetuning

# Using uv (recommended)
# Requires Python 3.10-3.12
uv sync

# Or using pip (requires Python 3.10+)
pip install fastapi uvicorn websockets librosa scipy numpy torch soundfile huggingface_hub transformers
```

> **Note**: This project requires Python 3.10, 3.11, or 3.12 due to accelerate==1.12.0 requirements.

### Step 2: Prepare Reference Audio
```bash
mkdir -p voices
# Copy your reference audio file
cp /path/to/your/reference.wav voices/reference.wav
```

### Step 3: Start the Server
```bash
# Using uv
uv run qwen_tts_server.py --port 8000

# Or using python directly
python qwen_tts_server.py --port 8000
```

### Step 4: Run Tests
```bash
# Run all tests
uv run test_qwen_websocket.py --voice reference.wav

# Run specific test
uv run test_qwen_websocket.py --voice reference.wav --test basic
```

## Testing Features

### 1. WebSocket Connection Testing
Verifies basic connectivity and audio generation:
```bash
uv run test_qwen_websocket.py --voice reference.wav --test basic
```

**Output:**
```
TEST 1: Basic WebSocket Connection
✓ Connected successfully!
✓ Sent config message
✓ Config acknowledged
✓ Audio generated successfully!
  - Saved to: test_output/test1_basic_output.wav
  - Sample rate: 24000 Hz
  - Audio duration: 2.456s
  - Generation time: 1.234s
  - Real-time factor: 1.990x
```

### 2. Streaming Support Testing
Tests multiple text chunks:
```bash
uv run test_qwen_websocket.py --voice reference.wav --test streaming
```

**Output:**
```
TEST 2: Streaming Support (Multiple Chunks)
✓ Connected successfully!
[1/3] Generating: 'Hello, this is the first chunk.'
  ✓ Generated in 1.234s (RTF: 1.990x)
[2/3] Generating: 'This is the second chunk of text.'
  ✓ Generated in 1.123s (RTF: 2.180x)
[3/3] Generating: 'And this is the third and final chunk.'
  ✓ Generated in 1.345s (RTF: 1.825x)

Timing Statistics:
  Number of chunks: 3
  Generation time - Mean: 1.234s, Std: 0.123s
  Audio duration - Mean: 2.456s, Std: 0.234s
  Real-time factor - Mean: 1.990x, Std: 0.123
```

### 3. Audio Similarity Testing
Compares generated audio with reference:
```bash
uv run test_qwen_websocket.py --voice reference.wav --test similarity
```

**Output:**
```
TEST 3: Audio Similarity Testing
Reference audio: voices/reference.wav
✓ Connected successfully!

Analyzing audio similarity...

Similarity Scores:
  MFCC similarity: 0.7234 (0-1 scale)
  Spectral similarity: 0.6891 (0-1 scale)
  Average similarity: 0.7063

Voice cloning quality: Excellent
```

### 4. ElevenLabs-Compatible Testing
Tests the ElevenLabs-compatible endpoint:
```bash
uv run test_qwen_websocket.py --voice reference.wav --test elevenlabs
```

## Understanding the Results

### Real-Time Factor (RTF)
- **RTF > 1.0**: Faster than real-time (good)
- **RTF < 1.0**: Slower than real-time
- **RTF = 2.0**: Generates 2 seconds of audio in 1 second

### Audio Similarity Scores
- **0.7 - 1.0**: Excellent voice cloning
- **0.5 - 0.7**: Good voice cloning
- **0.3 - 0.5**: Fair voice cloning
- **0.0 - 0.3**: Poor voice cloning

## Voice Cloning Modes

### ICL Mode (In-Context Learning)
- **Use when**: You have reference text and want highest quality
- **Requires**: Reference text that matches reference audio
- **Quality**: Excellent
- **Speed**: Medium

```python
{
    "type": "config",
    "use_xvector_only": False,  # ICL mode
}
{
    "type": "text",
    "text": "Target text",
    "ref_text": "Reference text matching audio"
}
```

### X-Vector Only Mode
- **Use when**: You don't have reference text or need speed
- **Requires**: Only reference audio
- **Quality**: Good
- **Speed**: Fast

```python
{
    "type": "config",
    "use_xvector_only": True,  # X-vector mode
}
{
    "type": "text",
    "text": "Target text"
    # No ref_text needed
}
```

## WebSocket Protocol Examples

### Standard Endpoint

```python
import asyncio
import json
import websockets

async def test():
    uri = "ws://localhost:8000/ws/voice-clone/reference.wav"
    
    async with websockets.connect(uri) as ws:
        # Configure
        await ws.send(json.dumps({
            "type": "config",
            "language": "Auto",
            "use_xvector_only": True,
            "model_size": "1.7B",
        }))
        
        # Wait for ack
        await ws.recv()
        
        # Generate
        await ws.send(json.dumps({
            "type": "text",
            "text": "Hello world",
        }))
        
        # Receive audio
        response = await ws.recv()
        data = json.loads(response)
        
        if data["type"] == "audio":
            audio_b64 = data["audio"]
            # Process audio...
        
        # End
        await ws.send(json.dumps({"type": "end"}))

asyncio.run(test())
```

### ElevenLabs-Compatible Endpoint

```python
import asyncio
import json
import websockets

async def test():
    uri = "ws://localhost:8000/v1/text-to-speech/reference.wav/stream-input"
    
    async with websockets.connect(uri) as ws:
        # Initialize
        await ws.send(json.dumps({
            "text": " ",
            "generation_config": {
                "language": "Auto",
                "use_xvector_only": True,
            }
        }))
        
        # Stream text
        await ws.send(json.dumps({"text": "Hello world."}))
        
        # Receive audio
        response = await ws.recv()
        data = json.loads(response)
        
        if "audio" in data:
            audio_b64 = data["audio"]
            # Process audio...
        
        # Close
        await ws.send(json.dumps({"text": ""}))

asyncio.run(test())
```

## Performance Optimization

### For Faster Generation
1. Use x-vector mode (`use_xvector_only: True`)
2. Use smaller model (`model_size: "0.6B"`)
3. Reduce `max_new_tokens` if possible

### For Better Quality
1. Use ICL mode (`use_xvector_only: False`)
2. Use larger model (`model_size: "1.7B"`)
3. Provide accurate reference text
4. Use longer reference audio (5-10 seconds)

### For Streaming
1. Use x-vector mode for faster chunk generation
2. Keep chunks short (1-2 sentences)
3. Use the same reference audio for all chunks

## Troubleshooting

### Server won't start
```bash
# Check if port is in use
lsof -i :8000

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

### WebSocket connection fails
```bash
# Check server health
curl http://localhost:8000/health

# Check if voice file exists
ls -la voices/reference.wav
```

### Audio generation errors
- Ensure reference audio is WAV format, mono
- Check reference text matches audio (for ICL mode)
- Verify GPU memory: `nvidia-smi`

### Low similarity scores
- Use longer reference audio (5-10 seconds)
- Ensure reference audio is clear and high quality
- Check that reference text accurately transcribes audio
- Try ICL mode for better quality

## Comparison with dia2

| Feature | dia2 | Qwen3-TTS |
|---------|------|-----------|
| WebSocket | ✓ | ✓ |
| Streaming | ✓ | ✓ |
| Time Analysis | Basic | Detailed |
| Audio Similarity | ✓ | ✓ |
| Voice Cloning | ✓ | ✓ |
| Model Sizes | 1 (2B) | 2 (0.6B, 1.7B) |
| Cloning Modes | 1 (prefix) | 2 (ICL, x-vector) |
| Initialization | Slow (30-60s) | Fast (2-5s) |
| RTF | 0.5-1.0x | 1.5-3.5x |

See `COMPARISON.md` for detailed comparison.

## Next Steps

1. **Run the quick start**:
   ```bash
   uv run quick_start.py
   ```

2. **Explore examples**:
   ```bash
   uv run example_usage.py
   ```

3. **Run comprehensive tests**:
   ```bash
   uv run test_qwen_websocket.py --voice reference.wav
   ```

4. **Review results**:
   ```bash
   ls -la test_output/
   cat test_output/test_results.json
   ```

5. **Integrate into your application**:
   - Use the WebSocket endpoints for real-time applications
   - Use the HTTP endpoint for batch processing
   - Choose ICL mode for quality, x-vector for speed

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the test output in `test_output/`
3. Examine the timing statistics in `test_results.json`
4. Compare with dia2 implementation in `COMPARISON.md`

## License

This code follows the same license as Qwen3-TTS (Apache-2.0).
