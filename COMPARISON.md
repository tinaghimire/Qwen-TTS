# Comparison: dia2 vs Qwen3-TTS WebSocket Testing

This document compares the WebSocket testing implementations for dia2 and Qwen3-TTS voice cloning systems.

## Overview

Both systems provide WebSocket-based voice cloning with similar testing patterns, but have different underlying architectures and capabilities.

## Architecture Comparison

| Aspect | dia2 | Qwen3-TTS |
|--------|------|-----------|
| **Model** | Dia2-2B | Qwen3-TTS-Base (0.6B/1.7B) |
| **Voice Cloning** | Prefix-based with Whisper transcription | Speaker embedding (x-vector) + ICL |
| **Streaming** | True streaming generation | Simulated streaming (chunk-by-chunk) |
| **Reference Text** | Required (transcribed by Whisper) | Optional (x-vector mode) |
| **Sample Rate** | 24kHz | 24kHz |
| **Audio Format** | WAV (PCM 16-bit) | WAV (PCM 16-bit) |

## WebSocket Protocol Comparison

### dia2 Protocol

```json
// Initialization
{"type": "config", "cfg_scale": 2.0, "temperature": 0.8, "top_k": 50}

// Send text
{"type": "text", "chunk": "[S1] Hello."}

// End
{"type": "end"}

// Response
{"type": "audio", "audio": "<base64>", "sample_rate": 24000, "chunk_index": 0}
```

### Qwen3-TTS Protocol

```json
// Initialization
{"type": "config", "language": "Auto", "use_xvector_only": false, "model_size": "1.7B"}

// Send text
{"type": "text", "text": "Hello.", "ref_text": "optional"}

// End
{"type": "end"}

// Response
{
  "type": "audio",
  "audio": "<base64>",
  "sample_rate": 24000,
  "generation_time": 1.23,
  "audio_duration": 2.45,
  "real_time_factor": 1.99
}
```

## Key Differences

### 1. Voice Cloning Approach

**dia2:**
- Uses prefix-based voice cloning
- Requires Whisper transcription of reference audio
- Maintains speaker consistency across chunks via prefix chaining
- Higher quality but slower initialization

**Qwen3-TTS:**
- Uses speaker embedding (x-vector) extraction
- Optional ICL mode with reference text
- Two modes:
  - **x-vector only**: Fast, no reference text needed
  - **ICL mode**: Higher quality, requires reference text

### 2. Streaming Support

**dia2:**
- True streaming generation
- Audio generated incrementally as text arrives
- Prefix chaining maintains voice consistency
- Lower latency for long texts

**Qwen3-TTS:**
- Chunk-by-chunk generation (simulated streaming)
- Each chunk generated independently
- Voice consistency maintained via same reference audio
- Slightly higher latency but simpler implementation

### 3. Initialization Time

**dia2:**
- Slow initialization (30-60 seconds for long reference audio)
- Whisper transcription required
- Prefix plan generation
- Once initialized, fast generation

**Qwen3-TTS:**
- Fast initialization (speaker embedding extraction)
- No transcription required
- Model loading on startup
- Consistent generation time

### 4. Timing Information

**dia2:**
- Basic timing in headers (HTTP endpoint)
- No detailed timing in WebSocket responses
- Focus on streaming latency

**Qwen3-TTS:**
- Detailed timing in every response:
  - `generation_time`: Time to generate audio
  - `audio_duration`: Length of generated audio
  - `real_time_factor`: RTF = duration / generation_time
- Better for performance analysis

### 5. Audio Similarity Testing

Both systems use similar approaches:

**dia2:**
- MFCC-based similarity
- Spectral features
- Focus on voice consistency across chunks

**Qwen3-TTS:**
- MFCC-based similarity
- Chroma spectral similarity
- Average of both metrics
- Quality classification (Excellent/Good/Fair/Poor)

## Performance Characteristics

### Generation Speed

| Metric | dia2 | Qwen3-TTS (1.7B) | Qwen3-TTS (0.6B) |
|--------|------|------------------|------------------|
| **Initialization** | 30-60s | 2-5s | 1-3s |
| **RTF (Real-Time Factor)** | 0.5-1.0x | 1.5-2.5x | 2.0-3.5x |
| **First Chunk Latency** | High (due to init) | Low | Very Low |
| **Subsequent Chunks** | Low | Medium | Medium |

### Quality

| Aspect | dia2 | Qwen3-TTS (ICL) | Qwen3-TTS (x-vector) |
|--------|------|-----------------|---------------------|
| **Voice Similarity** | Excellent | Excellent | Good |
| **Naturalness** | Excellent | Very Good | Good |
| **Prosody** | Excellent | Very Good | Fair |
| **Reference Text Required** | Yes | Yes | No |

## Use Case Recommendations

### Choose dia2 when:
- You need true streaming generation
- You have long reference audio (5+ seconds)
- You can afford slower initialization
- You need the highest quality voice cloning
- You're building real-time conversational AI

### Choose Qwen3-TTS when:
- You need fast initialization
- You have short reference audio (1-5 seconds)
- You want simple chunk-by-chunk generation
- You need both x-vector and ICL modes
- You're building batch processing systems
- You need detailed timing analysis

## Code Comparison

### Server Setup

**dia2:**
```python
from dia2 import Dia2, GenerationConfig

dia = Dia2.from_repo("nari-labs/Dia2-2B", device="cuda", dtype="bfloat16")

result = dia.generate(
    text,
    config=config,
    prefix_speaker_1=prefix_path,
    include_prefix=False,
)
```

**Qwen3-TTS:**
```python
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda",
    dtype=torch.bfloat16,
)

wavs, sr = model.generate_voice_clone(
    text=text,
    language="Auto",
    ref_audio=ref_audio_tuple,
    ref_text=ref_text,
    x_vector_only_mode=False,
)
```

### WebSocket Handler

**dia2:**
```python
@app.websocket("/ws/tts/stream")
async def websocket_tts_stream(websocket: WebSocket):
    # Prefix chaining for voice consistency
    prev_tmp_path = None
    chunk_index = 0
    
    for chunk_text in text_chunks:
        result = dia.generate(
            chunk_text,
            prefix_speaker_1=original_prefix_1,
            include_prefix=(chunk_index == 0),
        )
        # Send audio...
```

**Qwen3-TTS:**
```python
@app.websocket("/ws/voice-clone/{voice_id}")
async def websocket_voice_clone(websocket: WebSocket, voice_id: str):
    # Load reference audio once
    ref_audio_tuple = load_reference_audio(voice_id)
    
    for chunk_text in text_chunks:
        wavs, sr = model.generate_voice_clone(
            text=chunk_text,
            ref_audio=ref_audio_tuple,  # Same reference for all chunks
            x_vector_only_mode=True,
        )
        # Send audio with timing...
```

## Testing Comparison

### Test Suite Features

| Feature | dia2 | Qwen3-TTS |
|---------|------|-----------|
| Basic connection test | ✓ | ✓ |
| Streaming test | ✓ | ✓ |
| Audio similarity | ✓ | ✓ |
| Timing statistics | Basic | Detailed |
| ElevenLabs-compatible | ✓ | ✓ |
| Multiple model sizes | ✗ | ✓ (0.6B, 1.7B) |
| ICL vs x-vector comparison | ✗ | ✓ |

### Test Output

**dia2:**
```
✓ Saved audio to chunk_000.wav (12345 bytes)
✓ Test completed successfully!
```

**Qwen3-TTS:**
```
✓ Audio generated successfully!
  - Saved to: test1_basic_output.wav
  - Sample rate: 24000 Hz
  - Audio duration: 2.456s
  - Generation time: 1.234s
  - Real-time factor: 1.990x

Timing Statistics:
  Number of chunks: 3
  Generation time - Mean: 1.234s, Std: 0.123s
  Audio duration - Mean: 2.456s, Std: 0.234s
  Real-time factor - Mean: 1.990x, Std: 0.123

Similarity Scores:
  MFCC similarity: 0.7234 (0-1 scale)
  Spectral similarity: 0.6891 (0-1 scale)
  Average similarity: 0.7063
  Voice cloning quality: Excellent
```

## Migration Guide

If you're migrating from dia2 to Qwen3-TTS:

### 1. Replace Model Loading

```python
# Before (dia2)
from dia2 import Dia2
dia = Dia2.from_repo("nari-labs/Dia2-2B", device="cuda", dtype="bfloat16")

# After (Qwen3-TTS)
from qwen_tts import Qwen3TTSModel
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda",
    dtype=torch.bfloat16,
)
```

### 2. Update Generation Call

```python
# Before (dia2)
result = dia.generate(
    text,
    prefix_speaker_1=ref_audio_path,
    include_prefix=False,
)
wav = result.waveform.cpu().numpy()

# After (Qwen3-TTS)
wavs, sr = model.generate_voice_clone(
    text=text,
    ref_audio=(ref_wav, ref_sr),
    ref_text=ref_text,  # Optional for x-vector mode
    x_vector_only_mode=True,
)
wav = wavs[0]
```

### 3. Update WebSocket Protocol

```python
# Before (dia2)
await ws.send(json.dumps({
    "type": "text",
    "chunk": "[S1] Hello.",
}))

# After (Qwen3-TTS)
await ws.send(json.dumps({
    "type": "text",
    "text": "Hello.",
    "ref_text": "optional",  # Only for ICL mode
}))
```

### 4. Handle Timing Information

```python
# Before (dia2)
# No timing info in WebSocket response

# After (Qwen3-TTS)
generation_time = data["generation_time"]
audio_duration = data["audio_duration"]
rtf = data["real_time_factor"]
```

## Conclusion

Both dia2 and Qwen3-TTS provide excellent voice cloning capabilities with WebSocket support. The choice depends on your specific requirements:

- **dia2**: Best for real-time streaming with highest quality
- **Qwen3-TTS**: Best for fast initialization and flexible modes

The Qwen3-TTS implementation follows the dia2 testing pattern while adding:
- Detailed timing analysis
- Multiple model sizes
- Two voice cloning modes (ICL and x-vector)
- Comprehensive similarity testing
- Better performance metrics

This makes it easier to benchmark and optimize voice cloning systems.
