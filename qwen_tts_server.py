"""
FastAPI WebSocket Server for Qwen3-TTS Voice Cloning

This server provides:
1. WebSocket endpoint for streaming voice cloning
2. Time analysis for audio generation
3. Support for both streaming and non-streaming modes

Run:
    python qwen_tts_server.py --port 8000

Example WebSocket usage:
    ws://localhost:8000/ws/voice-clone/{voice_id}

Where voice_id is the voice directory name (e.g., "reference").
The audio file should be placed in voices/{voice_id}/ directory.

Voice directory structure:
    voices/
        ├── reference/
        │   └── reference.wav
        ├── alice/
        │   └── alice.mp3
        └── bob/
            └── bob.flac
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel, Field
from huggingface_hub import snapshot_download

# Add Qwen3-TTS to Python path
qwen_tts_path = Path(__file__).parent / "Qwen3-TTS"
if qwen_tts_path.exists():
    sys.path.insert(0, str(qwen_tts_path))

from qwen_tts import Qwen3TTSModel

# ============================================================================
# Configuration
# ============================================================================
DEFAULT_MODEL_SIZE = "1.7B"
DEFAULT_LANGUAGE = "Auto"
VOICES_DIR = Path("voices")
VOICES_DIR.mkdir(exist_ok=True)

# Supported audio file extensions
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}


def get_voice_audio_path(voice_id: str) -> Optional[Path]:
    """
    Get the audio file path for a given voice_id.
    
    The voice_id is the directory name (filename without extension).
    The audio file is searched inside voices/{voice_id}/ directory.
    
    Args:
        voice_id: Voice identifier (directory name)
    
    Returns:
        Path to the audio file, or None if not found
    """
    voice_dir = VOICES_DIR / voice_id
    
    if not voice_dir.exists() or not voice_dir.is_dir():
        return None
    
    # Search for audio files in the voice directory
    for ext in AUDIO_EXTENSIONS:
        # Check for files with this extension
        for audio_file in voice_dir.glob(f"*{ext}"):
            if audio_file.is_file():
                return audio_file
    
    return None


def list_available_voices() -> list[dict]:
    """
    List all available voices in the voices directory.
    
    Returns:
        List of voice dictionaries with 'id' and 'audio_file' keys
    """
    voices = []
    
    if not VOICES_DIR.exists():
        return voices
    
    for voice_dir in VOICES_DIR.iterdir():
        if voice_dir.is_dir():
            audio_file = get_voice_audio_path(voice_dir.name)
            if audio_file:
                voices.append({
                    "id": voice_dir.name,
                    "audio_file": str(audio_file.relative_to(VOICES_DIR))
                })
    
    return voices

# ============================================================================
# Global Model Loading
# ============================================================================
print("Loading Qwen3-TTS Base model...")
_model: Optional[Qwen3TTSModel] = None


def get_model(model_size: str = DEFAULT_MODEL_SIZE) -> Qwen3TTSModel:
    """Get or load the Qwen3-TTS Base model."""
    global _model
    if _model is None:
        model_path = snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base")
        _model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map="cuda",
            dtype=torch.bfloat16,
            attn_implementation="kernels-community/flash-attn3",
        )
        print(f"✓ Model loaded: {model_size}")
    return _model


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode mono float32 audio as WAV bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm16.tobytes())
    return buf.getvalue()


def normalize_audio(wav, eps=1e-12, clip=True):
    """Normalize audio to float32 in [-1, 1] range."""
    x = np.asarray(wav)
    
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    
    if clip:
        y = np.clip(y, -1.0, 1.0)
    
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    
    return y


def audio_to_tuple(audio):
    """Convert audio input to (wav, sr) tuple."""
    if audio is None:
        return None
    
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        wav = normalize_audio(wav)
        return wav, int(sr)
    
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        sr = int(audio["sampling_rate"])
        wav = normalize_audio(audio["data"])
        return wav, sr
    
    return None


# ============================================================================
# FastAPI App
# ============================================================================
app = FastAPI(
    title="Qwen3-TTS Voice Cloning Server",
    description="WebSocket-based voice cloning with Qwen3-TTS Base model",
)


class VoiceCloneRequest(BaseModel):
    """Request body for HTTP voice cloning endpoint."""
    
    target_text: str = Field(..., description="Text to synthesize with cloned voice")
    ref_text: Optional[str] = Field(None, description="Reference text (transcript of reference audio)")
    language: str = Field(default=DEFAULT_LANGUAGE, description="Language for synthesis")
    use_xvector_only: bool = Field(default=False, description="Use x-vector only mode")
    model_size: str = Field(default=DEFAULT_MODEL_SIZE, description="Model size (0.6B or 1.7B)")


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_size": DEFAULT_MODEL_SIZE if _model else None,
        "voices_dir": str(VOICES_DIR.absolute()),
    }


@app.get("/voices")
def list_voices() -> dict:
    """List all available voices."""
    voices = list_available_voices()
    return {
        "voices": voices,
        "count": len(voices),
        "voices_dir": str(VOICES_DIR.absolute())
    }


@app.post("/voice-clone/{voice_id}")
def voice_clone_http(voice_id: str, request: VoiceCloneRequest) -> Response:
    """
    HTTP endpoint for voice cloning.
    
    Args:
        voice_id: Voice identifier (directory name, e.g., "reference")
        request: Voice cloning parameters
    
    Returns:
        WAV audio file
    """
    # Load reference audio from voice directory
    ref_audio_path = get_voice_audio_path(voice_id)
    if ref_audio_path is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Voice '{voice_id}' not found. Expected audio file in voices/{voice_id}/"
        )
    
    # Load model
    model = get_model(request.model_size)
    
    # Load reference audio
    import librosa
    ref_wav, ref_sr = librosa.load(str(ref_audio_path), sr=None, mono=True)
    ref_audio_tuple = (ref_wav.astype(np.float32), int(ref_sr))
    
    # Generate audio
    start_time = time.time()
    wavs, sr = model.generate_voice_clone(
        text=request.target_text,
        language=request.language,
        ref_audio=ref_audio_tuple,
        ref_text=request.ref_text,
        x_vector_only_mode=request.use_xvector_only,
        max_new_tokens=2048,
    )
    generation_time = time.time() - start_time
    
    # Convert to WAV bytes
    wav_bytes_data = wav_bytes(wavs[0], sr)
    
    # Return audio with timing info in headers
    return Response(
        content=wav_bytes_data,
        media_type="audio/wav",
        headers={
            "X-Generation-Time": str(generation_time),
            "X-Audio-Duration": str(len(wavs[0]) / sr),
            "X-Sample-Rate": str(sr),
        }
    )


@app.websocket("/ws/voice-clone/{voice_id}")
async def websocket_voice_clone(websocket: WebSocket, voice_id: str):
    """
    WebSocket endpoint for streaming voice cloning.
    
    Protocol:
    - Client sends initialization (optional):
      {"type": "config", "language": "Auto", "use_xvector_only": false, "model_size": "1.7B"}
    
    - Client sends text to synthesize:
      {"type": "text", "text": "Hello world", "ref_text": "optional reference text"}
    
    - Client sends close signal:
      {"type": "end"}
    
    - Server sends audio response:
      {"type": "audio", "audio": "<base64_wav>", "sample_rate": 24000, 
       "generation_time": 1.23, "audio_duration": 2.45}
    
    - Server sends errors:
      {"type": "error", "message": "error description"}
    """
    await websocket.accept()
    
    # Load reference audio from voice directory
    ref_audio_path = get_voice_audio_path(voice_id)
    if ref_audio_path is None:
        await websocket.send_json({
            "type": "error",
            "message": f"Voice '{voice_id}' not found. Expected audio file in voices/{voice_id}/"
        })
        await websocket.close()
        return
    
    # Load reference audio
    import librosa
    ref_wav, ref_sr = librosa.load(str(ref_audio_path), sr=None, mono=True)
    ref_audio_tuple = (ref_wav.astype(np.float32), int(ref_sr))
    
    # Default config
    language = DEFAULT_LANGUAGE
    use_xvector_only = False
    model_size = DEFAULT_MODEL_SIZE
    
    # Load model
    model = get_model(model_size)
    
    try:
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "config":
                    # Update config
                    language = data.get("language", DEFAULT_LANGUAGE)
                    use_xvector_only = data.get("use_xvector_only", False)
                    model_size = data.get("model_size", DEFAULT_MODEL_SIZE)
                    
                    # Reload model if size changed
                    global _model
                    if _model is None or model_size != DEFAULT_MODEL_SIZE:
                        _model = None  # Force reload
                        model = get_model(model_size)
                    
                    await websocket.send_json({"type": "config_ack"})
                
                elif msg_type == "text":
                    # Generate audio
                    target_text = data.get("text", "").strip()
                    ref_text = data.get("ref_text", "").strip() or None
                    
                    if not target_text:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Text is required"
                        })
                        continue
                    
                    # Check if reference text is needed
                    if not use_xvector_only and not ref_text:
                        await websocket.send_json({
                            "type": "error",
                            "message": "Reference text is required when use_xvector_only is False"
                        })
                        continue
                    
                    # Generate with timing
                    start_time = time.time()
                    wavs, sr = model.generate_voice_clone(
                        text=target_text,
                        language=language,
                        ref_audio=ref_audio_tuple,
                        ref_text=ref_text,
                        x_vector_only_mode=use_xvector_only,
                        max_new_tokens=2048,
                    )
                    generation_time = time.time() - start_time
                    audio_duration = len(wavs[0]) / sr
                    
                    # Convert to WAV bytes
                    wav_bytes_data = wav_bytes(wavs[0], sr)
                    audio_b64 = base64.b64encode(wav_bytes_data).decode("utf-8")
                    
                    # Send audio with timing info
                    await websocket.send_json({
                        "type": "audio",
                        "audio": audio_b64,
                        "sample_rate": sr,
                        "generation_time": round(generation_time, 3),
                        "audio_duration": round(audio_duration, 3),
                        "real_time_factor": round(audio_duration / generation_time, 3),
                    })
                
                elif msg_type == "end":
                    await websocket.send_json({"type": "end_ack"})
                    break
                
                else:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}"
                    })
            
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
    
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/v1/text-to-speech/{voice_id}/stream-input")
async def websocket_elevenlabs_compatible(websocket: WebSocket, voice_id: str):
    """
    ElevenLabs-compatible streaming endpoint.
    
    URL format: ws://localhost:8000/v1/text-to-speech/{voice_id}/stream-input
    
    Client Sends:
        Initialization:
        {
            "text": " ",
            "generation_config": {
                "language": "Auto",
                "use_xvector_only": false
            }
        }
        
        Stream text:
        {"text": "Hello world."}
        
        Close:
        {"text": ""}
    
    Server Sends:
        {
            "audio": "<base64_pcm_wav>",
            "isFinal": false
        }
        
        Final:
        {"isFinal": true}
    """
    await websocket.accept()
    
    # Load reference audio from voice directory
    ref_audio_path = get_voice_audio_path(voice_id)
    if ref_audio_path is None:
        await websocket.send_json({
            "error": f"Voice '{voice_id}' not found. Expected audio file in voices/{voice_id}/"
        })
        await websocket.close()
        return
    
    # Load reference audio
    import librosa
    ref_wav, ref_sr = librosa.load(str(ref_audio_path), sr=None, mono=True)
    ref_audio_tuple = (ref_wav.astype(np.float32), int(ref_sr))
    
    # Default config
    language = DEFAULT_LANGUAGE
    use_xvector_only = False
    
    # Load model
    model = get_model()
    
    initialized = False
    
    try:
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON format"})
                continue
            
            text = data.get("text", "")
            
            # Initialization message
            if not initialized and "generation_config" in data:
                gen_cfg = data["generation_config"]
                language = gen_cfg.get("language", DEFAULT_LANGUAGE)
                use_xvector_only = gen_cfg.get("use_xvector_only", False)
                initialized = True
                continue
            
            # Close signal
            if text == "":
                await websocket.send_json({"isFinal": True})
                break
            
            # Generate audio
            start_time = time.time()
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_tuple,
                ref_text=None,  # Use x-vector mode for streaming
                x_vector_only_mode=True,  # Always use x-vector for streaming
                max_new_tokens=2048,
            )
            generation_time = time.time() - start_time
            
            # Convert to WAV bytes
            wav_bytes_data = wav_bytes(wavs[0], sr)
            audio_b64 = base64.b64encode(wav_bytes_data).decode("utf-8")
            
            # Send audio
            await websocket.send_json({
                "audio": audio_b64,
                "isFinal": False,
                "generation_time": round(generation_time, 3),
            })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="Qwen3-TTS Voice Cloning Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-size", type=str, default=DEFAULT_MODEL_SIZE, 
                       choices=["0.6B", "1.7B"], help="Model size")
    args = parser.parse_args()
    
    # Set default model size
    DEFAULT_MODEL_SIZE = args.model_size
    
    print(f"Starting Qwen3-TTS server on http://{args.host}:{args.port}")
    print(f"Model size: {args.model_size}")
    print(f"Voices directory: {VOICES_DIR.absolute()}")
    print(f"\nWebSocket endpoints:")
    print(f"  - ws://{args.host}:{args.port}/ws/voice-clone/{{voice_id}}")
    print(f"  - ws://{args.host}:{args.port}/v1/text-to-speech/{{voice_id}}/stream-input")
    print(f"\nHTTP endpoint:")
    print(f"  - POST http://{args.host}:{args.port}/voice-clone/{{voice_id}}")
    
    uvicorn.run(app, host=args.host, port=args.port)
