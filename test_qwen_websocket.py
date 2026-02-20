"""
Test script for Qwen3-TTS WebSocket voice cloning endpoint.

Features:
1. WebSocket connection testing
2. Streaming support verification
3. Time analysis for audio generation
4. Audio similarity testing (MFCC-based)

Usage:
    python test_qwen_websocket.py --voice reference --port 8000

Note: The --voice parameter expects the voice directory name (voice_id),
not the full filename. The audio file should be in voices/{voice_id}/.
"""

import argparse
import asyncio
import base64
import json
import time
import wave
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import websockets
import librosa
from scipy.spatial.distance import cosine


# Supported audio file extensions
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}


def find_voice_audio(voice_id: str) -> Optional[Path]:
    """
    Find the audio file for a given voice_id.
    
    Args:
        voice_id: Voice directory name
    
    Returns:
        Path to the audio file, or None if not found
    """
    voices_dir = Path(__file__).parent / "voices"
    voice_dir = voices_dir / voice_id
    
    if not voice_dir.exists() or not voice_dir.is_dir():
        return None
    
    # Search for audio files in the voice directory
    for ext in AUDIO_EXTENSIONS:
        for audio_file in voice_dir.glob(f"*{ext}"):
            if audio_file.is_file():
                return audio_file
    
    return None


class AudioAnalyzer:
    """Audio analysis utilities for similarity testing."""
    
    @staticmethod
    def load_audio(file_path: str, sr: int = 24000) -> Tuple[np.ndarray, int]:
        """Load audio file."""
        audio, orig_sr = librosa.load(file_path, sr=sr, mono=True)
        return audio, orig_sr
    
    @staticmethod
    def extract_mfcc(audio: np.ndarray, sr: int = 24000, n_mfcc: int = 13) -> np.ndarray:
        """Extract MFCC features from audio."""
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfcc
    
    @staticmethod
    def compute_similarity(audio1: np.ndarray, audio2: np.ndarray, sr: int = 24000) -> float:
        """
        Compute similarity between two audio clips using MFCC features.
        Returns a score between 0 (dissimilar) and 1 (similar).
        """
        # Extract MFCC features
        mfcc1 = AudioAnalyzer.extract_mfcc(audio1, sr)
        mfcc2 = AudioAnalyzer.extract_mfcc(audio2, sr)
        
        # Compute cosine similarity for each MFCC coefficient
        similarities = []
        for i in range(mfcc1.shape[0]):
            sim = 1 - cosine(mfcc1[i], mfcc2[i])
            similarities.append(sim)
        
        # Average similarity across all coefficients
        avg_similarity = np.mean(similarities)
        return float(avg_similarity)
    
    @staticmethod
    def compute_spectral_similarity(audio1: np.ndarray, audio2: np.ndarray, sr: int = 24000) -> float:
        """Compute spectral similarity using chroma features."""
        chroma1 = librosa.feature.chroma_stft(y=audio1, sr=sr)
        chroma2 = librosa.feature.chroma_stft(y=audio2, sr=sr)
        
        # Compute cosine similarity
        similarities = []
        for i in range(chroma1.shape[0]):
            sim = 1 - cosine(chroma1[i], chroma2[i])
            similarities.append(sim)
        
        return float(np.mean(similarities))


class TimingStats:
    """Track timing statistics for audio generation."""
    
    def __init__(self):
        self.generation_times: List[float] = []
        self.audio_durations: List[float] = []
        self.real_time_factors: List[float] = []
    
    def add(self, generation_time: float, audio_duration: float):
        """Add a timing measurement."""
        self.generation_times.append(generation_time)
        self.audio_durations.append(audio_duration)
        rtf = audio_duration / generation_time if generation_time > 0 else 0
        self.real_time_factors.append(rtf)
    
    def report(self) -> dict:
        """Generate timing report."""
        if not self.generation_times:
            return {}
        
        return {
            "num_generations": len(self.generation_times),
            "generation_time": {
                "mean": np.mean(self.generation_times),
                "std": np.std(self.generation_times),
                "min": np.min(self.generation_times),
                "max": np.max(self.generation_times),
                "total": np.sum(self.generation_times),
            },
            "audio_duration": {
                "mean": np.mean(self.audio_durations),
                "std": np.std(self.audio_durations),
                "min": np.min(self.audio_durations),
                "max": np.max(self.audio_durations),
                "total": np.sum(self.audio_durations),
            },
            "real_time_factor": {
                "mean": np.mean(self.real_time_factors),
                "std": np.std(self.real_time_factors),
                "min": np.min(self.real_time_factors),
                "max": np.max(self.real_time_factors),
            },
        }


def save_wav(audio_bytes: bytes, output_path: str):
    """Save audio bytes to WAV file."""
    with open(output_path, "wb") as f:
        f.write(audio_bytes)


async def test_websocket_basic(
    host: str = "localhost",
    port: int = 8000,
    voice_id: str = "reference",
    test_text: str = "Hello, this is a test of the voice cloning system.",
):
    """
    Test 1: Basic WebSocket connection and audio generation.
    """
    print("\n" + "="*60)
    print("TEST 1: Basic WebSocket Connection")
    print("="*60)
    
    uri = f"ws://{host}:{port}/ws/voice-clone/{voice_id}"
    print(f"Connecting to {uri}...")
    
    # Create test output directory
    test_dir = Path(__file__).parent / "test_output"
    test_dir.mkdir(exist_ok=True)
    
    try:
        async with websockets.connect(uri) as ws:
            print("✓ Connected successfully!")
            
            # Send config
            config_msg = {
                "type": "config",
                "language": "Auto",
                "use_xvector_only": False,
                "model_size": "1.7B",
            }
            await ws.send(json.dumps(config_msg))
            print("✓ Sent config message")
            
            # Wait for config ack
            response = await ws.recv()
            data = json.loads(response)
            if data.get("type") == "config_ack":
                print("✓ Config acknowledged")
            
            # Send text for generation
            print(f"\nGenerating audio for: '{test_text}'")
            text_msg = {
                "type": "text",
                "text": test_text,
                "ref_text": test_text,  # Using same text as reference for simplicity
            }
            await ws.send(json.dumps(text_msg))
            
            # Receive audio response
            response = await ws.recv()
            data = json.loads(response)
            
            if data.get("type") == "audio":
                audio_b64 = data["audio"]
                audio_bytes = base64.b64decode(audio_b64)
                sample_rate = data["sample_rate"]
                generation_time = data["generation_time"]
                audio_duration = data["audio_duration"]
                rtf = data["real_time_factor"]
                
                # Save audio
                output_file = test_dir / "test1_basic_output.wav"
                save_wav(audio_bytes, str(output_file))
                
                print(f"✓ Audio generated successfully!")
                print(f"  - Saved to: {output_file}")
                print(f"  - Sample rate: {sample_rate} Hz")
                print(f"  - Audio duration: {audio_duration:.3f}s")
                print(f"  - Generation time: {generation_time:.3f}s")
                print(f"  - Real-time factor: {rtf:.3f}x")
                
                return {
                    "success": True,
                    "audio_file": str(output_file),
                    "generation_time": generation_time,
                    "audio_duration": audio_duration,
                    "real_time_factor": rtf,
                }
            elif data.get("type") == "error":
                print(f"✗ Error: {data['message']}")
                return {"success": False, "error": data["message"]}
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"success": False, "error": str(e)}


async def test_websocket_streaming(
    host: str = "localhost",
    port: int = 8000,
    voice_id: str = "reference",
):
    """
    Test 2: Streaming support - multiple text chunks.
    """
    print("\n" + "="*60)
    print("TEST 2: Streaming Support (Multiple Chunks)")
    print("="*60)
    
    uri = f"ws://{host}:{port}/ws/voice-clone/{voice_id}"
    print(f"Connecting to {uri}...")
    
    test_dir = Path(__file__).parent / "test_output"
    test_dir.mkdir(exist_ok=True)
    
    text_chunks = [
        "Hello, this is the first chunk.",
        "This is the second chunk of text.",
        "And this is the third and final chunk.",
    ]
    
    try:
        async with websockets.connect(uri) as ws:
            print("✓ Connected successfully!")
            
            # Send config with x-vector only for streaming
            config_msg = {
                "type": "config",
                "language": "Auto",
                "use_xvector_only": True,  # Use x-vector for streaming
                "model_size": "1.7B",
            }
            await ws.send(json.dumps(config_msg))
            
            # Wait for config ack
            response = await ws.recv()
            data = json.loads(response)
            if data.get("type") == "config_ack":
                print("✓ Config acknowledged")
            
            # Send multiple text chunks
            timing_stats = TimingStats()
            chunk_files = []
            
            for i, chunk_text in enumerate(text_chunks):
                print(f"\n[{i+1}/{len(text_chunks)}] Generating: '{chunk_text}'")
                
                text_msg = {
                    "type": "text",
                    "text": chunk_text,
                }
                await ws.send(json.dumps(text_msg))
                
                # Receive audio response
                response = await ws.recv()
                data = json.loads(response)
                
                if data.get("type") == "audio":
                    audio_b64 = data["audio"]
                    audio_bytes = base64.b64decode(audio_b64)
                    generation_time = data["generation_time"]
                    audio_duration = data["audio_duration"]
                    rtf = data["real_time_factor"]
                    
                    # Save audio chunk
                    output_file = test_dir / f"test2_chunk_{i:02d}.wav"
                    save_wav(audio_bytes, str(output_file))
                    chunk_files.append(str(output_file))
                    
                    timing_stats.add(generation_time, audio_duration)
                    
                    print(f"  ✓ Generated in {generation_time:.3f}s (RTF: {rtf:.3f}x)")
                    print(f"  ✓ Saved to: {output_file}")
            
            # Send end signal
            await ws.send(json.dumps({"type": "end"}))
            response = await ws.recv()
            if json.loads(response).get("type") == "end_ack":
                print("\n✓ Stream completed successfully!")
            
            # Print timing statistics
            print("\n" + "-"*60)
            print("Timing Statistics:")
            print("-"*60)
            stats = timing_stats.report()
            print(f"  Number of chunks: {stats['num_generations']}")
            print(f"  Generation time - Mean: {stats['generation_time']['mean']:.3f}s, "
                  f"Std: {stats['generation_time']['std']:.3f}s")
            print(f"  Audio duration - Mean: {stats['audio_duration']['mean']:.3f}s, "
                  f"Std: {stats['audio_duration']['std']:.3f}s")
            print(f"  Real-time factor - Mean: {stats['real_time_factor']['mean']:.3f}x, "
                  f"Std: {stats['real_time_factor']['std']:.3f}")
            
            return {
                "success": True,
                "chunk_files": chunk_files,
                "timing_stats": stats,
            }
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"success": False, "error": str(e)}


async def test_audio_similarity(
    host: str = "localhost",
    port: int = 8000,
    voice_id: str = "reference",
    ref_audio_path: str = None,
):
    """
    Test 3: Audio similarity testing.
    Compare generated audio with reference audio.
    """
    print("\n" + "="*60)
    print("TEST 3: Audio Similarity Testing")
    print("="*60)
    
    if ref_audio_path is None:
        # Find the audio file in the voice directory
        audio_file = find_voice_audio(voice_id)
        if audio_file is None:
            print(f"✗ Voice '{voice_id}' not found in voices/ directory")
            return {"success": False, "error": f"Voice '{voice_id}' not found"}
        ref_audio_path = str(audio_file)
    
    if not Path(ref_audio_path).exists():
        print(f"✗ Reference audio not found: {ref_audio_path}")
        return {"success": False, "error": "Reference audio not found"}
    
    print(f"Reference audio: {ref_audio_path}")
    
    uri = f"ws://{host}:{port}/ws/voice-clone/{voice_id}"
    print(f"Connecting to {uri}...")
    
    test_dir = Path(__file__).parent / "test_output"
    test_dir.mkdir(exist_ok=True)
    
    # Test with similar text to reference
    test_text = "This is a test of voice similarity."
    
    try:
        async with websockets.connect(uri) as ws:
            print("✓ Connected successfully!")
            
            # Send config
            config_msg = {
                "type": "config",
                "language": "Auto",
                "use_xvector_only": False,
                "model_size": "1.7B",
            }
            await ws.send(json.dumps(config_msg))
            
            # Wait for config ack
            response = await ws.recv()
            data = json.loads(response)
            if data.get("type") == "config_ack":
                print("✓ Config acknowledged")
            
            # Generate audio
            print(f"\nGenerating audio for: '{test_text}'")
            text_msg = {
                "type": "text",
                "text": test_text,
                "ref_text": test_text,
            }
            await ws.send(json.dumps(text_msg))
            
            # Receive audio response
            response = await ws.recv()
            data = json.loads(response)
            
            if data.get("type") == "audio":
                audio_b64 = data["audio"]
                audio_bytes = base64.b64decode(audio_b64)
                
                # Save generated audio
                gen_audio_path = test_dir / "test3_generated.wav"
                save_wav(audio_bytes, str(gen_audio_path))
                print(f"✓ Generated audio saved to: {gen_audio_path}")
                
                # Load reference and generated audio
                print("\nAnalyzing audio similarity...")
                ref_audio, sr = AudioAnalyzer.load_audio(ref_audio_path)
                gen_audio, _ = AudioAnalyzer.load_audio(str(gen_audio_path), sr=sr)
                
                # Compute similarities
                mfcc_similarity = AudioAnalyzer.compute_similarity(ref_audio, gen_audio, sr)
                spectral_similarity = AudioAnalyzer.compute_spectral_similarity(ref_audio, gen_audio, sr)
                
                print(f"\nSimilarity Scores:")
                print(f"  MFCC similarity: {mfcc_similarity:.4f} (0-1 scale)")
                print(f"  Spectral similarity: {spectral_similarity:.4f} (0-1 scale)")
                print(f"  Average similarity: {(mfcc_similarity + spectral_similarity) / 2:.4f}")
                
                # Interpret results
                avg_sim = (mfcc_similarity + spectral_similarity) / 2
                if avg_sim > 0.7:
                    quality = "Excellent"
                elif avg_sim > 0.5:
                    quality = "Good"
                elif avg_sim > 0.3:
                    quality = "Fair"
                else:
                    quality = "Poor"
                
                print(f"\nVoice cloning quality: {quality}")
                
                return {
                    "success": True,
                    "mfcc_similarity": mfcc_similarity,
                    "spectral_similarity": spectral_similarity,
                    "average_similarity": avg_sim,
                    "quality": quality,
                    "generated_audio": str(gen_audio_path),
                }
            elif data.get("type") == "error":
                print(f"✗ Error: {data['message']}")
                return {"success": False, "error": data["message"]}
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"success": False, "error": str(e)}


async def test_elevenlabs_compatible(
    host: str = "localhost",
    port: int = 8000,
    voice_id: str = "reference",
):
    """
    Test 4: ElevenLabs-compatible endpoint.
    """
    print("\n" + "="*60)
    print("TEST 4: ElevenLabs-Compatible Endpoint")
    print("="*60)
    
    uri = f"ws://{host}:{port}/v1/text-to-speech/{voice_id}/stream-input"
    print(f"Connecting to {uri}...")
    
    test_dir = Path(__file__).parent / "test_output"
    test_dir.mkdir(exist_ok=True)
    
    try:
        async with websockets.connect(uri) as ws:
            print("✓ Connected successfully!")
            
            # Send initialization
            init_msg = {
                "text": " ",
                "generation_config": {
                    "language": "Auto",
                    "use_xvector_only": True,
                }
            }
            await ws.send(json.dumps(init_msg))
            print("✓ Sent initialization message")
            
            # Send text chunks
            text_chunks = [
                "Hello from the ElevenLabs-compatible endpoint.",
                "This is a second chunk of text.",
            ]
            
            chunk_files = []
            for i, chunk_text in enumerate(text_chunks):
                print(f"\n[{i+1}/{len(text_chunks)}] Sending: '{chunk_text}'")
                
                await ws.send(json.dumps({"text": chunk_text}))
                
                # Receive audio
                response = await ws.recv()
                data = json.loads(response)
                
                if data.get("isFinal") is True:
                    print("✓ Received final signal")
                    break
                
                if "audio" in data:
                    audio_b64 = data["audio"]
                    audio_bytes = base64.b64decode(audio_b64)
                    
                    output_file = test_dir / f"test4_el_chunk_{i:02d}.wav"
                    save_wav(audio_bytes, str(output_file))
                    chunk_files.append(str(output_file))
                    
                    gen_time = data.get("generation_time", 0)
                    print(f"  ✓ Generated in {gen_time:.3f}s")
                    print(f"  ✓ Saved to: {output_file}")
            
            # Send close signal
            await ws.send(json.dumps({"text": ""}))
            
            # Wait for final confirmation
            response = await ws.recv()
            data = json.loads(response)
            if data.get("isFinal") is True:
                print("\n✓ ElevenLabs-compatible test completed successfully!")
            
            return {
                "success": True,
                "chunk_files": chunk_files,
            }
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return {"success": False, "error": str(e)}


async def run_all_tests(
    host: str = "localhost",
    port: int = 8000,
    voice_id: str = "reference",
    ref_audio_path: str = None,
):
    """Run all tests and generate comprehensive report."""
    print("\n" + "="*60)
    print("QWEN3-TTS WEBSOCKET TEST SUITE")
    print("="*60)
    print(f"Host: {host}:{port}")
    print(f"Voice ID: {voice_id}")
    print(f"Reference audio: {ref_audio_path or voice_id}")
    
    results = {}
    
    # Test 1: Basic connection
    results["test1_basic"] = await test_websocket_basic(host, port, voice_id)
    
    # Test 2: Streaming
    results["test2_streaming"] = await test_websocket_streaming(host, port, voice_id)
    
    # Test 3: Audio similarity
    results["test3_similarity"] = await test_audio_similarity(
        host, port, voice_id, ref_audio_path
    )
    
    # Test 4: ElevenLabs-compatible
    results["test4_elevenlabs"] = await test_elevenlabs_compatible(host, port, voice_id)
    
    # Generate summary report
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result.get("success") else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not result.get("success"):
            print(f"  Error: {result.get('error', 'Unknown error')}")
    
    # Save results to JSON
    test_dir = Path(__file__).parent / "test_output"
    results_file = test_dir / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test Qwen3-TTS WebSocket voice cloning endpoint"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server hostname (default: localhost)",
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    
    parser.add_argument(
        "--voice",
        type=str,
        required=True,
        help="Voice ID (directory name, e.g., reference)",
    )
    
    parser.add_argument(
        "--ref-audio",
        type=str,
        default=None,
        help="Path to reference audio for similarity testing (default: voices/{voice})",
    )
    
    parser.add_argument(
        "--test",
        type=str,
        choices=["all", "basic", "streaming", "similarity", "elevenlabs"],
        default="all",
        help="Which test to run (default: all)",
    )
    
    args = parser.parse_args()
    
    # Run tests
    if args.test == "all":
        asyncio.run(run_all_tests(
            host=args.host,
            port=args.port,
            voice_id=args.voice,
            ref_audio_path=args.ref_audio,
        ))
    elif args.test == "basic":
        asyncio.run(test_websocket_basic(args.host, args.port, args.voice))
    elif args.test == "streaming":
        asyncio.run(test_websocket_streaming(args.host, args.port, args.voice))
    elif args.test == "similarity":
        asyncio.run(test_audio_similarity(
            args.host, args.port, args.voice, args.ref_audio
        ))
    elif args.test == "elevenlabs":
        asyncio.run(test_elevenlabs_compatible(args.host, args.port, args.voice))


if __name__ == "__main__":
    main()
