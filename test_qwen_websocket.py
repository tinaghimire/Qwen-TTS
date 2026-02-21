"""
Test script for Qwen3-TTS WebSocket voice cloning endpoint.

All configuration is read from .env file.
"""

import asyncio
import base64
import json
import os
import wave
from pathlib import Path
from typing import List
from dotenv import load_dotenv

import numpy as np
import websockets
import librosa
from scipy.spatial.distance import cosine

# Load environment variables
load_dotenv()

# Supported audio file extensions
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}

# Configuration from .env
HOST = os.getenv("HOST", "localhost")
PORT = int(os.getenv("PORT", 8000))
VOICE_ID = os.getenv("TEST_VOICE_ID", "english_voice")
TEST_OUTPUT_DIR = os.getenv("TEST_OUTPUT_DIR", "test_output")
TEST_TYPE = os.getenv("TEST_TYPE", "all")


def find_voice_audio(voice_id: str):
    """Find the audio file for a given voice_id."""
    voices_dir = Path(__file__).parent / "voices"
    voice_dir = voices_dir / voice_id

    if not voice_dir.exists() or not voice_dir.is_dir():
        return None

    for ext in AUDIO_EXTENSIONS:
        for audio_file in voice_dir.glob(f"*{ext}"):
            if audio_file.is_file():
                return audio_file

    return None


class AudioAnalyzer:
    """Audio analysis utilities for similarity testing."""

    @staticmethod
    def load_audio(file_path: str, sr: int = 24000):
        audio, orig_sr = librosa.load(file_path, sr=sr, mono=True)
        return audio, orig_sr

    @staticmethod
    def extract_mfcc(audio: np.ndarray, sr: int = 24000, n_mfcc: int = 13):
        return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    @staticmethod
    def compute_similarity(audio1: np.ndarray, audio2: np.ndarray, sr: int = 24000):
        from scipy.spatial.distance import cosine
        mfcc1 = AudioAnalyzer.extract_mfcc(audio1, sr)
        mfcc2 = AudioAnalyzer.extract_mfcc(audio2, sr)

        min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
        if min_frames == 0:
            return 0.0

        mfcc1_trimmed = mfcc1[:, :min_frames]
        mfcc2_trimmed = mfcc2[:, :min_frames]

        similarities = []
        for i in range(mfcc1_trimmed.shape[0]):
            sim = 1 - cosine(mfcc1_trimmed[i], mfcc2_trimmed[i])
            similarities.append(sim)

        return float(np.mean(similarities))

    @staticmethod
    def compute_spectral_similarity(audio1: np.ndarray, audio2: np.ndarray, sr: int = 24000):
        from scipy.spatial.distance import cosine
        chroma1 = librosa.feature.chroma_stft(y=audio1, sr=sr)
        chroma2 = librosa.feature.chroma_stft(y=audio2, sr=sr)

        min_frames = min(chroma1.shape[1], chroma2.shape[1])
        if min_frames == 0:
            return 0.0

        chroma1_trimmed = chroma1[:, :min_frames]
        chroma2_trimmed = chroma2[:, :min_frames]

        similarities = []
        for i in range(chroma1_trimmed.shape[0]):
            sim = 1 - cosine(chroma1_trimmed[i], chroma2_trimmed[i])
            similarities.append(sim)

        return float(np.mean(similarities))


class TimingStats:
    """Track timing statistics for audio generation."""

    def __init__(self):
        self.generation_times: List[float] = []
        self.audio_durations: List[float] = []
        self.real_time_factors: List[float] = []

    def add(self, generation_time: float, audio_duration: float):
        self.generation_times.append(generation_time)
        self.audio_durations.append(audio_duration)
        rtf = audio_duration / generation_time if generation_time > 0 else 0
        self.real_time_factors.append(rtf)

    def report(self):
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
    with open(output_path, "wb") as f:
        f.write(audio_bytes)


async def test_websocket_basic():
    uri = f"ws://{HOST}:{PORT}/ws/voice-clone/{VOICE_ID}"
    print("\n" + "="*60)
    print("TEST 1: Basic WebSocket Connection")
    print("="*60)
    print(f"Connecting to {uri}...")

    test_dir = Path(__file__).parent / TEST_OUTPUT_DIR
    test_dir.mkdir(exist_ok=True)

    test_text = "Hello, this is a test of the voice cloning system."

    try:
        async with websockets.connect(uri) as ws:
            print("✓ Connected successfully!")

            await ws.send(json.dumps({
                "type": "config",
                "language": "Auto",
                "use_xvector_only": False,
                "model_size": "1.7B",
            }))
            print("✓ Sent config message")

            response = await ws.recv()
            data = json.loads(response)
            if data.get("type") == "config_ack":
                print("✓ Config acknowledged")

            print(f"\nGenerating audio for: '{test_text}'")
            await ws.send(json.dumps({
                "type": "text",
                "text": test_text,
                "ref_text": test_text,
            }))

            response = await ws.recv()
            data = json.loads(response)

            if data.get("type") == "audio":
                audio_b64 = data["audio"]
                audio_bytes = base64.b64decode(audio_b64)
                generation_time = data["generation_time"]
                audio_duration = data["audio_duration"]
                rtf = data["real_time_factor"]

                output_file = test_dir / "test1_basic_output.wav"
                save_wav(audio_bytes, str(output_file))

                print(f"✓ Audio generated successfully!")
                print(f"  - Saved to: {output_file}")
                print(f"  - Sample rate: {data['sample_rate']} Hz")
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


async def test_websocket_streaming():
    uri = f"ws://{HOST}:{PORT}/ws/voice-clone/{VOICE_ID}"
    print("\n" + "="*60)
    print("TEST 2: Streaming Support (Multiple Chunks)")
    print("="*60)
    print(f"Connecting to {uri}...")

    test_dir = Path(__file__).parent / TEST_OUTPUT_DIR
    test_dir.mkdir(exist_ok=True)

    text_chunks = [
        "Hello, this is the first chunk.",
        "This is the second chunk of text.",
        "And this is the third and final chunk.",
    ]

    try:
        async with websockets.connect(uri) as ws:
            print("✓ Connected successfully!")

            await ws.send(json.dumps({
                "type": "config",
                "language": "Auto",
                "use_xvector_only": True,
                "model_size": "1.7B",
            }))

            response = await ws.recv()
            data = json.loads(response)
            if data.get("type") == "config_ack":
                print("✓ Config acknowledged")

            timing_stats = TimingStats()
            chunk_files = []

            for i, chunk_text in enumerate(text_chunks):
                print(f"\n[{i+1}/{len(text_chunks)}] Generating: '{chunk_text}'")

                await ws.send(json.dumps({
                    "type": "text",
                    "text": chunk_text,
                }))

                response = await ws.recv()
                data = json.loads(response)

                if data.get("type") == "audio":
                    audio_b64 = data["audio"]
                    audio_bytes = base64.b64decode(audio_b64)
                    generation_time = data["generation_time"]
                    audio_duration = data["audio_duration"]
                    rtf = data["real_time_factor"]

                    output_file = test_dir / f"test2_chunk_{i:02d}.wav"
                    save_wav(audio_bytes, str(output_file))
                    chunk_files.append(str(output_file))

                    timing_stats.add(generation_time, audio_duration)

                    print(f"  ✓ Generated in {generation_time:.3f}s (RTF: {rtf:.3f}x)")
                    print(f"  ✓ Saved to: {output_file}")

            await ws.send(json.dumps({"type": "end"}))
            response = await ws.recv()
            if json.loads(response).get("type") == "end_ack":
                print("\n✓ Stream completed successfully!")

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


async def test_audio_similarity():
    uri = f"ws://{HOST}:{PORT}/ws/voice-clone/{VOICE_ID}"
    print("\n" + "="*60)
    print("TEST 3: Audio Similarity Testing")
    print("="*60)

    ref_audio_path = find_voice_audio(VOICE_ID)
    if ref_audio_path is None:
        print(f"✗ Voice '{VOICE_ID}' not found in voices/ directory")
        return {"success": False, "error": f"Voice '{VOICE_ID}' not found"}

    if not Path(ref_audio_path).exists():
        print(f"✗ Reference audio not found: {ref_audio_path}")
        return {"success": False, "error": "Reference audio not found"}

    print(f"Reference audio: {ref_audio_path}")
    print(f"Connecting to {uri}...")

    test_dir = Path(__file__).parent / TEST_OUTPUT_DIR
    test_dir.mkdir(exist_ok=True)

    test_text = "This is a test of voice similarity."

    try:
        async with websockets.connect(uri) as ws:
            print("✓ Connected successfully!")

            await ws.send(json.dumps({
                "type": "config",
                "language": "Auto",
                "use_xvector_only": False,
                "model_size": "1.7B",
            }))

            response = await ws.recv()
            data = json.loads(response)
            if data.get("type") == "config_ack":
                print("✓ Config acknowledged")

            print(f"\nGenerating audio for: '{test_text}'")
            await ws.send(json.dumps({
                "type": "text",
                "text": test_text,
                "ref_text": test_text,
            }))

            response = await ws.recv()
            data = json.loads(response)

            if data.get("type") == "audio":
                audio_b64 = data["audio"]
                audio_bytes = base64.b64decode(audio_b64)

                gen_audio_path = test_dir / "test3_generated.wav"
                save_wav(audio_bytes, str(gen_audio_path))
                print(f"✓ Generated audio saved to: {gen_audio_path}")

                print("\nAnalyzing audio similarity...")
                ref_audio, sr = AudioAnalyzer.load_audio(str(ref_audio_path))
                gen_audio, _ = AudioAnalyzer.load_audio(str(gen_audio_path), sr=sr)

                mfcc_similarity = AudioAnalyzer.compute_similarity(ref_audio, gen_audio, sr)
                spectral_similarity = AudioAnalyzer.compute_spectral_similarity(ref_audio, gen_audio, sr)

                print(f"\nSimilarity Scores:")
                print(f"  MFCC similarity: {mfcc_similarity:.4f} (0-1 scale)")
                print(f"  Spectral similarity: {spectral_similarity:.4f} (0-1 scale)")
                print(f"  Average similarity: {(mfcc_similarity + spectral_similarity) / 2:.4f}")

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


async def run_all_tests():
    print("\n" + "="*60)
    print("QWEN3-TTS WEBSOCKET TEST SUITE")
    print("="*60)
    print(f"Host: {HOST}:{PORT}")
    print(f"Voice ID: {VOICE_ID}")

    results = {}

    if TEST_TYPE in ["all", "basic"]:
        results["test1_basic"] = await test_websocket_basic()

    if TEST_TYPE in ["all", "streaming"]:
        results["test2_streaming"] = await test_websocket_streaming()

    if TEST_TYPE in ["all", "similarity"]:
        results["test3_similarity"] = await test_audio_similarity()

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, result in results.items():
        status = "✓ PASSED" if result.get("success") else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not result.get("success"):
            print(f"  Error: {result.get('error', 'Unknown error')}")

    test_dir = Path(__file__).parent / TEST_OUTPUT_DIR
    results_file = test_dir / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")

    return results


def main():
    asyncio.run(run_all_tests())


if __name__ == "__main__":
    main()