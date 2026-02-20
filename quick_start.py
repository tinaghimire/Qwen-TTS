"""
Quick start script for Qwen3-TTS WebSocket testing.

This script provides a simple example to get started with voice cloning.
"""

import asyncio
import base64
import json
import wave
from pathlib import Path

import websockets


async def quick_test():
    """Quick test of the Qwen3-TTS WebSocket server."""
    
    # Configuration
    HOST = "localhost"
    PORT = 8000
    VOICE_ID = "reference.wav"  # Make sure this file exists in voices/ directory
    TEST_TEXT = "Hello, this is a quick test of the voice cloning system."
    
    print("="*60)
    print("Qwen3-TTS Quick Start Test")
    print("="*60)
    print(f"Server: ws://{HOST}:{PORT}")
    print(f"Voice ID: {VOICE_ID}")
    print(f"Test text: '{TEST_TEXT}'")
    print()
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    uri = f"ws://{HOST}:{PORT}/ws/voice-clone/{VOICE_ID}"
    
    try:
        async with websockets.connect(uri) as ws:
            print("✓ Connected to server!")
            
            # Send configuration
            config = {
                "type": "config",
                "language": "Auto",
                "use_xvector_only": True,  # Use x-vector mode for simplicity
                "model_size": "1.7B",
            }
            await ws.send(json.dumps(config))
            print("✓ Sent configuration")
            
            # Wait for acknowledgment
            response = await ws.recv()
            data = json.loads(response)
            if data.get("type") == "config_ack":
                print("✓ Configuration acknowledged")
            
            # Send text for generation
            print(f"\nGenerating audio...")
            text_msg = {
                "type": "text",
                "text": TEST_TEXT,
            }
            await ws.send(json.dumps(text_msg))
            
            # Receive audio
            response = await ws.recv()
            data = json.loads(response)
            
            if data.get("type") == "audio":
                # Decode and save audio
                audio_b64 = data["audio"]
                audio_bytes = base64.b64decode(audio_b64)
                
                output_file = output_dir / "quick_start_output.wav"
                with open(output_file, "wb") as f:
                    f.write(audio_bytes)
                
                # Print results
                print("✓ Audio generated successfully!")
                print(f"\nResults:")
                print(f"  - Output file: {output_file}")
                print(f"  - Sample rate: {data['sample_rate']} Hz")
                print(f"  - Audio duration: {data['audio_duration']:.3f}s")
                print(f"  - Generation time: {data['generation_time']:.3f}s")
                print(f"  - Real-time factor: {data['real_time_factor']:.3f}x")
                print(f"\n✓ Test completed successfully!")
                print(f"  Play the audio: {output_file}")
                
            elif data.get("type") == "error":
                print(f"✗ Error: {data['message']}")
            
            # Send end signal
            await ws.send(json.dumps({"type": "end"}))
    
    except ConnectionRefusedError:
        print("✗ Could not connect to server.")
        print("  Make sure the server is running:")
        print("    python qwen_tts_server.py")
    except Exception as e:
        print(f"✗ Error: {e}")


if __name__ == "__main__":
    print("\nBefore running this script:")
    print("1. Start the server: uv run qwen_tts_server.py")
    print("2. Place a reference audio file in voices/reference.wav")
    print()
    
    input("Press Enter to start the test...")
    
    asyncio.run(quick_test())
