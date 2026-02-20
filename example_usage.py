"""
Example usage of Qwen3-TTS WebSocket server.

This script demonstrates various ways to use the voice cloning API.
"""

import asyncio
import base64
import json
import wave
from pathlib import Path

import websockets


async def example_basic_voice_clone():
    """Example 1: Basic voice cloning."""
    print("\n" + "="*60)
    print("Example 1: Basic Voice Cloning")
    print("="*60)
    
    uri = "ws://localhost:8000/ws/voice-clone/reference.wav"
    
    async with websockets.connect(uri) as ws:
        # Configure
        await ws.send(json.dumps({
            "type": "config",
            "language": "Auto",
            "use_xvector_only": True,
            "model_size": "1.7B",
        }))
        await ws.recv()  # Wait for ack
        
        # Generate
        await ws.send(json.dumps({
            "type": "text",
            "text": "This is an example of voice cloning with Qwen3-TTS.",
        }))
        
        # Receive audio
        response = await ws.recv()
        data = json.loads(response)
        
        if data["type"] == "audio":
            audio_bytes = base64.b64decode(data["audio"])
            with open("example1_output.wav", "wb") as f:
                f.write(audio_bytes)
            print(f"✓ Saved to example1_output.wav")
            print(f"  Generation time: {data['generation_time']:.3f}s")
            print(f"  Real-time factor: {data['real_time_factor']:.3f}x")
        
        await ws.send(json.dumps({"type": "end"}))


async def example_streaming_chunks():
    """Example 2: Streaming multiple text chunks."""
    print("\n" + "="*60)
    print("Example 2: Streaming Multiple Chunks")
    print("="*60)
    
    uri = "ws://localhost:8000/ws/voice-clone/reference.wav"
    
    chunks = [
        "Welcome to this demonstration.",
        "We are testing streaming capabilities.",
        "Each chunk is generated independently.",
    ]
    
    async with websockets.connect(uri) as ws:
        # Configure for streaming
        await ws.send(json.dumps({
            "type": "config",
            "language": "Auto",
            "use_xvector_only": True,  # Use x-vector for streaming
            "model_size": "1.7B",
        }))
        await ws.recv()
        
        # Generate each chunk
        for i, chunk in enumerate(chunks):
            print(f"\n[{i+1}/{len(chunks)}] Generating: '{chunk}'")
            
            await ws.send(json.dumps({
                "type": "text",
                "text": chunk,
            }))
            
            response = await ws.recv()
            data = json.loads(response)
            
            if data["type"] == "audio":
                audio_bytes = base64.b64decode(data["audio"])
                with open(f"example2_chunk_{i}.wav", "wb") as f:
                    f.write(audio_bytes)
                print(f"  ✓ Saved to example2_chunk_{i}.wav")
                print(f"  Generation time: {data['generation_time']:.3f}s")
        
        await ws.send(json.dumps({"type": "end"}))
        print("\n✓ All chunks generated!")


async def example_icl_mode():
    """Example 3: ICL mode with reference text (higher quality)."""
    print("\n" + "="*60)
    print("Example 3: ICL Mode (Higher Quality)")
    print("="*60)
    
    uri = "ws://localhost:8000/ws/voice-clone/reference.wav"
    
    # Reference text should match the reference audio content
    reference_text = "This is the reference audio content."
    target_text = "This is new text spoken in the cloned voice."
    
    async with websockets.connect(uri) as ws:
        # Configure for ICL mode
        await ws.send(json.dumps({
            "type": "config",
            "language": "Auto",
            "use_xvector_only": False,  # ICL mode
            "model_size": "1.7B",
        }))
        await ws.recv()
        
        # Generate with reference text
        await ws.send(json.dumps({
            "type": "text",
            "text": target_text,
            "ref_text": reference_text,  # Required for ICL mode
        }))
        
        response = await ws.recv()
        data = json.loads(response)
        
        if data["type"] == "audio":
            audio_bytes = base64.b64decode(data["audio"])
            with open("example3_icl_output.wav", "wb") as f:
                f.write(audio_bytes)
            print(f"✓ Saved to example3_icl_output.wav")
            print(f"  Generation time: {data['generation_time']:.3f}s")
            print(f"  Note: ICL mode provides higher quality but requires reference text")
        
        await ws.send(json.dumps({"type": "end"}))


async def example_elevenlabs_compatible():
    """Example 4: ElevenLabs-compatible endpoint."""
    print("\n" + "="*60)
    print("Example 4: ElevenLabs-Compatible Endpoint")
    print("="*60)
    
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
        texts = [
            "Hello from the ElevenLabs-compatible endpoint.",
            "This follows the ElevenLabs API format.",
        ]
        
        for i, text in enumerate(texts):
            print(f"\n[{i+1}/{len(texts)}] Sending: '{text}'")
            await ws.send(json.dumps({"text": text}))
            
            response = await ws.recv()
            data = json.loads(response)
            
            if data.get("isFinal"):
                break
            
            if "audio" in data:
                audio_bytes = base64.b64decode(data["audio"])
                with open(f"example4_el_{i}.wav", "wb") as f:
                    f.write(audio_bytes)
                print(f"  ✓ Saved to example4_el_{i}.wav")
        
        # Close
        await ws.send(json.dumps({"text": ""}))
        print("\n✓ ElevenLabs-compatible test completed!")


async def example_http_endpoint():
    """Example 5: Using HTTP endpoint."""
    print("\n" + "="*60)
    print("Example 5: HTTP Endpoint")
    print("="*60)
    
    import requests
    
    url = "http://localhost:8000/voice-clone/reference.wav"
    
    payload = {
        "target_text": "This is generated via HTTP endpoint.",
        "language": "Auto",
        "use_xvector_only": True,
        "model_size": "1.7B",
    }
    
    print("Sending request...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        # Save audio
        with open("example5_http_output.wav", "wb") as f:
            f.write(response.content)
        
        # Get timing info from headers
        gen_time = float(response.headers.get("X-Generation-Time", 0))
        audio_duration = float(response.headers.get("X-Audio-Duration", 0))
        sample_rate = response.headers.get("X-Sample-Rate", "unknown")
        
        print(f"✓ Saved to example5_http_output.wav")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Audio duration: {audio_duration:.3f}s")
        print(f"  Generation time: {gen_time:.3f}s")
        print(f"  Real-time factor: {audio_duration/gen_time:.3f}x")
    else:
        print(f"✗ Error: {response.status_code}")
        print(response.text)


async def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("Qwen3-TTS WebSocket Examples")
    print("="*60)
    print("\nMake sure the server is running:")
    print("  uv run qwen_tts_server.py")
    print("\nAnd you have a reference audio file:")
    print("  voices/reference.wav")
    
    input("\nPress Enter to run examples...")
    
    try:
        # Run examples
        await example_basic_voice_clone()
        await example_streaming_chunks()
        await example_icl_mode()
        await example_elevenlabs_compatible()
        await example_http_endpoint()
        
        print("\n" + "="*60)
        print("All examples completed!")
        print("="*60)
        print("\nGenerated files:")
        print("  - example1_output.wav")
        print("  - example2_chunk_0.wav, example2_chunk_1.wav, example2_chunk_2.wav")
        print("  - example3_icl_output.wav")
        print("  - example4_el_0.wav, example4_el_1.wav")
        print("  - example5_http_output.wav")
        
    except ConnectionRefusedError:
        print("\n✗ Could not connect to server.")
        print("  Make sure the server is running:")
        print("    python qwen_tts_server.py")
    except Exception as e:
        print(f"\n✗ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
