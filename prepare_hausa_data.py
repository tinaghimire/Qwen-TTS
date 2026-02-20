#!/usr/bin/env python3
# coding=utf-8
"""
Data preparation script for Hausa TTS dataset.
Loads data from Hugging Face, processes audio codes, and saves to JSONL format
for use with sft_12hz.py training script.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS", "finetuning"))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


def prepare_hausa_data(
    split: str = "train",
    output_jsonl: str = None,
    model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    ref_audio_path: str = None,
    ref_text: str = None,
    max_samples: int = None,
    device: str = "cuda"
):
    """
    Prepare Hausa TTS data for training.
    
    Args:
        split: Dataset split (train, validation, test)
        output_jsonl: Output JSONL file path
        model_path: Path to Qwen3-TTS model
        ref_audio_path: Path to reference audio
        ref_text: Reference text
        max_samples: Maximum number of samples to process
        device: Device to use for processing
    """
    
    # Set default reference audio path
    if ref_audio_path is None:
        ref_audio_path = os.path.join(
            os.path.dirname(__file__),
            "voices", "english_voice", "english_voice.wav"
        )
    
    # Set default reference text
    if ref_text is None:
        ref_text = "MTN Entertainment and Lifestyle. Entertainment and Lifestyle are at the heart of MTN's offering. We bring you music, movies, games and more through our digital platforms. With MTN musicals, you can stream your favorite"
    
    print(f"Loading Qwen3-TTS model from {model_path}...")
    try:
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print(f"✓ Model loaded with flash_attention_2")
    except (ImportError, Exception) as e:
        print(f"⚠ Flash attention not available, falling back to SDPA: {e}")
        model = Qwen3TTSModel.from_pretrained(
            model_path,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print(f"✓ Model loaded with SDPA")
    
    print(f"Loading Hausa TTS dataset ({split} split)...")
    hf_dataset = load_dataset("vaghawan/hausa-tts-22k", split=split)
    
    # Limit samples if specified
    if max_samples is not None:
        hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))
    
    print(f"Processing {len(hf_dataset)} samples...")
    
    # Prepare output directory
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    
    # Process each sample
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(hf_dataset, desc=f"Processing {split}")):
            try:
                # Extract audio data
                audio_data = item["audio"]
                if isinstance(audio_data, dict):
                    audio_array = audio_data["array"]
                    audio_sr = audio_data["sampling_rate"]
                else:
                    audio_array = audio_data
                    audio_sr = 22000
                
                # Convert to float32 numpy array
                audio_array = np.array(audio_array, dtype=np.float32)
                
                # Resample to 24kHz if needed
                if audio_sr != 24000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=audio_sr, target_sr=24000)
                    audio_sr = 24000
                
                # Encode audio to get audio codes
                with torch.no_grad():
                    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).to(device)
                    audio_codes = model.codec_encoder(audio_tensor)
                    audio_codes = audio_codes.squeeze(0).cpu().numpy().tolist()
                
                # Prepare data entry
                data_entry = {
                    "audio": f"sample_{idx}.wav",  # Placeholder, actual audio is in array
                    "text": item["text"],
                    "audio_codes": audio_codes,
                    "language": item.get("language", "ha"),
                    "ref_audio": ref_audio_path,
                    "ref_text": ref_text,
                    "speaker_id": item.get("speaker_id", "unknown"),
                    "gender": item.get("gender", "unknown"),
                    "age_range": item.get("age_range", "unknown"),
                    "phase": item.get("phase", "unknown")
                }
                
                # Write to JSONL
                f.write(json.dumps(data_entry, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue
    
    print(f"Data preparation complete! Saved to {output_jsonl}")
    print(f"Total samples processed: {len(hf_dataset)}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Hausa TTS data for training")
    
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "validation", "test"],
                       help="Dataset split to process")
    parser.add_argument("--output_jsonl", type=str, required=True,
                       help="Output JSONL file path")
    parser.add_argument("--model_path", type=str,
                       default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                       help="Path to Qwen3-TTS model")
    parser.add_argument("--ref_audio_path", type=str, default=None,
                       help="Path to reference audio")
    parser.add_argument("--ref_text", type=str, default=None,
                       help="Reference text")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use for processing")
    
    args = parser.parse_args()
    
    prepare_hausa_data(
        split=args.split,
        output_jsonl=args.output_jsonl,
        model_path=args.model_path,
        ref_audio_path=args.ref_audio_path,
        ref_text=args.ref_text,
        max_samples=args.max_samples,
        device=args.device
    )


if __name__ == "__main__":
    main()
