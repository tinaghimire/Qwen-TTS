#!/usr/bin/env python3
# coding=utf-8
"""
Unified Dataset Loading and Preparation Tool for Qwen3-TTS.

This module provides a comprehensive toolkit for:
1. Loading datasets from Hugging Face
2. Preparing audio codes for training
3. Saving/loading data in JSONL format
4. Creating PyTorch DataLoaders

Usage:
    # Prepare data from Hugging Face
    from dataset_tool import prepare_dataset, load_jsonl_dataset
    
    # Prepare and save to JSONL
    prepare_dataset(
        dataset_name="vaghawan/hausa-tts-22k",
        split="train",
        output_jsonl="./data/train.jsonl",
        tokenizer_path="Qwen/Qwen3-TTS-Tokenizer-12Hz"
    )
    
    # Load from JSONL
    dataset = load_jsonl_dataset("./data/train.jsonl")
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS", "finetuning"))

from qwen_tts import Qwen3TTSTokenizer


class HausaTTSDataset(Dataset):
    """
    PyTorch Dataset for Hausa TTS data loaded from JSONL.
    """
    
    def __init__(self, jsonl_path: str):
        """
        Initialize dataset from JSONL file.
        
        Args:
            jsonl_path: Path to JSONL file
        """
        self.jsonl_path = jsonl_path
        self.data = []
        
        print(f"Loading dataset from {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        
        print(f"Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def prepare_dataset(
    dataset_name: str = "vaghawan/hausa-tts-22k",
    split: str = "train",
    output_jsonl: str = None,
    tokenizer_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    ref_audio_path: str = None,
    ref_text: str = None,
    max_samples: int = None,
    device: str = "cuda"
) -> str:
    """
    Prepare dataset from Hugging Face and save to JSONL format.
    
    Args:
        dataset_name: Hugging Face dataset name
        split: Dataset split (train, validation, test)
        output_jsonl: Output JSONL file path
        tokenizer_path: Path to Qwen3-TTS tokenizer for encoding audio
        ref_audio_path: Path to reference audio
        ref_text: Reference text for voice cloning
        max_samples: Maximum number of samples to process
        device: Device to use for processing
    
    Returns:
        Path to the output JSONL file
    """
    # Set default output path
    if output_jsonl is None:
        output_jsonl = f"./data/{split}.jsonl"
    
    # Set default reference audio path (24kHz version)
    if ref_audio_path is None:
        ref_audio_path = os.path.join(
            os.path.dirname(__file__),
            "voices", "english_voice", "english_voice_24k.wav"
        )
    
    # Set default reference text
    if ref_text is None:
        ref_text = "MTN Entertainment and Lifestyle. Entertainment and Lifestyle are at the heart of MTN's offering. We bring you music, movies, games and more through our digital platforms. With MTN musicals, you can stream your favorite"
    
    print("="*60)
    print(f"Preparing {split} dataset")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Output: {output_jsonl}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")
    print("="*60)
    
    # Load tokenizer
    print(f"\nLoading Qwen3-TTS tokenizer from {tokenizer_path}...")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_path,
        device_map=device,
    )
    print(f"✓ Tokenizer loaded")
    
    # Load dataset
    print(f"\nLoading dataset from Hugging Face ({split} split)...")
    hf_dataset = load_dataset(dataset_name, split=split)
    
    # Limit samples if specified
    if max_samples is not None:
        hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))
    
    print(f"Processing {len(hf_dataset)} samples...")
    
    # Prepare output directory
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    
    # Process each sample
    processed_count = 0
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(hf_dataset, desc=f"Processing {split}")):
            try:
                # Extract audio data
                audio_data = item["audio"]
                
                # Handle different audio data formats
                if isinstance(audio_data, dict):
                    audio_array = audio_data["array"]
                    audio_sr = audio_data["sampling_rate"]
                elif hasattr(audio_data, 'get_all_samples'):
                    # AudioDecoder object from datasets library (torchcodec)
                    decoded = audio_data.get_all_samples()
                    # AudioSamples has a 'data' attribute that contains the actual tensor
                    if hasattr(decoded, 'data'):
                        audio_array = decoded.data.numpy()
                    else:
                        # Fallback: try to convert directly
                        audio_array = torch.tensor(decoded).numpy()
                    audio_sr = audio_data.metadata.sample_rate
                elif hasattr(audio_data, 'array') and hasattr(audio_data, 'sampling_rate'):
                    # Standard Audio feature from datasets
                    audio_array = audio_data.array
                    audio_sr = audio_data.sampling_rate
                else:
                    # Fallback: assume it's the array itself
                    audio_array = audio_data
                    audio_sr = 22000
                
                # Convert to float32 numpy array - handle nested structures
                if isinstance(audio_array, (list, tuple)):
                    # Flatten nested lists if present
                    audio_array = np.array(audio_array, dtype=object)
                    if audio_array.ndim > 1:
                        # Flatten if it's a nested structure
                        audio_array = audio_array.flatten()
                    audio_array = audio_array.astype(np.float32)
                elif isinstance(audio_array, np.ndarray):
                    if audio_array.dtype != np.float32:
                        audio_array = audio_array.astype(np.float32)
                else:
                    audio_array = np.array([audio_array], dtype=np.float32)
                
                # Resample to 24kHz if needed
                if audio_sr != 24000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=audio_sr, target_sr=24000)
                    audio_sr = 24000
                
                # Encode audio to get audio codes using the tokenizer
                with torch.no_grad():
                    # Tokenizer expects audio array with sr parameter
                    enc_result = tokenizer.encode([audio_array], sr=audio_sr)
                    audio_codes = enc_result.audio_codes[0].cpu().numpy().tolist()
                
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
                processed_count += 1
                
            except Exception as e:
                print(f"\nError processing sample {idx}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print(f"Data preparation complete!")
    print(f"Output: {output_jsonl}")
    print(f"Total samples processed: {processed_count}/{len(hf_dataset)}")
    print(f"{'='*60}")
    
    return output_jsonl


def load_jsonl_dataset(jsonl_path: str) -> HausaTTSDataset:
    """
    Load dataset from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
    
    Returns:
        HausaTTSDataset instance
    """
    return HausaTTSDataset(jsonl_path)


def create_dataloader(
    jsonl_path: str,
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """
    Create PyTorch DataLoader from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    dataset = load_jsonl_dataset(jsonl_path)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda x: x  # Return list of samples
    )
    
    return dataloader


def get_dataset_info(jsonl_path: str) -> Dict:
    """
    Get information about a dataset from JSONL file.
    
    Args:
        jsonl_path: Path to JSONL file
    
    Returns:
        Dictionary with dataset information
    """
    info = {
        "path": jsonl_path,
        "num_samples": 0,
        "languages": set(),
        "speakers": set(),
        "genders": set(),
        "age_ranges": set(),
        "text_lengths": []
    }
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            info["num_samples"] += 1
            info["languages"].add(data.get("language", "unknown"))
            info["speakers"].add(data.get("speaker_id", "unknown"))
            info["genders"].add(data.get("gender", "unknown"))
            info["age_ranges"].add(data.get("age_range", "unknown"))
            info["text_lengths"].append(len(data.get("text", "")))
    
    # Convert sets to lists for JSON serialization
    info["languages"] = sorted(list(info["languages"]))
    info["speakers"] = sorted(list(info["speakers"]))
    info["genders"] = sorted(list(info["genders"]))
    info["age_ranges"] = sorted(list(info["age_ranges"]))
    
    if info["text_lengths"]:
        info["avg_text_length"] = sum(info["text_lengths"]) / len(info["text_lengths"])
        info["min_text_length"] = min(info["text_lengths"])
        info["max_text_length"] = max(info["text_lengths"])
    
    return info


def main():
    """Command-line interface for dataset preparation."""
    parser = argparse.ArgumentParser(
        description="Dataset Loading and Preparation Tool for Qwen3-TTS"
    )
    
    # Dataset preparation arguments
    parser.add_argument("--dataset_name", type=str, default="vaghawan/hausa-tts-22k",
                       help="Hugging Face dataset name")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "validation", "test"],
                       help="Dataset split to process")
    parser.add_argument("--output_jsonl", type=str, default=None,
                       help="Output JSONL file path")
    parser.add_argument("--tokenizer_path", type=str,
                       default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
                       help="Path to Qwen3-TTS tokenizer")
    parser.add_argument("--ref_audio_path", type=str, default=None,
                       help="Path to reference audio")
    parser.add_argument("--ref_text", type=str, default=None,
                       help="Reference text")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use for processing")
    
    # Dataset info arguments
    parser.add_argument("--info", type=str, default=None,
                       help="Get info about a JSONL dataset (provide path)")
    
    args = parser.parse_args()
    
    # Get dataset info
    if args.info:
        info = get_dataset_info(args.info)
        print("\n" + "="*60)
        print("Dataset Information")
        print("="*60)
        print(f"Path: {info['path']}")
        print(f"Number of samples: {info['num_samples']}")
        print(f"Languages: {', '.join(info['languages'])}")
        print(f"Speakers: {', '.join(info['speakers'])}")
        print(f"Genders: {', '.join(info['genders'])}")
        print(f"Age ranges: {', '.join(info['age_ranges'])}")
        if 'avg_text_length' in info:
            print(f"Average text length: {info['avg_text_length']:.1f}")
            print(f"Text length range: {info['min_text_length']} - {info['max_text_length']}")
        print("="*60)
        return
    
    # Prepare dataset
    prepare_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        output_jsonl=args.output_jsonl,
        tokenizer_path=args.tokenizer_path,
        ref_audio_path=args.ref_audio_path,
        ref_text=args.ref_text,
        max_samples=args.max_samples,
        device=args.device
    )


if __name__ == "__main__":
    main()