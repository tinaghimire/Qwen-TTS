#!/usr/bin/env python3
# coding=utf-8
"""
Load JSONL dataset from Hugging Face and prepare for SFT fine-tuning.

Supports loading multiple splits (train, validation, test).

Configuration is read from .env file.

Usage:
    python load_from_huggingface.py
"""

import json
import os
from typing import List, Optional
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def load_dataset_from_hf(
    repo_id: str,
    split: str = "train",
    output_path: str = None,
    ref_audio_path: str = None,
) -> str:
    """
    Load dataset from Hugging Face and save as JSONL for SFT training.

    Args:
        repo_id: Hugging Face repository ID (e.g., "username/dataset-name")
        split: Dataset split to load
        output_path: Path to save the JSONL file
        ref_audio_path: Optional path to override ref_audio in the dataset

    Returns:
        Path to the saved JSONL file
    """
    print(f"\n📥 Loading {split} split...")
    dataset = load_dataset(repo_id, split=split)
    print(f"✓ Loaded {len(dataset)} samples")

    # Set default output path
    if output_path is None:
        output_path = f"./data/{split}.jsonl"

    # Create output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Save as JSONL
    print(f"💾 Saving to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            # Convert to dict and optionally override ref_audio
            data = dict(item)
            if ref_audio_path is not None:
                data['ref_audio'] = ref_audio_path

            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"✓ Saved {len(dataset)} samples to {output_path}")

    return output_path


def main():
    # Get configuration from environment variables
    repo_id = os.getenv("HF_REPO_ID")
    splits_str = os.getenv("HF_SPLITS", "train")
    output_dir = os.getenv("HF_OUTPUT_DIR", "./data")
    ref_audio_path = os.getenv("REF_AUDIO_PATH")

    # Parse splits
    splits = splits_str.split() if splits_str else ["train"]

    # Validate repo_id
    if not repo_id:
        raise ValueError("HF_REPO_ID environment variable must be provided in .env file")

    print("="*60)
    print("Loading Dataset from Hugging Face")
    print("="*60)
    print(f"Repository: {repo_id}")
    print(f"Splits: {', '.join(splits)}")
    print(f"Output directory: {output_dir}")
    if ref_audio_path:
        print(f"Reference audio override: {ref_audio_path}")
    print("="*60)

    # Load all requested splits
    loaded_files = []
    for split in splits:
        output_path = os.path.join(output_dir, f"{split}.jsonl")
        try:
            loaded_path = load_dataset_from_hf(
                repo_id=repo_id,
                split=split,
                output_path=output_path,
                ref_audio_path=ref_audio_path,
            )
            loaded_files.append(loaded_path)
        except Exception as e:
            print(f"⚠ Error loading {split} split: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Dataset ready for SFT training!")
    print(f"\nLoaded files:")
    for file in loaded_files:
        print(f"  - {file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
