#!/usr/bin/env python3
# coding=utf-8
"""
Dataset Loading and Preparation Tool for Qwen3-TTS.

Loads data directly from HuggingFace and provides PyTorch DataLoader.

All configuration is read from .env file.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS", "finetuning"))

from qwen_tts import Qwen3TTSTokenizer

# Configuration from .env
DATASET_NAME = os.getenv("DATASET_NAME", "vaghawan/hausa-tts-22k")
TRAIN_SPLIT = os.getenv("TRAIN_SPLIT", "train")
VALIDATION_SPLIT = os.getenv("VALIDATION_SPLIT", "validation")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "Qwen/Qwen3-TTS-Tokenizer-12Hz")
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", None)
REF_TEXT = os.getenv("REF_TEXT", None)
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES")) if os.getenv("MAX_TRAIN_SAMPLES") else None
MAX_EVAL_SAMPLES = int(os.getenv("MAX_EVAL_SAMPLES")) if os.getenv("MAX_EVAL_SAMPLES") else None
DEVICE = os.getenv("DEVICE", "cuda")


class QwenTTSDataset(TorchDataset):
    """
    PyTorch Dataset that loads directly from HuggingFace and processes on-the-fly.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split (train, validation, test)
        tokenizer_path: Path to Qwen3-TTS tokenizer
        ref_audio_path: Reference audio path for voice cloning
        ref_text: Reference text for voice cloning
        max_samples: Maximum number of samples to load
        audio_sr_device: Device for audio processing
        cache_dir: Optional cache directory for datasets
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer_path: str,
        ref_audio_path: Optional[str] = None,
        ref_text: Optional[str] = None,
        max_samples: Optional[int] = None,
        audio_sr_device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer_path = tokenizer_path
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.max_samples = max_samples
        self.audio_sr_device = audio_sr_device

        # Set default reference audio
        if self.ref_audio_path is None:
            self.ref_audio_path = os.path.join(
                os.path.dirname(__file__),
                "voices", "english_voice", "english_voice_24k.wav"
            )

        # Set default reference text
        if self.ref_text is None:
            self.ref_text = "MTN Entertainment and Lifestyle. Entertainment and Lifestyle are at the heart of MTN's offering. We bring you music, movies, games and more through our digital platforms. With MTN musicals, you can stream your favorite"

        print("="*60)
        print(f"Loading {split} dataset from HuggingFace")
        print("="*60)
        print(f"Dataset: {dataset_name}")
        print(f"Split: {split}")
        print(f"Max samples: {max_samples if max_samples else 'All'}")
        print("="*60)

        # Load dataset
        print(f"\nLoading dataset from HuggingFace...")
        self.hf_dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

        # Limit samples if specified
        if max_samples is not None:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))

        print(f"✓ Loaded {len(self.hf_dataset)} samples")

        # Load tokenizer
        print(f"\nLoading Qwen3-TTS tokenizer from {tokenizer_path}...")
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_path,
            device_map=audio_sr_device,
        )
        print(f"✓ Tokenizer loaded on {audio_sr_device}")

        print("="*60)
        print("Dataset ready!")
        print("="*60)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.

        Returns:
            Dictionary containing:
                - audio: audio array (np.ndarray)
                - text: transcription text (str)
                - audio_codes: encoded audio codes (List[List[int]])
                - sr: sampling rate (int)
                - language: language code (str)
                - ref_audio: reference audio path (str)
                - ref_text: reference text (str)
        """
        item = self.hf_dataset[idx]
        audio_data = item.get("audio")

        # Extract audio array and sampling rate
        audio_array, audio_sr = self._extract_audio(audio_data)

        # Resample to 24kHz if needed
        if audio_sr != 24000:
            audio_array = self._resample_audio(audio_array, audio_sr, 24000)
            audio_sr = 24000

        # Encode audio to codes
        with torch.no_grad():
            enc_result = self.tokenizer.encode([audio_array], sr=audio_sr)
            audio_codes = enc_result.audio_codes[0].cpu().numpy().tolist()

        return {
            "audio": audio_array,
            "text": item.get("text", ""),
            "audio_codes": audio_codes,
            "sr": audio_sr,
            "language": item.get("language", "ha"),
            "ref_audio": self.ref_audio_path,
            "ref_text": self.ref_text,
        }

    def _extract_audio(self, audio_data: Any) -> Tuple[np.ndarray, int]:
        """Extract audio array and sampling rate from various formats."""
        if isinstance(audio_data, dict):
            audio_array = audio_data["array"]
            audio_sr = audio_data["sampling_rate"]
        elif hasattr(audio_data, 'get_all_samples'):
            decoded = audio_data.get_all_samples()
            if hasattr(decoded, 'data'):
                audio_array = decoded.data.numpy()
            else:
                audio_array = torch.tensor(decoded).numpy()
            audio_sr = audio_data.metadata.sample_rate
        elif hasattr(audio_data, 'array') and hasattr(audio_data, 'sampling_rate'):
            audio_array = audio_data.array
            audio_sr = audio_data.sampling_rate
        else:
            audio_array = audio_data
            audio_sr = 22000

        # Convert to float32 numpy array
        if isinstance(audio_array, (list, tuple)):
            audio_array = np.array(audio_array, dtype=object)
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()
            audio_array = audio_array.astype(np.float32)
        elif isinstance(audio_array, np.ndarray):
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
        else:
            audio_array = np.array([audio_array], dtype=np.float32)

        return audio_array, audio_sr

    def _resample_audio(self, audio_array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sampling rate."""
        import librosa
        return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)


def get_dataloader(
    dataset_name: str = DATASET_NAME,
    split: str = TRAIN_SPLIT,
    batch_size: int = 4,
    tokenizer_path: str = TOKENIZER_PATH,
    ref_audio_path: Optional[str] = REF_AUDIO_PATH,
    ref_text: Optional[str] = REF_TEXT,
    max_samples: Optional[int] = None,
    audio_sr_device: str = DEVICE,
    num_workers: int = 0,
    shuffle: bool = True,
    cache_dir: Optional[str] = None,
) -> TorchDataLoader:
    """
    Create a PyTorch DataLoader for training or evaluation.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split (train, validation, test)
        batch_size: Batch size for DataLoader
        tokenizer_path: Path to Qwen3-TTS tokenizer
        ref_audio_path: Reference audio path for voice cloning
        ref_text: Reference text for voice cloning
        max_samples: Maximum number of samples
        audio_sr_device: Device for audio processing
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle the data
        cache_dir: Optional cache directory

    Returns:
        PyTorch DataLoader ready for training
    """
    if split == "train":
        max_samples = max_samples or MAX_TRAIN_SAMPLES
    else:
        max_samples = max_samples or MAX_EVAL_SAMPLES

    # Create dataset
    dataset = QwenTTSDataset(
        dataset_name=dataset_name,
        split=split,
        tokenizer_path=tokenizer_path,
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        max_samples=max_samples,
        audio_sr_device=audio_sr_device,
        cache_dir=cache_dir,
    )

    # Create dataloader
    dataloader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if audio_sr_device == "cuda" else False,
        drop_last=False,
    )

    print(f"✓ DataLoader created with batch_size={batch_size}, num_workers={num_workers}")

    return dataloader


def get_train_dataloader(
    batch_size: int = 2,
    num_workers: int = 0,
    shuffle: bool = True,
    **kwargs
) -> TorchDataLoader:
    """
    Create training DataLoader with default settings from .env.

    Args:
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle
        **kwargs: Additional arguments for get_dataloader

    Returns:
        PyTorch DataLoader for training
    """
    return get_dataloader(
        dataset_name=DATASET_NAME,
        split=TRAIN_SPLIT,
        batch_size=batch_size,
        tokenizer_path=TOKENIZER_PATH,
        ref_audio_path=REF_AUDIO_PATH,
        ref_text=REF_TEXT,
        max_samples=MAX_TRAIN_SAMPLES,
        audio_sr_device=DEVICE,
        num_workers=num_workers,
        shuffle=shuffle,
        **kwargs
    )


def get_eval_dataloader(
    batch_size: int = 2,
    num_workers: int = 0,
    shuffle: bool = False,
    **kwargs
) -> TorchDataLoader:
    """
    Create evaluation DataLoader with default settings from .env.

    Args:
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle (usually False for eval)
        **kwargs: Additional arguments for get_dataloader

    Returns:
        PyTorch DataLoader for evaluation
    """
    return get_dataloader(
        dataset_name=DATASET_NAME,
        split=VALIDATION_SPLIT,
        batch_size=batch_size,
        tokenizer_path=TOKENIZER_PATH,
        ref_audio_path=REF_AUDIO_PATH,
        ref_text=REF_TEXT,
        max_samples=MAX_EVAL_SAMPLES,
        audio_sr_device=DEVICE,
        num_workers=num_workers,
        shuffle=shuffle,
        **kwargs
    )


def get_dataset_info(
    dataset_name: str = DATASET_NAME,
    split: str = TRAIN_SPLIT,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get information about a dataset without loading samples.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split
        cache_dir: Optional cache directory

    Returns:
        Dictionary with dataset information
    """
    print(f"\nLoading dataset info from HuggingFace...")
    hf_dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    first_item = hf_dataset[0]

    info = {
        "name": dataset_name,
        "split": split,
        "num_samples": len(hf_dataset),
        "features": list(hf_dataset.features.keys()),
        "languages": set(),
        "text_lengths": [],
    }

    # Sample up to 100 items to get statistics
    sample_size = min(100, len(hf_dataset))
    for i in tqdm(range(sample_size), desc="Analyzing dataset"):
        item = hf_dataset[i]
        info["languages"].add(item.get("language", "unknown"))
        info["text_lengths"].append(len(item.get("text", "")))

    info["languages"] = sorted(list(info["languages"]))

    if info["text_lengths"]:
        info["avg_text_length"] = sum(info["text_lengths"]) / len(info["text_lengths"])
        info["min_text_length"] = min(info["text_lengths"])
        info["max_text_length"] = max(info["text_lengths"])

    # Add audio information
    if "audio" in first_item:
        audio_data = first_item["audio"]
        if isinstance(audio_data, dict):
            info["audio_sampling_rate"] = audio_data.get("sampling_rate", None)
        elif hasattr(audio_data, 'metadata'):
            info["audio_sampling_rate"] = getattr(audio_data.metadata, 'sample_rate', None)

    return info


def main():
    """Main function for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Qwen3-TTS Dataset Tool")
    parser.add_argument(
        "--action",
        choices=["train", "eval", "info"],
        default="train",
        help="Action to perform: train (create train dataloader), eval (create eval dataloader), info (show dataset info)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for dataloader"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to load"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to use (overrides train/eval)"
    )

    args = parser.parse_args()

    if args.action == "info":
        # Show dataset information
        split = args.split or TRAIN_SPLIT
        info = get_dataset_info(dataset_name=DATASET_NAME, split=split)

        print("\n" + "="*60)
        print("Dataset Information")
        print("="*60)
        print(f"Name: {info['name']}")
        print(f"Split: {info['split']}")
        print(f"Number of samples: {info['num_samples']}")
        print(f"Features: {', '.join(info['features'])}")
        print(f"Languages: {', '.join(info['languages'])}")
        if 'avg_text_length' in info:
            print(f"Average text length: {info['avg_text_length']:.1f}")
            print(f"Text length range: {info['min_text_length']} - {info['max_text_length']}")
        if 'audio_sampling_rate' in info:
            print(f"Audio sampling rate: {info['audio_sampling_rate']}")
        print("="*60)

    elif args.action == "train" or (args.action == "eval" and not args.split):
        # Create and test dataloader
        max_samples = args.max_samples or (MAX_TRAIN_SAMPLES if args.action == "train" else MAX_EVAL_SAMPLES)
        split = args.split or (TRAIN_SPLIT if args.action == "train" else VALIDATION_SPLIT)

        dataloader = get_dataloader(
            dataset_name=DATASET_NAME,
            split=split,
            batch_size=args.batch_size,
            max_samples=max_samples,
            num_workers=args.num_workers,
        )

        print(f"\nTesting DataLoader with {len(dataloader)} batches...")

        # Iterate first batch to test
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing dataloader")):
            if batch_idx == 0:
                print(f"\nFirst batch sample keys: {batch[0].keys()}")
                print(f"Text sample: {batch[0]['text']}")
                print(f"Audio codes length: {len(batch[0]['audio_codes'])}")
                print(f"Audio sampling rate: {batch[0]['sr']}")
                print("\n✓ DataLoader working correctly!")
                break

    print("\nDone!")


if __name__ == "__main__":
    main()