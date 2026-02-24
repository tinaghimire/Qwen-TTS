#!/usr/bin/env python3
# coding=utf-8
"""
Unified Data Processing Module for Qwen3-TTS Training.

Provides two approaches for data handling:
1. JSONLDataPreparer: Prepare data from HuggingFace to JSONL with audio codes
2. HFDirectDataLoader: Load data directly from HuggingFace for on-the-fly processing

Configuration is read from .env file.
"""

import json
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
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES")) if os.getenv("MAX_TRAIN_SAMPLES") else None
MAX_EVAL_SAMPLES = int(os.getenv("MAX_EVAL_SAMPLES")) if os.getenv("MAX_EVAL_SAMPLES") else None
DEVICE = os.getenv("DEVICE", "cuda")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./data")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))


class JSONLDataPreparer:
    """
    Prepare data from HuggingFace to JSONL files with pre-computed audio codes.

    This approach:
    - Downloads data from HuggingFace
    - Encodes audio to codes using Qwen3-TTS tokenizer
    - Saves to JSONL files for fast loading during training
    - Best for: Large datasets, repeated training runs, offline training

    Usage:
        preparer = JSONLDataPreparer(
            dataset_name="vaghawan/hausa-tts-22k",
            output_dir="./data",
            tokenizer_path="Qwen/Qwen3-TTS-Tokenizer-12Hz"
        )
        output_files = preparer.prepare_all_splits()
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        output_dir: str = OUTPUT_DIR,
        tokenizer_path: str = TOKENIZER_PATH,
        ref_audio_path: Optional[str] = REF_AUDIO_PATH,
        max_samples: Optional[int] = None,
        device: str = DEVICE,
        batch_size: int = BATCH_SIZE,
    ):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.tokenizer_path = tokenizer_path
        self.ref_audio_path = ref_audio_path
        self.max_samples = max_samples
        self.device = device
        self.batch_size = batch_size

        # Set default reference audio
        if self.ref_audio_path is None:
            self.ref_audio_path = os.path.join(
                os.path.dirname(__file__),
                "voices", "english_voice", "english_voice_24k.wav"
            )


        # Load tokenizer
        print(f"Loading Qwen3-TTS tokenizer from {tokenizer_path}...")
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_path,
            device_map=device,
        )
        print(f"✓ Tokenizer loaded on {device}")

    def prepare_split(
        self,
        split: str,
        max_samples: Optional[int] = None,
    ) -> str:
        """
        Prepare a single split and save to JSONL.

        Args:
            split: Dataset split name (train, validation, test)
            max_samples: Maximum number of samples to process

        Returns:
            Path to the output JSONL file
        """
        print(f"\n{'='*60}")
        print(f"Preparing {split} split")
        print(f"{'='*60}")

        # Load dataset
        print(f"Loading {split} split from HuggingFace...")
        hf_dataset = load_dataset(self.dataset_name, split=split, keep_in_memory=False)

        # Limit samples if specified
        if max_samples is not None:
            hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

        print(f"Processing {len(hf_dataset)} samples...")

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        output_jsonl = os.path.join(self.output_dir, f"{split}.jsonl")

        processed_count = 0
        skipped_count = 0

        # Process in batches for GPU efficiency
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            batch_audio_arrays = []
            batch_sample_rates = []
            batch_texts = []
            batch_indices = []

            for idx, item in enumerate(tqdm(hf_dataset, desc=f"Processing {split}")):
                try:
                    audio_data = item["audio"]

                    # Extract audio array and sampling rate
                    audio_array, audio_sr = self._extract_audio(audio_data)

                    # Resample to 24kHz if needed
                    if audio_sr != 24000:
                        audio_array = self._resample_audio(audio_array, audio_sr, 24000)
                        audio_sr = 24000

                    # Add to batch
                    batch_audio_arrays.append(audio_array)
                    batch_sample_rates.append(audio_sr)
                    batch_texts.append(item["text"])
                    batch_indices.append(idx)

                    # Process batch when it reaches batch_size
                    if len(batch_audio_arrays) >= self.batch_size:
                        with torch.no_grad():
                            enc_result = self.tokenizer.encode(batch_audio_arrays, sr=24000)
                            audio_codes_list = enc_result.audio_codes

                        # Write batch results
                        for i, (codes, text, orig_idx) in enumerate(zip(audio_codes_list, batch_texts, batch_indices)):
                            data_entry = {
                                "audio": f"sample_{orig_idx}.wav",
                                "text": text,
                                "audio_codes": codes.cpu().numpy().tolist(),
                                "ref_audio": self.ref_audio_path
                            }
                            f.write(json.dumps(data_entry, ensure_ascii=False) + '\n')
                            processed_count += 1

                        # Clear batch
                        batch_audio_arrays = []
                        batch_sample_rates = []
                        batch_texts = []
                        batch_indices = []

                except Exception as e:
                    error_msg = str(e)
                    if any(err in error_msg for err in ["End of file", "Failed to open", "Invalid data", "Could not receive frame"]):
                        print(f"\n⚠ Corrupted audio file at sample {idx}: {e}")
                        print(f"   Skipping this sample and continuing...")
                        skipped_count += 1
                        continue
                    else:
                        raise

            # Process remaining samples in the last batch
            if batch_audio_arrays:
                with torch.no_grad():
                    enc_result = self.tokenizer.encode(batch_audio_arrays, sr=24000)
                    audio_codes_list = enc_result.audio_codes

                # Write batch results
                for i, (codes, text, orig_idx) in enumerate(zip(audio_codes_list, batch_texts, batch_indices)):
                    data_entry = {
                        "audio": f"sample_{orig_idx}.wav",
                        "text": text,
                        "audio_codes": codes.cpu().numpy().tolist(),
                        "ref_audio": self.ref_audio_path
                    }
                    f.write(json.dumps(data_entry, ensure_ascii=False) + '\n')
                    processed_count += 1

        print(f"\n{split} split complete!")
        print(f"Output: {output_jsonl}")
        print(f"Samples processed: {processed_count}/{len(hf_dataset)}")
        if skipped_count > 0:
            print(f"Skipped samples (corrupted/invalid): {skipped_count}")

        return output_jsonl

    def prepare_all_splits(
        self,
        splits: Optional[List[str]] = None,
        max_samples_per_split: Optional[int] = None,
    ) -> List[str]:
        """
        Prepare all splits and save to JSONL files.

        Args:
            splits: List of splits to prepare (default: all available)
            max_samples_per_split: Maximum samples per split

        Returns:
            List of paths to output JSONL files
        """
        if splits is None:
            # Load dataset to get available splits
            hf_dataset = load_dataset(self.dataset_name, keep_in_memory=False)
            splits = list(hf_dataset.keys())

        print("="*60)
        print(f"Preparing dataset from HuggingFace to JSONL")
        print("="*60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Splits: {splits}")
        print(f"Max samples per split: {max_samples_per_split if max_samples_per_split else 'All'}")
        print(f"Batch size: {self.batch_size}")
        print(f"Device: {self.device}")
        print("="*60)

        output_files = []
        total_processed = 0
        total_skipped = 0

        for split in splits:
            max_samples = max_samples_per_split or self.max_samples
            output_file = self.prepare_split(split, max_samples)
            output_files.append(output_file)

            # Count processed samples
            with open(output_file, 'r') as f:
                total_processed += sum(1 for _ in f)

        print(f"\n{'='*60}")
        print(f"All data preparation complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"Total samples processed: {total_processed}")
        if total_skipped > 0:
            print(f"Total skipped samples: {total_skipped}")
        print(f"Output files: {output_files}")
        print(f"{'='*60}")

        return output_files

    def _extract_audio(self, audio_data: Any) -> Tuple[np.ndarray, int]:
        """Extract audio array and sampling rate from various formats."""
        if isinstance(audio_data, dict) and "array" in audio_data:
            audio_array = audio_data["array"]
            audio_sr = audio_data["sampling_rate"]
        elif isinstance(audio_data, dict) and "bytes" in audio_data:
            import io
            import soundfile as sf
            audio_bytes = audio_data["bytes"]
            with io.BytesIO(audio_bytes) as f_audio:
                audio_array, audio_sr = sf.read(f_audio, dtype="float32")
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


class HFDirectDataLoader(TorchDataset):
    """
    PyTorch Dataset that loads directly from HuggingFace and processes on-the-fly.

    This approach:
    - Loads data directly from HuggingFace during training
    - Encodes audio to codes on-the-fly
    - No intermediate JSONL files needed
    - Best for: Quick experiments, small datasets, online training

    Usage:
        dataset = HFDirectDataLoader(
            dataset_name="vaghawan/hausa-tts-22k",
            split="train",
            tokenizer_path="Qwen/Qwen3-TTS-Tokenizer-12Hz"
        )
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    """

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = TRAIN_SPLIT,
        tokenizer_path: str = TOKENIZER_PATH,
        ref_audio_path: Optional[str] = REF_AUDIO_PATH,
        max_samples: Optional[int] = None,
        audio_sr_device: str = DEVICE,
        cache_dir: Optional[str] = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer_path = tokenizer_path
        self.ref_audio_path = ref_audio_path
        self.max_samples = max_samples
        self.audio_sr_device = audio_sr_device

        # Set default reference audio
        if self.ref_audio_path is None:
            self.ref_audio_path = os.path.join(
                os.path.dirname(__file__),
                "voices", "english_voice", "english_voice_24k.wav"
            )

        print("="*60)
        print(f"Loading {split} dataset from HuggingFace (Direct)")
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
    max_samples: Optional[int] = None,
    audio_sr_device: str = DEVICE,
    num_workers: int = 0,
    shuffle: bool = True,
    cache_dir: Optional[str] = None,
) -> TorchDataLoader:
    """
    Create a PyTorch DataLoader for training or evaluation using direct HuggingFace loading.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split (train, validation, test)
        batch_size: Batch size for DataLoader
        tokenizer_path: Path to Qwen3-TTS tokenizer
        ref_audio_path: Reference audio path for voice cloning
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
    dataset = HFDirectDataLoader(
        dataset_name=dataset_name,
        split=split,
        tokenizer_path=tokenizer_path,
        ref_audio_path=ref_audio_path,
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

    parser = argparse.ArgumentParser(description="Qwen3-TTS Data Processing Tool")
    parser.add_argument(
        "--mode",
        choices=["prepare", "direct", "info"],
        default="prepare",
        help="Mode: prepare (HuggingFace to JSONL), direct (direct HuggingFace loading), info (dataset info)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for dataloader"
    )

    args = parser.parse_args()

    if args.mode == "prepare":
        # Prepare data from HuggingFace to JSONL
        preparer = JSONLDataPreparer(
            dataset_name=DATASET_NAME,
            output_dir=OUTPUT_DIR,
            tokenizer_path=TOKENIZER_PATH,
            batch_size=args.batch_size,
        )

        if args.split:
            # Prepare single split
            output_file = preparer.prepare_split(args.split, args.max_samples)
            print(f"\n✓ Prepared {args.split} split: {output_file}")
        else:
            # Prepare all splits
            output_files = preparer.prepare_all_splits(max_samples_per_split=args.max_samples)
            print(f"\n✓ Prepared all splits: {output_files}")

    elif args.mode == "direct":
        # Test direct HuggingFace loading
        split = args.split or TRAIN_SPLIT
        max_samples = args.max_samples or MAX_TRAIN_SAMPLES

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

    elif args.mode == "info":
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

    print("\nDone!")


if __name__ == "__main__":
    main()
