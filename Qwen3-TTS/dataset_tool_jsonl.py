#!/usr/bin/env python3
# coding=utf-8
"""
Dataset Loading and Preparation Tool for Qwen3-TTS.

All configuration is read from .env file.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS", "finetuning"))

from qwen_tts import Qwen3TTSTokenizer

# Configuration from .env
DATASET_NAME = os.getenv("DATASET_NAME", "vaghawan/hausa-tts-22k")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./data")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "Qwen/Qwen3-TTS-Tokenizer-12Hz")
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", None)
REF_TEXT = os.getenv("REF_TEXT", None)
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES")) if os.getenv("MAX_SAMPLES") else None
DEVICE = os.getenv("DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Batch size for GPU processing

class HausaTTSDataset:
    def __init__(self, jsonl_path: str):
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
    dataset_name: str = DATASET_NAME,
    output_dir: str = OUTPUT_DIR,
    tokenizer_path: str = TOKENIZER_PATH,
    ref_audio_path: str = REF_AUDIO_PATH,
    ref_text: str = REF_TEXT,
    max_samples: int = MAX_SAMPLES,
    device: str = DEVICE,
    batch_size: int = BATCH_SIZE,
    data_files: Optional[Dict[str, str]] = DATA_FILES,
) -> List[str]:
    if ref_audio_path is None:
        ref_audio_path = os.path.join(
            os.path.dirname(__file__),
            "voices", "english_voice", "english_voice_24k.wav"
        )

    if ref_text is None:
        ref_text = "MTN Entertainment and Lifestyle. Entertainment and Lifestyle are at the heart of MTN's offering. We bring you music, movies, games and more through our digital platforms. With MTN musicals, you can stream your favorite"

    print("="*60)
    print(f"Preparing dataset from all splits")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Output directory: {output_dir}")
    print(f"Tokenizer: {tokenizer_path}")
    print(f"Max samples per split: {max_samples if max_samples else 'All'}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    if data_files:
        valid_files = {k: v for k, v in data_files.items() if v is not None}
        if valid_files:
            print(f"Data files: {valid_files}")
        else:
            print(f"Data files: None (will load all splits)")
    print("="*60)

    print(f"\nLoading Qwen3-TTS tokenizer from {tokenizer_path}...")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_path,
        device_map=device,
    )
    print(f"✓ Tokenizer loaded")

    # Parse data_files if provided
    data_files_dict = None
    if data_files:
        print(f"\nData files configuration provided: {data_files}")

        # Filter out None values (splits without specified files)
        data_files_dict = {k: v for k, v in data_files.items() if v is not None}

        if data_files_dict:
            print(f"Data files to load: {data_files_dict}")
        else:
            print("⚠ Warning: No valid data files specified (all values are None)")
            print("   Will load all splits from dataset instead")
            data_files_dict = None

    print(f"\nLoading dataset from Hugging Face...")
    # Load all splits
    if data_files_dict:
        print(f"Loading specific data files for each split...")
        hf_dataset = load_dataset(dataset_name, data_files=data_files_dict, keep_in_memory=False)

        train_size = len(hf_dataset["train"])
        validation_size = len(hf_dataset["validation"])
        test_size = len(hf_dataset["test"])

        print(f"Train size: {train_size}")
        print(f"Validation size: {validation_size}")
        print(f"Test size: {test_size}")

        train_dataset = hf_dataset["train"].select(range(train_size))
        validation_dataset = hf_dataset["validation"].select(range(validation_size))
        test_dataset = hf_dataset["test"].select(range(test_size))

        hf_dataset = DatasetDict({
            "train": train_dataset,
            "validation": val_data,
            "test": test_data
        })

    else:
        print(f"Loading all splits from dataset...")
        hf_dataset = load_dataset(dataset_name, keep_in_memory=False)

    # Get available splits
    available_splits = list(hf_dataset.keys())
    print(f"Available splits: {available_splits}")

    os.makedirs(output_dir, exist_ok=True)

    output_files = []
    total_processed = 0
    total_skipped = 0

    # Process each split
    for split in available_splits:
        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")

        split_dataset = hf_dataset[split]

        if max_samples is not None:
            split_dataset = split_dataset.select(range(min(max_samples, len(split_dataset))))

        print(f"Processing {len(split_dataset)} samples...")

        output_jsonl = os.path.join(output_dir, f"{split}.jsonl")

        processed_count = 0
        skipped_count = 0

        # Process in batches for GPU efficiency
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            batch_audio_arrays = []
            batch_sample_rates = []
            batch_texts = []
            batch_indices = []

            for idx, item in enumerate(tqdm(split_dataset, desc=f"Processing {split}")):
                try:
                    audio_data = item["audio"]

                    # Handle audio data - if it's already decoded, use it; otherwise decode manually
                    try:
                        if isinstance(audio_data, dict) and "array" in audio_data:
                            # Audio is already decoded
                            audio_array = audio_data["array"]
                            audio_sr = audio_data["sampling_rate"]
                        elif isinstance(audio_data, dict) and "bytes" in audio_data:
                            # Audio is in bytes format, decode manually
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
                    except (RuntimeError, Exception) as e:
                        if any(err in str(e) for err in ["End of file", "Failed to open", "Invalid data", "Could not receive frame"]):
                            print(f"\n⚠ Corrupted audio file at sample {idx}: {e}")
                            print(f"   Skipping this sample and continuing...")
                            skipped_count += 1
                            continue
                        else:
                            raise

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

                    if audio_sr != 24000:
                        import librosa
                        audio_array = librosa.resample(audio_array, orig_sr=audio_sr, target_sr=24000)
                        audio_sr = 24000

                    # Add to batch
                    batch_audio_arrays.append(audio_array)
                    batch_sample_rates.append(audio_sr)
                    batch_texts.append(item["text"])
                    batch_indices.append(idx)

                    # Process batch when it reaches batch_size
                    if len(batch_audio_arrays) >= batch_size:
                        with torch.no_grad():
                            enc_result = tokenizer.encode(batch_audio_arrays, sr=24000)
                            audio_codes_list = enc_result.audio_codes

                        # Write batch results
                        for i, (codes, text, orig_idx) in enumerate(zip(audio_codes_list, batch_texts, batch_indices)):
                            data_entry = {
                                "audio": f"sample_{orig_idx}.wav",
                                "text": text,
                                "audio_codes": codes.cpu().numpy().tolist(),
                                "ref_audio": ref_audio_path
                            }
                            f.write(json.dumps(data_entry, ensure_ascii=False) + '\n')
                            processed_count += 1

                        # Clear batch
                        batch_audio_arrays = []
                        batch_sample_rates = []
                        batch_texts = []
                        batch_indices = []

                except RuntimeError as e:
                    error_msg = str(e)
                    if any(err in error_msg for err in ["End of file", "Failed to open", "Invalid data", "Could not receive frame"]):
                        print(f"\n⚠ Corrupted audio file at sample {idx}: {e}")
                        print(f"   Skipping this sample and continuing...")
                        skipped_count += 1
                        continue
                    else:
                        raise
                except Exception as e:
                    print(f"\n⚠ Error processing sample {idx}: {e}")
                    print(f"   Skipping this sample and continuing...")
                    skipped_count += 1
                    continue

            # Process remaining samples in the last batch
            if batch_audio_arrays:
                with torch.no_grad():
                    enc_result = tokenizer.encode(batch_audio_arrays, sr=24000)
                    audio_codes_list = enc_result.audio_codes

                # Write batch results
                for i, (codes, text, orig_idx) in enumerate(zip(audio_codes_list, batch_texts, batch_indices)):
                    data_entry = {
                        "audio": f"sample_{orig_idx}.wav",
                        "text": text,
                        "audio_codes": codes.cpu().numpy().tolist(),
                        "ref_audio": ref_audio_path
                    }
                    f.write(json.dumps(data_entry, ensure_ascii=False) + '\n')
                    processed_count += 1

        print(f"\n{split} split complete!")
        print(f"Output: {output_jsonl}")
        print(f"Samples processed: {processed_count}/{len(split_dataset)}")
        if skipped_count > 0:
            print(f"Skipped samples (corrupted/invalid): {skipped_count}")

        output_files.append(output_jsonl)
        total_processed += processed_count
        total_skipped += skipped_count

    print(f"\n{'='*60}")
    print(f"All data preparation complete!")
    print(f"Output directory: {output_dir}")
    print(f"Total samples processed across all splits: {total_processed}")
    if total_skipped > 0:
        print(f"Total skipped samples (corrupted/invalid): {total_skipped}")
    print(f"Output files: {output_files}")
    print(f"{'='*60}")

    return output_files


def load_jsonl_dataset(jsonl_path: str) -> HausaTTSDataset:
    return HausaTTSDataset(jsonl_path)



def main():
    if DATASET_INFO_PATH := os.getenv("DATASET_INFO_PATH"):
        print(f"Loading dataset from {DATASET_INFO_PATH}...")
        dataset = load_jsonl_dataset(DATASET_INFO_PATH)
        print(f"Loaded {len(dataset)} samples")
    else:
        print("No dataset information path provided. Preparing dataset from all splits...")
        prepare_dataset()


if __name__ == "__main__":
    main()