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
load_dotenv()

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS", "finetuning"))

from qwen_tts import Qwen3TTSTokenizer

# Configuration from .env
DATASET_NAME = os.getenv("DATASET_NAME", "vaghawan/hausa-tts-22k")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "train")
OUTPUT_JSONL = os.getenv("OUTPUT_JSONL", "./data/train.jsonl")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "Qwen/Qwen3-TTS-Tokenizer-12Hz")
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", None)
REF_TEXT = os.getenv("REF_TEXT", None)
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES")) if os.getenv("MAX_SAMPLES") else None
DEVICE = os.getenv("DEVICE", "cuda")


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
    split: str = DATASET_SPLIT,
    output_jsonl: str = OUTPUT_JSONL,
    tokenizer_path: str = TOKENIZER_PATH,
    ref_audio_path: str = REF_AUDIO_PATH,
    ref_text: str = REF_TEXT,
    max_samples: int = MAX_SAMPLES,
    device: str = DEVICE,
) -> str:
    if output_jsonl is None:
        output_jsonl = f"./data/{split}.jsonl"

    if ref_audio_path is None:
        ref_audio_path = os.path.join(
            os.path.dirname(__file__),
            "voices", "english_voice", "english_voice_24k.wav"
        )

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

    print(f"\nLoading Qwen3-TTS tokenizer from {tokenizer_path}...")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_path,
        device_map=device,
    )
    print(f"✓ Tokenizer loaded")

    print(f"\nLoading dataset from Hugging Face ({split} split)...")
    hf_dataset = load_dataset(dataset_name, split=split)

    if max_samples is not None:
        hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

    print(f"Processing {len(hf_dataset)} samples...")

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)

    processed_count = 0
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(tqdm(hf_dataset, desc=f"Processing {split}")):
            try:
                audio_data = item["audio"]

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

                with torch.no_grad():
                    enc_result = tokenizer.encode([audio_array], sr=audio_sr)
                    audio_codes = enc_result.audio_codes[0].cpu().numpy().tolist()

                data_entry = {
                    "audio": f"sample_{idx}.wav",
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
    return HausaTTSDataset(jsonl_path)


def get_dataset_info(jsonl_path: str) -> Dict:
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
    if DATASET_INFO_PATH := os.getenv("DATASET_INFO_PATH"):
        info = get_dataset_info(DATASET_INFO_PATH)
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
    else:
        prepare_dataset()


if __name__ == "__main__":
    main()