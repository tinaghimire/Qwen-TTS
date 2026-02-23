#!/usr/bin/env python3
# coding=utf-8
"""
Training script for Hausa TTS using sft_12hz.py.

All configuration is read from .env file.
"""

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from .env
DATASET_NAME = os.getenv("DATASET_NAME", "vaghawan/hausa-tts-22k")
TRAIN_JSONL = os.getenv("TRAIN_JSONL", "./data/hausa_train.jsonl")
VALIDATION_JSONL = os.getenv("VALIDATION_JSONL", None)
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "Qwen/Qwen3-TTS-Tokenizer-12Hz")
REF_AUDIO_PATH = os.getenv("REF_AUDIO_PATH", None)
REF_TEXT = os.getenv("REF_TEXT", None)
DEVICE = os.getenv("DEVICE", "cuda")
INIT_MODEL_PATH = os.getenv("INIT_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
OUTPUT_MODEL_PATH = os.getenv("OUTPUT_MODEL_PATH", "./output")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 2))
LR = float(os.getenv("LR", 2e-5))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", 3))
SPEAKER_NAME = os.getenv("SPEAKER_NAME", "reference_speaker")
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES")) if os.getenv("MAX_TRAIN_SAMPLES") else None
MAX_EVAL_SAMPLES = int(os.getenv("MAX_EVAL_SAMPLES")) if os.getenv("MAX_EVAL_SAMPLES") else None

def prepare_data(train_only=False):
    print("="*60)
    print("Step 1: Preparing Hausa TTS data")
    print("="*60)

    train_cmd = [
        sys.executable,
        "dataset_tool.py",
    ]

    print(f"Running: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)

    if train_only:
        return

    if VALIDATION_JSONL:
        # Prepare validation data
        env = os.environ.copy()
        env["DATASET_SPLIT"] = "validation"
        env["OUTPUT_JSONL"] = VALIDATION_JSONL
        val_cmd = [
            sys.executable,
            "dataset_tool.py",
        ]
        print(f"Running validation data preparation...")
        subprocess.run(val_cmd, env=env, check=True)

    print("Data preparation complete!")


def train_model():
    print("\n" + "="*60)
    print("Step 2: Training model with sft_12hz.py")
    print("="*60)

    train_cmd = [
        sys.executable,
        "Qwen3-TTS/finetuning/sft_12hz.py",
        "--init_model_path", INIT_MODEL_PATH,
        "--output_model_path", OUTPUT_MODEL_PATH,
        "--train_jsonl", TRAIN_JSONL,
        "--batch_size", str(BATCH_SIZE),
        "--lr", str(LR),
        "--num_epochs", str(NUM_EPOCHS),
        "--speaker_name", SPEAKER_NAME
    ]

    print(f"Running: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)

    print("Training complete!")


def main():
    global REF_AUDIO_PATH, REF_TEXT

    if REF_AUDIO_PATH is None:
        REF_AUDIO_PATH = os.path.join(
            os.path.dirname(__file__),
            "voices", "english_voice", "english_voice_24k.wav"
        )

    if REF_TEXT is None:
        REF_TEXT = "MTN Entertainment and Lifestyle. Entertainment and Lifestyle are at the heart of MTN's offering. We bring you music, movies, games and more through our digital platforms. With MTN musicals, you can stream your favorite"

    if os.getenv("PREPARE_ONLY"):
        prepare_data()
        print("\n" + "="*60)
        print("Data preparation only (--prepare_only flag set)")
        print("="*60)
        return

    if not os.getenv("SKIP_PREPARE"):
        prepare_data()
    else:
        print("Skipping data preparation (--skip_prepare flag set)")
        prepare_data(train_only=True)

    train_model()

    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()