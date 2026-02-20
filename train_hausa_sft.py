#!/usr/bin/env python3
# coding=utf-8
"""
Training script for Hausa TTS using sft_12hz.py.
This script prepares the data and then calls the sft_12hz.py training script.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def prepare_data(args):
    """Prepare Hausa TTS data for training."""
    print("="*60)
    print("Step 1: Preparing Hausa TTS data")
    print("="*60)
    
    # Prepare train data
    train_cmd = [
        sys.executable,
        "prepare_hausa_data.py",
        "--split", "train",
        "--output_jsonl", args.train_jsonl,
        "--model_path", args.init_model_path,
        "--ref_audio_path", args.ref_audio_path,
        "--ref_text", args.ref_text,
        "--max_samples", str(args.max_train_samples) if args.max_train_samples else "None",
        "--device", args.device
    ]
    
    # Remove None values
    train_cmd = [arg for arg in train_cmd if arg != "None"]
    
    print(f"Running: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, check=True)
    
    # Prepare validation data if specified
    if args.validation_jsonl:
        val_cmd = [
            sys.executable,
            "prepare_hausa_data.py",
            "--split", "validation",
            "--output_jsonl", args.validation_jsonl,
            "--model_path", args.init_model_path,
            "--ref_audio_path", args.ref_audio_path,
            "--ref_text", args.ref_text,
            "--max_samples", str(args.max_eval_samples) if args.max_eval_samples else "None",
            "--device", args.device
        ]
        
        val_cmd = [arg for arg in val_cmd if arg != "None"]
        
        print(f"Running: {' '.join(val_cmd)}")
        result = subprocess.run(val_cmd, check=True)
    
    print("Data preparation complete!")


def train_model(args):
    """Train the model using sft_12hz.py."""
    print("\n" + "="*60)
    print("Step 2: Training model with sft_12hz.py")
    print("="*60)
    
    # Build command for sft_12hz.py
    train_cmd = [
        sys.executable,
        "Qwen3-TTS/finetuning/sft_12hz.py",
        "--init_model_path", args.init_model_path,
        "--output_model_path", args.output_model_path,
        "--train_jsonl", args.train_jsonl,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--num_epochs", str(args.num_epochs),
        "--speaker_name", args.speaker_name
    ]
    
    print(f"Running: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, check=True)
    
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train Qwen3-TTS on Hausa data using sft_12hz.py"
    )
    
    # Data preparation arguments
    parser.add_argument("--train_jsonl", type=str, default="./data/hausa_train.jsonl",
                       help="Output JSONL file for training data")
    parser.add_argument("--validation_jsonl", type=str, default=None,
                       help="Output JSONL file for validation data (optional)")
    parser.add_argument("--ref_audio_path", type=str, default=None,
                       help="Path to reference audio")
    parser.add_argument("--ref_text", type=str, default=None,
                       help="Reference text")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Maximum number of training samples")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                       help="Maximum number of evaluation samples")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use for data preparation")
    
    # Training arguments (passed to sft_12hz.py)
    parser.add_argument("--init_model_path", type=str,
                       default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                       help="Path to initial model")
    parser.add_argument("--output_model_path", type=str, default="./output",
                       help="Output directory for trained model")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--speaker_name", type=str, default="hausa_speaker",
                       help="Speaker name for the model")
    
    # Workflow control
    parser.add_argument("--skip_prepare", action="store_true",
                       help="Skip data preparation if already done")
    parser.add_argument("--prepare_only", action="store_true",
                       help="Only prepare data, don't train")
    
    args = parser.parse_args()
    
    # Set default reference audio path
    if args.ref_audio_path is None:
        args.ref_audio_path = os.path.join(
            os.path.dirname(__file__),
            "voices", "english_voice", "english_voice.wav"
        )
    
    # Set default reference text
    if args.ref_text is None:
        args.ref_text = "MTN Entertainment and Lifestyle. Entertainment and Lifestyle are at the heart of MTN's offering. We bring you music, movies, games and more through our digital platforms. With MTN musicals, you can stream your favorite"
    
    print("="*60)
    print("Hausa TTS Training Pipeline")
    print("="*60)
    print(f"Model: {args.init_model_path}")
    print(f"Output: {args.output_model_path}")
    print(f"Train data: {args.train_jsonl}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Speaker name: {args.speaker_name}")
    print("="*60)
    
    # Step 1: Prepare data
    if not args.skip_prepare:
        prepare_data(args)
    else:
        print("Skipping data preparation (--skip_prepare flag set)")
    
    # Step 2: Train model
    if not args.prepare_only:
        train_model(args)
    else:
        print("Data preparation only (--prepare_only flag set)")
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()
