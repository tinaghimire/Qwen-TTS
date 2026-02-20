#!/usr/bin/env python3
# coding=utf-8
"""
Example script for training Qwen3-TTS on Hausa TTS dataset.

This script demonstrates how to:
1. Load the Hausa TTS dataset from Hugging Face
2. Prepare audio codes for training
3. Train the model with the new Trainer class
4. Upload models to Hugging Face Hub

Usage:
    python example_hausa_training.py --output_dir ./hausa_output --num_epochs 3
"""

import os
import sys

# Add the finetuning directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS", "finetuning"))

from train_hausa import main


if __name__ == "__main__":
    # Example command line arguments
    # You can modify these or pass them via command line
    
    import argparse
    
    # Create a parser to show example usage
    parser = argparse.ArgumentParser(description="Train Qwen3-TTS on Hausa data")
    
    # Basic training arguments
    parser.add_argument("--init_model_path", type=str, 
                       default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                       help="Path to the initial model")
    parser.add_argument("--output_dir", type=str, default="./hausa_output",
                       help="Output directory for checkpoints")
    parser.add_argument("--ref_audio_path", type=str, default=None,
                       help="Path to reference audio (default: voices/english_voice/english_voice.wav)")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch size for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps")
    
    # Dataset settings
    parser.add_argument("--train_split", type=str, default="train",
                       help="Training split name")
    parser.add_argument("--validation_split", type=str, default="validation",
                       help="Validation split name")
    parser.add_argument("--max_train_samples", type=int, default=None,
                       help="Maximum number of training samples (for debugging)")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                       help="Maximum number of evaluation samples (for debugging)")
    
    # Speaker settings
    parser.add_argument("--speaker_name", type=str, default="hausa_speaker",
                       help="Name for the speaker in the model")
    
    # Logging and checkpointing
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    
    # WandB settings
    parser.add_argument("--use_wandb", action="store_true", default=True,
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="qwen3-tts-hausa",
                       help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="WandB run name")
    
    # Hugging Face upload settings
    parser.add_argument("--upload_to_hub", action="store_true", default=True,
                       help="Upload models to Hugging Face Hub")
    parser.add_argument("--hub_model_id_best", type=str, default="vaghawan/tts-best",
                       help="Hub repository ID for best model")
    parser.add_argument("--hub_model_id_last", type=str, default="vaghawan/tts-last",
                       help="Hub repository ID for last model")
    parser.add_argument("--hub_token", type=str, default=None,
                       help="Hugging Face API token")
    
    # Mixed precision
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                       choices=["no", "fp16", "bf16"],
                       help="Mixed precision training mode")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Qwen3-TTS Hausa Fine-tuning")
    print("="*60)
    print(f"Model: {args.init_model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Speaker name: {args.speaker_name}")
    print(f"Use WandB: {args.use_wandb}")
    print(f"Upload to Hub: {args.upload_to_hub}")
    print("="*60)
    
    # Start training
    main()
