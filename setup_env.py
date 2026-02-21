#!/usr/bin/env python3
"""
Setup script to create .env file from .env.training.example.
Run this to create your environment configuration.
"""

import os
import shutil
from pathlib import Path


def main():
    print("="*60)
    print("Qwen3-TTS Training Environment Setup")
    print("="*60)
    print()
    
    # Check if .env already exists
    env_path = Path(__file__).parent / ".env"
    example_path = Path(__file__).parent / ".env.training.example"
    
    if not example_path.exists():
        print("✗ .env.training.example not found!")
        print("  Please ensure this file exists in the same directory.")
        return 1
    
    if env_path.exists():
        response = input(".env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Setup cancelled.")
            return 0
    
    # Copy example to .env
    shutil.copy(example_path, env_path)
    print(f"✓ Created .env from .env.training.example")
    print()
    
    print("="*60)
    print("Next Steps")
    print("="*60)
    print()
    print("1. Edit .env file to configure your training settings:")
    print("   - Update DEVICE (cuda or cpu)")
    print("   - Update BATCH_SIZE based on your GPU memory")
    print("   - Update REF_AUDIO_PATH to point to your reference audio")
    print("   - Update HF_TOKEN if you want to upload models to Hugging Face")
    print()
    print("2. Verify your setup:")
    print("   python test_setup.py")
    print()
    print("3. Start training:")
    print("   python train_using_sft.py          # Simple training")
    print("   python train_wandb_validation.py   # Advanced training")
    print()
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())