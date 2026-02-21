#!/usr/bin/env python3
"""
Simple test to verify the setup is working correctly.
Run this after uv sync to ensure all dependencies are installed.
"""

import sys


def test_python_version():
    """Test Python version compatibility."""
    print("Testing Python version...")
    import sys
    
    version = sys.version_info
    print(f"  Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 10 <= version.minor <= 12:
        print(f"  ✓ Python version is compatible (3.10-3.12)")
        return True
    else:
        print(f"  ✗ Python version is not compatible (requires 3.10-3.12)")
        return False


def test_imports():
    """Test that all required packages can be imported."""
    print("\nTesting imports...")
    
    packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("websockets", "WebSockets"),
        ("librosa", "Librosa"),
        ("scipy", "SciPy"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("soundfile", "SoundFile"),
        ("huggingface_hub", "HuggingFace Hub"),
        ("transformers", "Transformers"),
        ("dotenv", "python-dotenv"),
    ]
    
    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)
    
    return len(failed) == 0, failed


def test_cuda():
    """Test if CUDA is available."""
    print("\nTesting CUDA availability...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA is available")
            print(f"    Device: {torch.cuda.get_device_name(0)}")
            print(f"    Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print(f"  ⚠ CUDA is not available (CPU mode)")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False


def test_qwen_tts():
    """Test if Qwen3-TTS can be imported."""
    print("\nTesting Qwen3-TTS import...")
    try:
        # Add Qwen3-TTS to Python path
        import sys
        from pathlib import Path
        qwen_path = Path(__file__).parent / "Qwen3-TTS"
        if qwen_path.exists():
            sys.path.insert(0, str(qwen_path))
        
        from qwen_tts import Qwen3TTSModel
        print(f"  ✓ Qwen3-TTS can be imported")
        return True
    except ImportError as e:
        print(f"  ✗ Qwen3-TTS import failed: {e}")
        print(f"    Make sure Qwen3-TTS directory exists in the project")
        return False


def test_env_file():
    """Test if .env file exists."""
    print("\nTesting .env file...")
    import os
    from pathlib import Path
    
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        print(f"  ✓ .env file exists")
        
        # Check for required variables
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            required_vars = [
                "INIT_MODEL_PATH",
                "OUTPUT_MODEL_PATH",
                "BATCH_SIZE",
                "LR",
                "NUM_EPOCHS",
            ]
            
            missing = []
            for var in required_vars:
                if not os.getenv(var):
                    missing.append(var)
            
            if not missing:
                print(f"  ✓ All required environment variables are set")
                return True
            else:
                print(f"  ⚠ Missing environment variables: {', '.join(missing)}")
                print(f"    Run: python setup_env.py to create .env file")
                return False
        except Exception as e:
            print(f"  ✗ Error loading .env: {e}")
            return False
    else:
        print(f"  ⚠ .env file does not exist")
        print(f"    Run: python setup_env.py to create .env file from .env.training.example")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Qwen3-TTS Training Setup Verification")
    print("="*60)
    print()
    
    # Test Python version
    python_ok = test_python_version()
    
    # Test imports
    imports_ok, failed = test_imports()
    
    # Test CUDA
    cuda_ok = test_cuda()
    
    # Test Qwen3-TTS
    qwen_ok = test_qwen_tts()
    
    # Test .env file
    env_ok = test_env_file()
    
    # Summary
    print()
    print("="*60)
    print("Summary")
    print("="*60)
    
    if python_ok and imports_ok and qwen_ok and env_ok:
        print("✓ All dependencies and configuration are correct!")
        if cuda_ok:
            print("✓ CUDA is available for GPU acceleration")
        else:
            print("⚠ CUDA is not available (will use CPU)")
        print()
        print("You can now run:")
        print("  python train_using_sft.py          # Simple training")
        print("  python train_wandb_validation.py   # Advanced training")
        return 0
    else:
        print("✗ Setup incomplete!")
        if not python_ok:
            print("  Python version is not compatible (requires 3.10-3.12)")
        if failed:
            print(f"  Missing packages: {', '.join(failed)}")
        if not qwen_ok:
            print("  Qwen3-TTS cannot be imported")
        if not env_ok:
            print("  .env file is missing or incomplete")
        print()
        print("Please run:")
        print("  uv sync                             # Install dependencies")
        print("  python setup_env.py                 # Create .env file")
        print("  python test_setup.py                # Re-run this test")
        return 1


if __name__ == "__main__":
    sys.exit(main())