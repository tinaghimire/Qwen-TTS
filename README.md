# Qwen3-TTS WebSocket Server & Training Pipeline

A comprehensive WebSocket server for Qwen3-TTS voice cloning with support for Hausa TTS fine-tuning.

## 📋 Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [WebSocket Server](#websocket-server)
- [Training Pipeline](#training-pipeline)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## ✨ Features

### WebSocket Server
- **Real-time voice cloning** via WebSocket
- **ElevenLabs-compatible** endpoint
- **HTTP endpoint** for batch processing
- **Streaming support** for long texts
- **Time analysis** for audio generation
- **Multiple voice profiles** support

### Training Pipeline
- **Unified dataset tool** for loading and preparation
- **Simple training** for quick experiments
- **Advanced training** with validation, metrics, and WandB
- **Flexible training** with evaluation
- **Model upload** to Hugging Face Hub
- **Mixed precision training** support
- **Environment variable configuration** for reproducibility

## 🔧 Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **CUDA**: 13.0 (for GPU acceleration)
- **GPU**: NVIDIA GPU with compute capability 7.0+ (e.g., RTX 5070 Ti, RTX 4090, etc.)

Check your CUDA version:
```bash
nvidia-smi  # Look for "CUDA Version"
```

## 📦 Installation

### Using uv (Recommended)

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (requires Python 3.10-3.12)
uv sync
```

### Using pip

```bash
# Install PyTorch with CUDA 13.0 support
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchaudio

# Install other dependencies
pip install fastapi uvicorn websockets librosa scipy numpy soundfile \
    huggingface_hub transformers==4.57.3 accelerate==1.12.0 \
    einops sox onnxruntime datasets>=2.14.0 python-dotenv

# Install flash-attn (optional, for faster inference)
pip install flash-attn --no-build-isolation
```

### Setup Environment Configuration

```bash
# Create .env file from example template
python setup_env.py

# Or manually copy
cp .env.training.example .env

# Edit .env with your configuration
nano .env  # or your preferred editor
```

### Verify Installation

```bash
# Run tests to verify everything is set up correctly
python test_setup.py
```

## 🚀 Quick Start

```bash
# Setup environment
python setup_env.py

# Verify installation
python test_setup.py

# Simple training (recommended for beginners)
python train_using_sft.py

# Advanced training (with validation and WandB)
python train_wandb_validation.py
```

## 🎯 Training Pipeline

### Overview

Train Qwen3-TTS on the Hausa TTS dataset from Hugging Face for voice cloning in Hausa language.

All training is configured via environment variables in the `.env` file for reproducibility and easy configuration management.

### Dataset

- **Source**: `vaghawan/hausa-tts-22k`
- **Splits**: train, validation, test
- **Features**: audio, text, speaker_id, language, gender, age_range, phase

### Environment Configuration

The `.env` file controls all training parameters. Here's a summary of the key variables:

```bash
# Device and Model
DEVICE=cuda
INIT_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base
OUTPUT_MODEL_PATH=./output

# Dataset
DATASET_NAME=vaghawan/hausa-tts-22k
TRAIN_JSONL=./data/train.jsonl
VALIDATION_JSONL=./data/validation.jsonl
MAX_TRAIN_SAMPLES=
MAX_EVAL_SAMPLES=

# Training Hyperparameters
BATCH_SIZE=2
LR=2e-5
NUM_EPOCHS=3
WEIGHT_DECAY=0.01
WARMUP_STEPS=100
GRADIENT_ACCUMULATION_STEPS=4
MAX_GRAD_NORM=1.0

# Speaker Configuration
SPEAKER_NAME=hausa_speaker
REF_AUDIO_PATH=/path/to/reference/audio.wav
REF_TEXT="Your reference text here..."

# WandB (Advanced Training)
USE_WANDB=true
WANDB_PROJECT=qwen3-tts-hausa
WANDB_RUN_NAME=

# Hugging Face Hub (Advanced Training)
UPLOAD_TO_HUB=false
HUB_MODEL_ID_BEST=vaghawan/tts-best
HUB_MODEL_ID_LAST=vaghawan/tts-last
HF_TOKEN=hf_your_token_here

# Logging and Checkpointing (Advanced Training)
LOGGING_STEPS=10
SAVE_STEPS=500
EVAL_STEPS=500
SAVE_TOTAL_LIMIT=3

# Mixed Precision
MIXED_PRECISION=bf16

# Workflow Control
SKIP_PREPARE=false
PREPARE_ONLY=false
```

### Training Options

#### Option 1: Simple Training (Beginner Friendly)

Uses `sft_12hz.py` directly - perfect for quick experiments.

```bash
# Train with settings from .env
python train_using_sft.py

# Edit .env to change settings, then train again
nano .env
python train_using_sft.py

# Skip data preparation if already done
echo "SKIP_PREPARE=true" >> .env
python train_using_sft.py

# Only prepare data, don't train
echo "PREPARE_ONLY=true" >> .env
python train_using_sft.py
```

#### Option 2: Advanced Training (Production Ready)

Includes validation, metrics, WandB logging, and model checkpointing.

```bash
# Train with settings from .env (includes WandB by default)
python train_wandb_validation.py

# Disable WandB - edit .env:
echo "USE_WANDB=false" >> .env
python train_wandb_validation.py

# Enable model upload to Hugging Face - edit .env:
echo "UPLOAD_TO_HUB=true" >> .env
echo "HF_TOKEN=hf_your_token_here" >> .env
python train_wandb_validation.py

# Skip data preparation if already done
echo "SKIP_PREPARE=true" >> .env
python train_wandb_validation.py
```

#### Option 3: Dataset Preparation Only

```bash
# Prepare training data
python dataset_tool.py --dataset_name vaghawan/hausa-tts-22k --split train --output_jsonl ./data/train.jsonl

# Prepare validation data
python dataset_tool.py --dataset_name vaghawan/hausa-tts-22k --split validation --output_jsonl ./data/validation.jsonl

# Limit number of samples (for debugging)
python dataset_tool.py --dataset_name vaghawan/hausa-tts-22k --split train --max_samples 100

# Get dataset info
python dataset_tool.py --info ./data/train.jsonl
```

### Environment Variable Reference

#### Core Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DEVICE` | string | `cuda` | Device to use (cuda/cpu) |
| `INIT_MODEL_PATH` | string | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Initial model path |
| `OUTPUT_MODEL_PATH` | string | `./output` | Output model path |
| `OUTPUT_DIR` | string | `./output` | Output directory (advanced) |
| `TOKENIZER_PATH` | string | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | Tokenizer path |

#### Dataset Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATASET_NAME` | string | `vaghawan/hausa-tts-22k` | Hugging Face dataset name |
| `TRAIN_JSONL` | string | `./data/train.jsonl` | Training data path |
| `VALIDATION_JSONL` | string | `./data/validation.jsonl` | Validation data path |
| `MAX_TRAIN_SAMPLES` | int | None | Max training samples |
| `MAX_EVAL_SAMPLES` | int | None | Max validation samples |

#### Training Hyperparameters

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `BATCH_SIZE` | int | `2` | Batch size |
| `LR` | float | `2e-5` | Learning rate |
| `NUM_EPOCHS` | int | `3` | Number of epochs |
| `WEIGHT_DECAY` | float | `0.01` | Weight decay |
| `WARMUP_STEPS` | int | `100` | Warmup steps |
| `GRADIENT_ACCUMULATION_STEPS` | int | `4` | Gradient accumulation |
| `MAX_GRAD_NORM` | float | `1.0` | Max gradient norm |

#### Speaker Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SPEAKER_NAME` | string | `hausa_speaker` | Speaker name |
| `REF_AUDIO_PATH` | string | (default) | Reference audio path |
| `REF_TEXT` | string | (default) | Reference text |

#### WandB Configuration (Advanced)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `USE_WANDB` | bool | `true` | Enable WandB |
| `WANDB_PROJECT` | string | `qwen3-tts-hausa` | WandB project |
| `WANDB_RUN_NAME` | string | None | Custom run name |

#### Hugging Face Configuration (Advanced)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `UPLOAD_TO_HUB` | bool | `false` | Upload to Hub |
| `HUB_MODEL_ID_BEST` | string | `vaghawan/tts-best` | Best model repo |
| `HUB_MODEL_ID_LAST` | string | `vaghawan/tts-last` | Last model repo |
| `HF_TOKEN` | string | None | Hugging Face token |

#### Logging & Checkpointing (Advanced)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LOGGING_STEPS` | int | `10` | Logging frequency |
| `SAVE_STEPS` | int | `500` | Save frequency |
| `EVAL_STEPS` | int | `500` | Eval frequency |
| `SAVE_TOTAL_LIMIT` | int | `3` | Max checkpoints |

#### Workflow Control

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SKIP_PREPARE` | bool | `false` | Skip data prep |
| `PREPARE_ONLY` | bool | `false` | Only prep data |
| `MIXED_PRECISION` | string | `bf16` | Precision mode |

### Dataset Tool Features

The `dataset_tool.py` provides a unified interface for dataset operations:

- ✅ Load datasets from Hugging Face
- ✅ Prepare audio codes for training
- ✅ Save/load data in JSONL format
- ✅ Create PyTorch DataLoaders
- ✅ Get dataset statistics
- ✅ Command-line and Python API

### Python API Usage

```python
from dataset_tool import prepare_dataset, load_jsonl_dataset, create_dataloader, get_dataset_info

# Prepare dataset from Hugging Face
prepare_dataset(
    dataset_name="vaghawan/hausa-tts-22k",
    split="train",
    output_jsonl="./data/train.jsonl",
    model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    max_samples=1000
)

# Load dataset from JSONL
dataset = load_jsonl_dataset("./data/train.jsonl")
print(f"Loaded {len(dataset)} samples")

# Create PyTorch DataLoader
dataloader = create_dataloader("./data/train.jsonl", batch_size=4, shuffle=True)

# Get dataset information
info = get_dataset_info("./data/train.jsonl")
print(f"Languages: {info['languages']}")
print(f"Speakers: {info['speakers']}")
```

### Output Structure

#### Simple Training Output

```
output/
├── config.json
├── model.safetensors
└── tokenizer files...
```

#### Advanced Training Output

```
output/
├── best/                          # Best model (lowest validation loss)
│   ├── config.json
│   ├── model.safetensors
│   └── training_state.pt
├── last/                          # Last checkpoint
│   ├── config.json
│   ├── model.safetensors
│   └── training_state.pt
├── checkpoint-500/                # Intermediate checkpoints
├── checkpoint-1000/
├── epoch-1/
├── epoch-2/
└── epoch-3/
```

### Using the Trained Model

```python
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# Load the fine-tuned model
model = Qwen3TTSModel.from_pretrained("./output/best")

# Generate speech
text = "Hello, this is a test in Hausa."
ref_audio = "voices/english_voice/english_voice.wav"

output = model.generate(
    text=text,
    reference_audio=ref_audio,
    language="ha"
)

# Save output
import soundfile as sf
sf.write("output.wav", output.audio, output.sampling_rate)
```

### Training Comparison

| Feature | Simple Training | Advanced Training |
|---------|----------------|-------------------|
| Uses sft_12hz.py | ✅ | ❌ |
| Validation | ❌ | ✅ |
| WandB Logging | ❌ | ✅ |
| Checkpointing | Basic | Advanced |
| Hub Upload | ❌ | ✅ |
| Mixed Precision | Default | Configurable |
| Gradient Accumulation | ❌ | ✅ |
| Best Model Saving | ❌ | ✅ |
| Env Config | ✅ | ✅ |

### Use Cases

#### Use Simple Training When:
- Quick experimentation
- Small datasets
- No need for validation
- Simple workflow

#### Use Advanced Training When:
- Production training
- Large datasets
- Need validation metrics
- Want WandB monitoring
- Need checkpoint management
- Want to upload to Hub

### Typical Workflow

1. **Setup** (one-time):
   ```bash
   python setup_env.py          # Create .env file
   nano .env                     # Edit configuration
   python test_setup.py          # Verify setup
   ```

2. **Experiment** (development):
   ```bash
   # Edit .env to set small fast config
   echo "BATCH_SIZE=4" >> .env
   echo "NUM_EPOCHS=1" >> .env
   echo "MAX_TRAIN_SAMPLES=100" >> .env
   
   # Run simple training
   python train_using_sft.py
   ```

3. **Production** (final training):
   ```bash
   # Edit .env for production config
   echo "BATCH_SIZE=2" >> .env
   echo "NUM_EPOCHS=10" >> .env
   echo "USE_WANDB=true" >> .env
   echo "UPLOAD_TO_HUB=true" >> .env
   
   # Run advanced training
   python train_wandb_validation.py
   ```

## 🔧 Helper Scripts

### setup_env.py

Creates a `.env` file from `.env.training.example` template.

```bash
python setup_env.py
```

### test_setup.py

Verifies that your environment is correctly configured.

```bash
python test_setup.py
```

### dataset_tool.py

Unified interface for dataset operations.

```bash
# Prepare data
python dataset_tool.py --dataset_name vaghawan/hausa-tts-22k --split train

# Get info
python dataset_tool.py --info ./data/train.jsonl
```

This code follows the same license as Qwen3-TTS (Apache-2.0).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions, please open an issue on GitHub.