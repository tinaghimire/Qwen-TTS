# Qwen3-TTS Training Pipeline

Comprehensive training pipeline for Qwen3-TTS voice cloning with Hausa TTS fine-tuning. This project provides two training pipelines and a unified dataset tool.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Environment Configuration](#environment-configuration)
- [Project Structure](#project-structure)
- [Dataset Tool](#dataset-tool)
- [Training](#training)
  - [Unified Training Script (`train.py`)](#unified-training-script-trainpy)
  - [Single GPU vs Multi-GPU Training](#single-gpu-vs-multi-gpu-training)
  - [Legacy Training Script (`finetuning/sft_12hz.py`)](#legacy-training-script-finetuningsft_12hzpy)
  - [Training Scripts Comparison](#training-scripts-comparison)
  - [Which Script Should You Use?](#which-script-should-you-use)
  - [Example Workflows](#example-workflows)
  - [Migration from `sft_12hz.py` to `train.py`](#migration-from-sft_12hzpy-to-trainpy)
- [Layer Replacement and Addition](#layer-replacement-and-addition)
- [Training Workflows](#training-workflows)
- [Monitoring Training](#monitoring-training)
- [Using the Trained Model](#using-the-trained-model)
- [Architecture Overview](#architecture-overview)
- [Troubleshooting](#troubleshooting)
- [Training Analysis & Best Practices](#training-analysis--best-practices)

## Prerequisites

- **Python**: 3.10, 3.11, or 3.12
- **CUDA**: 13.0 (for GPU acceleration)
- **GPU**: NVIDIA GPU with compute capability 7.0+
- **RAM**: 32GB+ recommended (16GB minimal)
- **Disk**: 50GB+ free space for model storage

## Installation

```bash
# Install uv and dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Setup configuration
cp .env.training.example .env

# Edit settings
nano .env

# Verify installation
python test_setup.py

# Or verify using uv run
uv run test_setup.py
```

**Using UV:**

The project is configured to work seamlessly with `uv` for dependency management. You can run any script with `uv run <script>` to use the project's virtual environment:

```bash
# All training commands work with uv run
uv run train.py                           # Single GPU training
uv run accelerate launch --num_processes=4 train.py  # Multi-GPU training
uv run data_processing.py --mode info    # Test data processing

# Or use the convenience scripts
./uv_train.sh                             # UV-based single GPU
./uv_train_4gpu.sh                        # UV-based multi-GPU
```

## Quick Start

```bash
# Create .env from template
cp .env.training.example .env

# Configure your settings
nano .env

# Training Options:

# Option 1: Using uv run (recommended for UV-managed environments)
# Single GPU
uv run train.py
# Or use the script:
./uv_train.sh

# Multi-GPU (4 GPUs)
uv run accelerate launch --num_processes=4 train.py
# Or use the script:
./uv_train_4gpu.sh

# Option 2: Using Python directly
# Single GPU
python train.py
# Or use the script:
./launch_gpu.sh

# Multi-GPU (4 GPUs)
accelerate launch --num_processes=4 train.py
# Or use the script:
./launch_4gpu.sh
```

**Training Scripts:**
- `train.py` - Main training script (supports single/multi-GPU)
- `uv_train.sh` - UV-based single GPU launch script
- `uv_train_4gpu.sh` - UV-based multi-GPU launch script
- `launch_gpu.sh` - Python-based single GPU launch script
- `launch_4gpu.sh` - Python-based multi-GPU launch script

## Environment Configuration

All training settings are configured in `.env`. Here are the key variables:

### Device & Training

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | Device: cuda or cpu |
| `MIXED_PRECISION` | `bf16` | Precision: bf16, fp16, or no |

### Model & Data

| Variable | Default | Description |
|----------|---------|-------------|
| `INIT_MODEL_PATH` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base model path |
| `TOKENIZER_PATH` | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | Tokenizer path |
| `OUTPUT_MODEL_PATH` | `./output` | Output directory |
| `DATASET_NAME` | `vaghawan/hausa-tts-22k` | Hugging Face dataset |
| `TRAIN_SPLIT` | `train` | Training split |
| `VALIDATION_SPLIT` | `validation` | Validation split |

### Training Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE` | `2` | Batch size (16GB GPU: 1-2, 24GB: 2-4, 40GB+: 4-8) |
| `LR` | `2e-5` | Learning rate |
| `NUM_EPOCHS` | `3` | Number of epochs |
| `GRADIENT_ACCUMULATION_STEPS` | `4` | Gradient accumulation |
| `WEIGHT_DECAY` | `0.01` | Weight decay |
| `WARMUP_STEPS` | `100` | Warmup steps |
| `MAX_GRAD_NORM` | `1.0` | Gradient clipping |

### Reference Audio

| Variable | Default | Description |
|----------|---------|-------------|
| `REF_AUDIO_PATH` | `voices/speaker/reference.wav` | Reference audio (5-10s) |
| `SPEAKER_NAME` | `reference_speaker` | Speaker name |

### Dataset Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_TRAIN_SAMPLES` | - | Max training samples (empty = all) |
| `MAX_EVAL_SAMPLES` | - | Max evaluation samples (empty = all) |

### Data Loading

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_WORKERS` | `0` | Number of data loading workers (0 = main process) |

### Logging & Checkpointing

| Variable | Default | Description |
|----------|---------|-------------|
| `LOGGING_STEPS` | `10` | Log frequency |
| `SAVE_STEPS` | `500` | Checkpoint save frequency |
| `EVAL_STEPS` | `500` | Evaluation frequency |
| `SAVE_TOTAL_LIMIT` | `3` | Max checkpoints |

### WandB

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_WANDB` | `true` | Enable WandB |
| `WANDB_PROJECT` | `qwen3-tts-hausa` | Project name |
| `WANDB_RUN_NAME` | - | Run name (empty = auto) |

### Hugging Face Upload

| Variable | Default | Description |
|----------|---------|-------------|
| `UPLOAD_TO_HUB` | `false` | Upload to Hub |
| `HUB_MODEL_ID_BEST` | `your-username/tts-best` | Best model repo |
| `HUB_MODEL_ID_LAST` | `your-username/tts-last` | Last checkpoint repo |
| `HF_TOKEN` | - | Hugging Face token |

### Workflow Control

| Variable | Default | Description |
|----------|---------|-------------|
| `PREPARE_ONLY` | `false` | Only test DataLoader, don't train |

### Example Configurations

```bash
# Minimal CPU setup
DEVICE=cpu
BATCH_SIZE=1
MAX_TRAIN_SAMPLES=100
NUM_EPOCHS=1
USE_WANDB=false

# Medium GPU training
DEVICE=cuda
BATCH_SIZE=4
MAX_TRAIN_SAMPLES=1000
NUM_EPOCHS=3
GRADIENT_ACCUMULATION_STEPS=4

# Full production with upload
DEVICE=cuda
BATCH_SIZE=8
NUM_EPOCHS=5
GRADIENT_ACCUMULATION_STEPS=2
USE_WANDB=true
UPLOAD_TO_HUB=true
HF_TOKEN=hf_******
```

### Verification Script

Before starting training, verify your setup:

```bash
python test_setup.py
```

This script checks:
- Python version compatibility (3.10-3.12)
- Required package installations
- CUDA/GPU availability
- Qwen3-TTS import
- .env file configuration

## Project Structure

```
Qwen3-TTS-finetuning/
├── train.py                      # Unified training script
├── data_processing.py            # Unified data processing module
├── test_setup.py                 # Setup verification script
├── .env.training.example         # Environment template
├── .env                          # Your configuration (copy from template)
├── voices/                       # Reference audio files
│   └── english_voice/
│       └── english_voice.wav
├── data/                         # Prepared datasets (created automatically)
│   ├── train.jsonl
│   └── validation.jsonl
└── Qwen3-TTS/                    # Core Qwen3-TTS library
    └── finetuning/
        ├── sft_12hz.py            # Base training script
        ├── dataset.py             # Dataset class for training
        └── layer_utils.py         # Layer replacement utilities
```

## Data Processing

The `data_processing.py` module provides two approaches for data handling:

### 1. JSONLDataPreparer - Prepare data from HuggingFace to JSONL

Best for: Large datasets, repeated training runs, offline training

```bash
# Prepare all splits to JSONL
python data_processing.py --mode prepare

# Or with uv run:
uv run data_processing.py --mode prepare

# Prepare specific split
python data_processing.py --mode prepare --split train

# Limit samples
python data_processing.py --mode prepare --max_samples 1000
```

**Usage in Python:**

```python
from data_processing import JSONLDataPreparer

# Create preparer
preparer = JSONLDataPreparer(
    dataset_name="vaghawan/hausa-tts-22k",
    output_dir="./data",
    tokenizer_path="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device="cuda",
)

# Prepare all splits
output_files = preparer.prepare_all_splits()

# Prepare specific split
train_file = preparer.prepare_split("train", max_samples=1000)
```

### 2. HFDirectDataLoader - Load directly from HuggingFace

Best for: Large datasets (>100k samples), multi-GPU training, online training

**Multi-GPU Support:**

HFDirectDataLoader automatically detects and supports multi-GPU training:
- Auto-detects device for each distributed process
- Loads tokenizer on appropriate GPU for each process
- Streams data to avoid memory issues with large datasets
- Data sharding handled automatically by Accelerate

```bash
# Test direct loading (auto-detects GPU setup)
python data_processing.py --mode direct --batch_size 4

# Or with uv run:
uv run data_processing.py --mode direct --batch_size 4

# Test with limited samples
python data_processing.py --mode direct --max_samples 100

# Test with multi-GPU (each process uses its own tokenizer)
accelerate launch --num_processes=4 data_processing.py --mode direct --max_samples 100

# Or with uv run:
uv run accelerate launch --num_processes=4 data_processing.py --mode direct --max_samples 100
```

**Usage in Python:**

```python
from data_processing import HFDirectDataLoader, get_dataloader

# Single GPU training - device auto-detected
dataset = HFDirectDataLoader(
    dataset_name="vaghawan/hausa-tts-22k",
    split="train",
    tokenizer_path="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    max_samples=1000,
)

# Multi-GPU training - create in each process, device auto-detected
from accelerate import Accelerator
accelerator = Accelerator()

# Device automatically detected for this process
dataset = HFDirectDataLoader(
    dataset_name="vaghawan/hausa-tts-22k",
    split="train",
    tokenizer_path="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    max_samples=600000,  # Large dataset - streams from HF
)

dataloader = get_dataloader(
    dataset_name="vaghawan/hausa-tts-22k",
    split="train",
    batch_size=8,  # Per GPU
    max_samples=600000,
)

# Prepare for multi-GPU training
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
```

Auto-Detection Features:
- ✅ Automatic device detection (`get_device_for_current_process()`)
- ✅ Automatic num_workers based on GPU count (`get_num_workers_for_dataloader()`)
- ✅ Pin memory enabled by default for GPU training
- ✅ Compatible with Accelerate's DistributedSampler

### Get Dataset Information

```bash
# Get info about a dataset
python data_processing.py --mode info

# Get info for specific split
python data_processing.py --mode info --split validation
```

### Data Format

Both approaches return the same data format:

```python
{
    "audio": np.ndarray,           # Audio waveform array (24kHz)
    "text": str,                   # Transcription text
    "audio_codes": List[List[int]], # Encoded audio codes (16 channels)
    "sr": int,                     # Sampling rate (24000)
    "language": str,               # Language code
    "ref_audio": str,              # Reference audio path
    "ref_text": str,               # Reference text for ICL
}
```

### Choosing the Right Approach

| Approach | Best For | Pros | Cons | Multi-GPU |
|----------|----------|------|------|-----------|
| **JSONLDataPreparer** | Medium datasets, repeated training | Fast loading, offline training, reproducible | Requires storage space, initial prep time | ❌ Loads all into RAM |
| **HFDirectDataLoader** | Large datasets, multi-GPU, online training | Minimal RAM, auto-detects GPU, streams data | Slower first iteration, requires HF access | ✅ Full support, auto-detection |

**Multi-GPU Recommendations:**

| Dataset Size | Recommended Mode | Configuration |
|--------------|------------------|---------------|
| < 10k samples | JSONL or Direct | `DATA_MODE=jsonl` or `direct` |
| 10k - 100k samples | JSONL or Direct | `DATA_MODE=jsonl` or `direct` |
| > 100k samples | ✅ **Direct (Required)** | `DATA_MODE=direct` - JSONL will cause OOM |
| 600k samples | ✅ **Direct (Required)** | `DATA_MODE=direct` + multi-GPU |

## Training

### Unified Training Script (`train.py`)

The main training script that provides a complete training pipeline with all features integrated.

**Main Workflow:**
1. Load/prepare dataset
2. Load models
3. Login to WandB
4. Fine-tune with parameters

**Features:**
- **Multiple Data Modes**: JSONL, direct HuggingFace loading, or skip data preparation
- **25Hz Tokenizer**: Uses Qwen3-TTS-Tokenizer-25Hz for high-quality audio
- **Layer Replacement**: Replace and add layers for better fine-tuning
- **Training Logging**: Logs training loss to `training_log.jsonl`
- **Validation Logging**: Logs validation loss and metrics to `validation_log.jsonl`
- **WandB Tracking**: Comprehensive metrics tracking
- **Checkpointing**: Saves best and last models
- **HuggingFace Upload**: Optional upload to HuggingFace Hub

**Usage:**
```bash
# Train with default settings
python train.py

# Train with specific data mode
DATA_MODE=jsonl python train.py
DATA_MODE=direct python train.py
DATA_MODE=none python train.py

# Train with custom layer replacement
REPLACE_LAST_N_LAYERS=4 ADD_NEW_LAYERS=8 python train.py
```

**Configuration:**

Create a `.env` file with the following variables:

```bash
# Data Configuration
DATA_MODE=jsonl  # Options: jsonl, direct, none
DATASET_NAME=vaghawan/hausa-tts-22k
TRAIN_JSONL=./data/train.jsonl
VALIDATION_JSONL=./data/validation.jsonl

# Model Configuration
INIT_MODEL_PATH=Qwen/Qwen3-TTS-25Hz-1.7B-Base
TOKENIZER_PATH=Qwen/Qwen3-TTS-Tokenizer-25Hz
OUTPUT_DIR=./output
SPEAKER_NAME=reference_speaker

# Training Hyperparameters
BATCH_SIZE=2
LEARNING_RATE=2e-5
NUM_EPOCHS=3
GRADIENT_ACCUMULATION_STEPS=8
WEIGHT_DECAY=0.01
WARMUP_STEPS=200
MAX_GRAD_NORM=1.0
SUB_TALKER_LOSS_WEIGHT=0.3

# Layer Replacement Configuration
REPLACE_LAST_N_LAYERS=2
ADD_NEW_LAYERS=4
FREEZE_ORIGINAL_LAYERS=true

# Logging and Checkpointing
LOGGING_STEPS=10
SAVE_STEPS=500
EVAL_STEPS=500
SAVE_TOTAL_LIMIT=3

# WandB Configuration
USE_WANDB=true
WANDB_PROJECT=qwen3-tts-training
WANDB_RUN_NAME=my_training_run
WANDB_ENTITY=your_wandb_entity

# HuggingFace Configuration
HF_TOKEN=your_huggingface_token_here
UPLOAD_TO_HF=false
HF_BEST_MODEL_REPO=your-username/tts-best
HF_LAST_MODEL_REPO=your-username/tts-last

# Device and Precision
DEVICE=cuda
MIXED_PRECISION=bf16

# Data Limits
MAX_TRAIN_SAMPLES=
MAX_EVAL_SAMPLES=

# Reference Audio
REF_AUDIO_PATH=
```

**Data Modes:**

| Mode | Description | Best For |
|------|-------------|----------|
| `jsonl` | Uses pre-prepared JSONL files | Large datasets, repeated training |
| `direct` | Loads directly from HuggingFace | Quick experiments, small datasets |
| `none` | Skips data preparation | Testing, debugging |

**Output Structure:**

```
output/
├── best/                          # Best model (lowest validation loss)
│   ├── model.safetensors
│   ├── config.json
│   ├── training_state.json
│   └── processor files...
├── last/                          # Last checkpoint
│   ├── model.safetensors
│   ├── config.json
│   ├── training_state.json
│   └── processor files...
├── checkpoint-500/                # Intermediate checkpoints
├── checkpoint-1000/
├── training_log.jsonl             # Training loss logs
└── validation_log.jsonl           # Validation loss and metrics logs
```

**Log Files:**

**training_log.jsonl** - Training metrics:
```json
{
  "step": 100,
  "epoch": 0,
  "loss": 2.3456,
  "learning_rate": 1.8e-5,
  "timestamp": "2026-02-24T10:30:00"
}
```

**validation_log.jsonl** - Validation metrics:
```json
{
  "step": 500,
  "epoch": 0,
  "loss": 2.1234,
  "metrics": {
    "speaker_embedding_similarity": 0.95
  },
  "timestamp": "2026-02-24T10:35:00"
}
```

**Layer Replacement:**

The script supports layer replacement to improve fine-tuning:

1. **Original Layers**: The base model has a certain number of layers (e.g., 32 layers)
2. **Replacement**: The last N layers are replaced with newly initialized layers
3. **Addition**: M additional layers are added after the replacement
4. **Freezing**: Original (non-replaced) layers can be frozen to prevent modification

**Example Configuration:**

If the original model has 32 layers:
- `REPLACE_LAST_N_LAYERS=2`: Replace layers 31-32
- `ADD_NEW_LAYERS=4`: Add 4 new layers (33-36)
- `FREEZE_ORIGINAL_LAYERS=true`: Freeze layers 1-30

**Result**: 36 total layers (30 frozen + 6 trainable)

**Benefits:**
- **Prevents Catastrophic Forgetting**: Freezing original layers preserves pre-trained knowledge
- **Adapts to New Data**: New layers learn speaker-specific characteristics
- **Flexible Architecture**: Adjust the number of layers based on your needs

**Benefits:**
- **Prevents Catastrophic Forgetting**: Freezing original layers preserves pre-trained knowledge
- **Adapts to New Data**: New layers learn speaker-specific characteristics
- **Flexible Architecture**: Adjust the number of layers based on your needs

**Data Requirements:**

The script expects JSONL files in the `data/` directory:

- `data/train.jsonl`: Training data (required)
- `data/validation.jsonl`: Validation data (optional)

Each line in the JSONL file should contain:
```json
{
  "text": "Your text here",
  "audio": "path/to/audio.wav"
}
```

**How it works:**
1. Checks for local JSONL files in `data/` directory
2. Calls `finetuning/sft_12hz.py` with layer replacement parameters
3. Trains model with frozen original layers and trainable new layers
4. Saves model checkpoints to `OUTPUT_MODEL_PATH`
5. (Optional) Uploads models to HuggingFace Hub

**Output:**

```
output/
├── checkpoint-epoch-0/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   ├── preprocessor_config.json
│   ├── speech_tokenizer/
│   ├── speaker_encoder/
│   └── README.md
├── checkpoint-epoch-1/
└── checkpoint-epoch-2/
```

**Best Practices:**
1. **Start Small**: Begin with `REPLACE_LAST_N_LAYERS=2` and `ADD_NEW_LAYERS=4`
2. **Monitor Training**: Check loss curves to ensure convergence
3. **Validate**: Use validation data to prevent overfitting
4. **Save Checkpoints**: Keep multiple checkpoints for comparison
5. **Test Loading**: Verify checkpoints can be loaded after training

### Legacy Training Script (`finetuning/sft_12hz.py`)

A simpler training script for quick experiments and debugging. This script uses command-line arguments instead of environment variables.

**Features:**
- **Simple Configuration**: Command-line arguments for all parameters
- **12Hz Model**: Uses Qwen3-TTS-12Hz-1.7B-Base by default
- **Layer Replacement**: Supports layer replacement and addition
- **Basic Training**: Simple training loop without validation
- **Per-Episode Checkpoints**: Saves checkpoints after each epoch
- **No Logging**: Console output only, no log files

**Usage:**
```bash
# Basic training with defaults
python finetuning/sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl data/train.jsonl \
    --output_model_path output \
    --batch_size 2 \
    --lr 2e-5 \
    --num_epochs 3 \
    --speaker_name my_speaker

# Training with layer replacement
python finetuning/sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl data/train.jsonl \
    --output_model_path output \
    --batch_size 2 \
    --lr 2e-5 \
    --num_epochs 3 \
    --speaker_name my_speaker \
    --replace_last_n_layers 3 \
    --add_new_layers 5 \
    --freeze_original_layers True

# Training with frozen speaker encoder
python finetuning/sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl data/train.jsonl \
    --output_model_path output \
    --batch_size 2 \
    --lr 2e-5 \
    --num_epochs 3 \
    --speaker_name my_speaker \
    --freeze_speaker_encoder True
```

**Command-Line Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--init_model_path` | str | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base model path |
| `--output_model_path` | str | `output` | Output directory |
| `--train_jsonl` | str | (required) | Path to training JSONL file |
| `--batch_size` | int | `2` | Batch size |
| `--lr` | float | `2e-5` | Learning rate |
| `--num_epochs` | int | `3` | Number of epochs |
| `--speaker_name` | str | `speaker_test` | Speaker name |
| `--replace_last_n_layers` | int | `2` | Number of last layers to replace |
| `--add_new_layers` | int | `4` | Number of additional layers to add |
| `--freeze_original_layers` | bool | `True` | Whether to freeze original layers |
| `--freeze_speaker_encoder` | bool | `False` | Whether to freeze speaker encoder |

**Output Structure:**
```
output/
├── checkpoint-epoch-0/
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt
│   ├── preprocessor_config.json
│   ├── speech_tokenizer/
│   ├── speaker_encoder/
│   └── README.md
├── checkpoint-epoch-1/
└── checkpoint-epoch-2/
```

**When to Use:**
- Quick debugging and testing
- Simple training without validation
- Experiments with 12Hz model specifically
- Learning the training pipeline

### Training Scripts Comparison

| Feature | `train.py` | `finetuning/sft_12hz.py` |
|---------|-----------|---------------------------|
| **Configuration** | `.env` file | Command-line arguments |
| **Data Modes** | 3 (jsonl, direct, none) | JSONL only |
| **Validation** | ✅ Yes | ❌ No |
| **Logging** | ✅ JSONL + WandB | ❌ Console only |
| **Checkpointing** | Best + Last + Periodic | Per epoch only |
| **Learning Rate Scheduler** | Cosine with warmup | None (default AdamW) |
| **Gradient Accumulation** | Configurable | Fixed at 8 |
| **Mixed Precision** | Configurable (bf16/fp16/no) | Fixed at bf16 |
| **WandB Tracking** | ✅ Yes | ❌ No |
| **HuggingFace Upload** | ✅ Yes | ❌ No |
| **Training Metrics** | Loss, LR, Perplexity, Speaker Consistency | Loss only |
| **Default Model** | 12Hz or 25Hz (configurable) | 12Hz only |
| **Speaker Encoder Freeze** | Configurable | Configurable |
| **Layer Replacement** | ✅ Yes | ✅ Yes |
| **Multi-GPU Support** | ✅ Yes (via Accelerate) | ❌ No |
| **Recommended For** | Production training | Quick experiments |

### Single GPU vs Multi-GPU Training

The `train.py` script supports both single GPU and multi-GPU distributed training using Hugging Face Accelerate.

#### Overview

| Aspect | Single GPU | Multi-GPU (4x GPUs) |
|--------|-----------|-------------------|
| **Setup** | `python train.py` | `accelerate launch --num_processes=4 train.py` |
| **Batch Size** | Per GPU batch size | Per GPU batch size (effective: × num_gpus) |
| **Learning Rate** | Base LR | Scaled LR (recommended: × #GPUs/4) |
| **Memory** | Single GPU memory | Distributed across GPUs |
| **Training Speed** | Baseline | ~3.5-4x faster (near-linear scaling) |
| **Data Loading** | Sequential | Distributed (automatic sharding) |

#### Single GPU Training

For single GPU training, you can use any of these methods:

**Method 1: Using uv run (Recommended for UV-managed environments)**

```bash
# Run with uv
uv run train.py

# Or use the convenience script
./uv_train.sh

# With specific GPU
CUDA_VISIBLE_DEVICES=0 uv run train.py
```

**Method 2: Using Python directly**

```bash
python train.py

# Or use the convenience script
./launch_gpu.sh
```

**Recommended configuration for single 16GB GPU:**

```bash
# .env configuration
BATCH_SIZE=4
LEARNING_RATE=3e-4
NUM_EPOCHS=3
GRADIENT_ACCUMULATION_STEPS=1
WARMUP_STEPS=500
```

**Data mode recommendation:**
- Small datasets (< 10k samples): Use `DATA_MODE=jsonl` (load into RAM)
- Medium datasets (10k-100k): Use `DATA_MODE=jsonl` or `direct`
- Large datasets (> 100k): Use `DATA_MODE=direct` (streaming)

#### Multi-GPU Training

For multi-GPU training, you can use any of these methods:

**Method 1: Using uv run with accelerate (Recommended for UV-managed environments)**

```bash
# Basic multi-GPU launch (4 GPUs)
uv run accelerate launch --num_processes=4 train.py

# Or use the convenience script
./uv_train_4gpu.sh

# With specific GPU selection
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch --num_processes=4 train.py

# With all environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
uv run accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision="bf16" \
    --dynamo_backend="no" \
    train.py
```

**Method 2: Using Python directly with accelerate**

```bash
# Basic multi-GPU launch (4 GPUs)
accelerate launch --num_processes=4 train.py

# With specific GPU selection
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 train.py

# With all environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --num_processes=4 \
    --num_machines=1 \
    --mixed_precision="bf16" \
    --dynamo_backend="no" \
    train.py
```

**Method 3: Using Python convenience script**

```bash
# Use the existing launch script
./launch_4gpu.sh
```

**Recommended configuration for 4x 16GB GPUs:**

```bash
# .env configuration
DATA_MODE=direct                    # Required for large datasets
BATCH_SIZE=8                        # Per GPU (32 total effective batch size)
LEARNING_RATE=1e-3                  # Scaled for multi-GPU
NUM_EPOCHS=3
GRADIENT_ACCUMULATION_STEPS=1       # No accumulation needed
WARMUP_STEPS=1000                   # Increased warmup
MIXED_PRECISION=bf16                # Critical for efficiency
```

**Large dataset configuration (600k training / 30k validation):**

```bash
# .env file
DATA_MODE=direct
DATASET_NAME=your_dataset_name      # Must be on HuggingFace
MAX_TRAIN_SAMPLES=600000
MAX_EVAL_SAMPLES=30000
BATCH_SIZE=8
LEARNING_RATE=1e-3
NUM_EPOCHS=3
GRADIENT_ACCUMULATION_STEPS=1
WARMUP_STEPS=1000
LOGGING_STEPS=100
SAVE_STEPS=2500                     # Less frequent for long training
EVAL_STEPS=2500
SAVE_TOTAL_LIMIT=5
MIXED_PRECISION=bf16
```

#### Multi-GPU Launch Script

You can use any of these convenient scripts for multi-GPU training:

**Option 1: UV-based launch script (Recommended for UV-managed environments)**

The project includes `uv_train_4gpu.sh` which uses `uv run` internally:

```bash
# Make executable (first time only)
chmod +x uv_train_4gpu.sh

# Run training
./uv_train_4gpu.sh
```

The UV-based script automatically:
- Checks if uv is installed
- Verifies accelerate is available
- Shows GPU status
- Launches training with `uv run accelerate launch`

**Option 2: Python-based launch script**

For environments not using UV, use `launch_4gpu.sh`:

```bash
# Make executable (first time only)
chmod +x launch_4gpu.sh

# Run training
./launch_4gpu.sh
```

The Python-based script:
- Checks accelerate installation
- Shows GPU status
- Launches training with `accelerate launch`

**Option 3: Manual launch script**

Create a custom `launch_4gpu_manual.sh` script:

```bash
#!/bin/bash

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Environment variables for distributed training
export NCCL_DEBUG=INFO              # NCCL debugging (optional)
export OMP_NUM_THREADS=4            # Thread count per process

# Choose launch method:
# UV-managed:
uv run accelerate launch \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision="bf16" \
    --dynamo_backend="no" \
    train.py

# Or Python (uncomment the line below):
# accelerate launch \
#     --num_processes=4 \
#     --num_machines=1 \
#     --mixed_precision="bf16" \
#     --dynamo_backend="no" \
#     train.py
```

#### Data Loading for Large Datasets

**Critical**: For datasets > 100k samples, you MUST use `DATA_MODE=direct`:

```bash
# ❌ DON'T use jsonl mode for large datasets (will cause OOM)
DATA_MODE=jsonl
MAX_TRAIN_SAMPLES=600000           # Will crash - loads all into RAM

# ✅ DO use direct mode for streaming
DATA_MODE=direct
DATASET_NAME=your-dataset          # Streams from HuggingFace
MAX_TRAIN_SAMPLES=600000           # Efficient streaming
```

If your dataset is not on HuggingFace, you can:

1. **Push to HuggingFace** (recommended):
```bash
huggingface-cli login
# Use datasets library to upload your data
```

2. **Use WebDataset format** (advanced):
   - Convert to sharded `.tar` files
   - Use WebDataset library for streaming

3. **Modify the dataloader** to stream from your custom source

#### Learning Rate Scaling

When scaling to multiple GPUs, adjust the learning rate using the **linear scaling rule**:

**Formula**: `LR_multi_gpu = LR_single_gpu × (batch_size_multi_gpu / batch_size_single_gpu)`

**Example**:
- Single GPU: batch_size=4, LR=3e-4
- 4 GPUs: batch_size=32 (4×8), LR=3e-4 × (32/4) = 2.4e-3

**Conservative approach (recommended)**: Scale by 2-3x instead of full 8x:
- Single GPU: 3e-4
- 4 GPUs: 1e-3 (instead of 2.4e-3)

#### Training Time Estimates

| Configuration | Dataset Size | Steps per Epoch | Total Steps | Training Time |
|--------------|--------------|-----------------|-------------|---------------|
| 1× GPU, batch=4 | 10k | 2,500 | 7,500 | 4-6 hours |
| 1× GPU, batch=4 | 70k | 17,500 | 52,500 | 28-40 hours |
| 4× GPU, batch=8 | 70k | 2,188 | 6,563 | 8-12 hours |
| 4× GPU, batch=8 | 600k | 18,750 | 56,250 | 32-47 hours |

#### Memory Requirements

**Per GPU Memory:**

| Component | Single GPU | Multi-GPU (per GPU) |
|-----------|-----------|-------------------|
| Model weights | ~6 GB | ~6 GB |
| Batch (batch=4) | ~4 GB | ~8 GB (for batch=8) |
| Activations | ~2 GB | ~2 GB |
| Gradients | ~1 GB | ~1 GB |
| Overhead | ~1 GB | ~1 GB |
| **Total** | ~14 GB | ~18 GB ⚠️ |

For 16GB GPUs with batch_size=8, you may be close to the limit. If you encounter OOM errors:

```bash
# Option 1: Reduce batch size
BATCH_SIZE=4                      # Reduced batch
GRADIENT_ACCUMULATION_STEPS=2    # Maintain effective batch size

# Option 2: Use gradient checkpointing (if supported)
# Add to model configuration
```

#### Monitoring Multi-GPU Training

**Watch GPU utilization:**

```bash
# Terminal 1: Monitor all GPUs
watch -n 1 nvidia-smi

# Terminal 2: Monitor training logs
tail -f output/training_log.jsonl

# Terminal 3: Monitor WandB
# Visit wandb.ai for live metrics
```

**Expected training behavior:**

- **GPU Utilization**: All 4 GPUs should show 90-100% during training
- **Memory Usage**: Consistent ~14-18 GB per GPU
- **Speed**: ~3.5-4x faster than single GPU
- **Loss Pattern**: Similar to single GPU but reaches target faster

#### Troubleshooting Multi-GPU Issues

**Issue: Only 1 GPU is being used**

```bash
# Check available GPUs
nvidia-smi

# Verify accelerate configuration
accelerate config

# Manually specify GPU count
accelerate launch --num_processes=4 train.py
```

**Issue: NCCL communication errors**

```bash
# Add to launch script
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0

# Or use gloo instead of NCCL (slower but more compatible)
accelerate launch --num_processes=4 --multi_gpu backend=gloo train.py
```

**Issue: Out of Memory**

```bash
# Reduce batch size
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2

# Or enable gradient checkpointing in model config
```

**Issue: Data loading bottleneck**

```bash
# Increase num_workers
NUM_WORKERS=8  # More workers for multi-GPU

# Or use direct mode with streaming
DATA_MODE=direct
```

#### Performance Optimization Tips

1. **Enable mixed precision**: Always use `MIXED_PRECISION=bf16` for multi-GPU
2. **Pin memory**: DataLoader uses `pin_memory=True` by default
3. **Increase num_workers**: 2 workers per GPU (8 workers for 4 GPUs)
4. **Use direct mode**: For large datasets to avoid loading everything into RAM
5. **Adjust logging frequency**: Less frequent logs for long training runs
6. **Optimize checkpoint frequency**: `SAVE_STEPS=2500` for 50k+ steps

#### Pre-flight Checklist for Multi-GPU Training

Before starting multi-GPU training:

- [ ] All GPUs visible: `nvidia-smi -L` shows all GPUs
- [ ] Accelerate installed: `uv run -c "import accelerate; print(accelerate.__version__)"` or `accelerate --version`
- [ ] UV installed (if using UV-based scripts): `uv --version`
- [ ] Dataset accessible: Test with `uv run data_processing.py --mode info`
- [ ] Sufficient disk space: 50-70 GB free
- [ ] Memory available: Check per-GPU memory with nvidia-smi
- [ ] NCCL working: Test quick multi-GPU run (10 steps)
- [ ] WandB configured: If using tracking
- [ ] HuggingFace token ready: If uploading

### Which Script Should You Use?

#### Use `train.py` for:
- ✅ **Production training** - Complete monitoring and evaluation
- ✅ **Large datasets** - Efficient data loading with multiple modes
- ✅ **Model selection** - Automatic best model saving based on validation
- ✅ **Experiment tracking** - WandB integration for metrics visualization
- ✅ **Model sharing** - Easy upload to HuggingFace Hub
- ✅ **Reproducible runs** - Configuration saved in `.env` file
- ✅ **25Hz models** - Support for higher quality audio generation

#### Use `finetuning/sft_12hz.py` for:
- ✅ **Quick debugging** - Simple setup, fast iteration
- ✅ **Learning the pipeline** - Easier to understand the code
- ✅ **Small experiments** - No validation overhead
- ✅ **12Hz models** - Specifically designed for 12Hz models
- ✅ **Testing layer replacement** - Quick verification of layer modifications

### Example Workflows

#### Workflow 1: Production Training with `train.py`

```bash
# Step 1: Create .env configuration
cat > .env << EOF
DATA_MODE=jsonl
DATASET_NAME=vaghawan/hausa-tts-22k
TRAIN_JSONL=./data/train.jsonl
VALIDATION_JSONL=./data/validation.jsonl
INIT_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base
TOKENIZER_PATH=Qwen/Qwen3-TTS-Tokenizer-12Hz
OUTPUT_DIR=./output
SPEAKER_NAME=hausa_speaker
BATCH_SIZE=1
LEARNING_RATE=3e-5
NUM_EPOCHS=30
GRADIENT_ACCUMULATION_STEPS=16
USE_WANDB=true
WANDB_PROJECT=qwen3-tts-hausa
EOF

# Step 2: Run training
python train.py

# Step 3: Monitor training
tail -f output/training_log.jsonl
tail -f output/validation_log.jsonl

# Step 4: Use best model
python -c "
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
model = Qwen3TTSModel.from_pretrained('./output/best')
"
```

#### Workflow 2: Quick Experiment with `sft_12hz.py`

```bash
# Step 1: Prepare data (if needed)
python data_processing.py --mode prepare --max_samples 100

# Step 2: Run quick training
python finetuning/sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl data/train.jsonl \
    --output_model_path output_quick \
    --batch_size 2 \
    --lr 2e-5 \
    --num_epochs 1 \
    --speaker_name test_speaker

# Step 3: Test the model
python -c "
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
model = Qwen3TTSModel.from_pretrained('./output_quick/checkpoint-epoch-0')
"
```

#### Workflow 3: Layer Replacement Experiment

```bash
# Using train.py
REPLACE_LAST_N_LAYERS=4 ADD_NEW_LAYERS=8 python train.py

# Using sft_12hz.py
python finetuning/sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl data/train.jsonl \
    --output_model_path output_layers \
    --batch_size 2 \
    --lr 2e-5 \
    --num_epochs 3 \
    --speaker_name test_speaker \
    --replace_last_n_layers 4 \
    --add_new_layers 8 \
    --freeze_original_layers True
```

### Migration from `sft_12hz.py` to `train.py`

If you're currently using `sft_12hz.py` and want to migrate to `train.py`:

```bash
# Old command (sft_12hz.py)
python finetuning/sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl data/train.jsonl \
    --output_model_path output \
    --batch_size 2 \
    --lr 2e-5 \
    --num_epochs 3 \
    --speaker_name my_speaker \
    --replace_last_n_layers 2 \
    --add_new_layers 4 \
    --freeze_original_layers True

# New equivalent (train.py with .env)
cat > .env << EOF
DATA_MODE=jsonl
TRAIN_JSONL=./data/train.jsonl
VALIDATION_JSONL=./data/validation.jsonl
INIT_MODEL_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base
TOKENIZER_PATH=Qwen/Qwen3-TTS-Tokenizer-12Hz
OUTPUT_DIR=./output
SPEAKER_NAME=my_speaker
BATCH_SIZE=2
LEARNING_RATE=2e-5
NUM_EPOCHS=3
GRADIENT_ACCUMULATION_STEPS=8
REPLACE_LAST_N_LAYERS=2
ADD_NEW_LAYERS=4
FREEZE_ORIGINAL_LAYERS=true
EOF

python train.py
```

**Key Differences to Note:**
1. `train.py` uses `LEARNING_RATE` instead of `--lr`
2. `train.py` requires `TOKENIZER_PATH` to be specified
3. `train.py` automatically creates validation logs if `VALIDATION_JSONL` is provided
4. `train.py` saves best model based on validation loss, not just last epoch

## Layer Replacement and Addition

The layer replacement feature allows you to replace the last N layers of the Qwen3TTSTalker model with newly initialized layers and add M additional layers for better fine-tuning.

### Overview

This feature enables:
- Replace the last N layers with newly initialized layers
- Add M additional layers after the replacement
- Freeze the original layers to preserve pre-trained knowledge
- Fine-tune only the new layers for better speaker adaptation

### Architecture Change

```
Original Model (20 layers):
├── Layers 1-18 (pre-trained)
├── Layer 19 (pre-trained)
└── Layer 20 (pre-trained)

Modified Model (24 layers):
├── Layers 1-18 (frozen, pre-trained)
├── Layer 19 (newly initialized)
├── Layer 20 (newly initialized)
├── Layer 21 (newly initialized)
├── Layer 22 (newly initialized)
├── Layer 23 (newly initialized)
└── Layer 24 (newly initialized)
```

### Usage

#### Basic Usage (Default: Replace 2, Add 4)

```bash
python finetuning/sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl data/train.jsonl \
    --output_model_path output \
    --batch_size 2 \
    --lr 2e-5 \
    --num_epochs 3 \
    --speaker_name my_speaker
```

This will:
- Replace the last 2 layers (19-20) with newly initialized layers
- Add 4 additional layers (21-24)
- Freeze layers 1-18
- Train only on layers 19-24

#### Custom Configuration

```bash
python finetuning/sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl data/train.jsonl \
    --output_model_path output \
    --batch_size 2 \
    --lr 2e-5 \
    --num_epochs 3 \
    --speaker_name my_speaker \
    --replace_last_n_layers 3 \
    --add_new_layers 5 \
    --freeze_original_layers True
```

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--replace_last_n_layers` | int | 2 | Number of last layers to replace |
| `--add_new_layers` | int | 4 | Number of additional layers to add |
| `--freeze_original_layers` | bool | True | Whether to freeze original layers |

### Training Output

When you run the training script, you'll see detailed output about the layer replacement:

```
============================================================
Layer Replacement and Addition
============================================================
Original number of layers: 20
Replacing last 2 layers
Adding 4 new layers
Total new layers: 24

Keeping layers 1-18 (frozen: True)
  - Layer 1: Frozen
  - Layer 2: Frozen
  ...
  - Layer 18: Frozen
  - Layer 19: Replacement (freshly initialized)
  - Layer 20: Replacement (freshly initialized)
  - Layer 21: Additional (freshly initialized)
  - Layer 22: Additional (freshly initialized)
  - Layer 23: Additional (freshly initialized)
  - Layer 24: Additional (freshly initialized)

============================================================
Layer replacement complete!
New total number of layers: 24
Configuration updated: num_hidden_layers = 24
============================================================

============================================================
Model Summary
============================================================
Total layers: 24
Hidden size: 1024
Intermediate size: 2048
Attention heads: 16

Parameter counts:
  Trainable: 123,456,789 (15.23%)
  Frozen: 687,654,321 (84.77%)
  Total: 811,111,110

Layer status:
  Trainable layers: 6
  Frozen layers: 18
============================================================
```

### Checkpoint Saving

During training, checkpoints are saved with the updated configuration:

```
Epoch 0 | Step 0 | Loss: 2.3456
Saving checkpoint with 24 layers (actual: 24)
Saving 24 layers in checkpoint
✓ Saved generation_config.json
✓ Saved processor and tokenizer files
✓ Saved speech_tokenizer to output/checkpoint-epoch-0/speech_tokenizer
✓ Saved speaker encoder config for speaker: my_speaker
✓ Saved model.safetensors

============================================================
Checkpoint contents:
============================================================
  - config.json
  - generation_config.json
  - model.safetensors
  - tokenizer_config.json
  - vocab.json
  - merges.txt
  - preprocessor_config.json
  - speech_tokenizer/config.json
  - speech_tokenizer/model.safetensors
  - speech_tokenizer/tokenizer_config.json
  - speaker_encoder/speaker_config.json
  - README.md
============================================================
Total files saved: 12
============================================================

✓ Checkpoint saved to output/checkpoint-epoch-0
```

The saved checkpoint includes:
- All 24 layers (18 frozen + 6 trainable)
- Updated configuration with `num_hidden_layers = 24`
- Speaker embedding at index 3000
- **Complete set of files for model loading**:
  - `config.json` - Model configuration
  - `generation_config.json` - Generation parameters
  - `model.safetensors` - Model weights
  - `tokenizer_config.json` - Text tokenizer config
  - `vocab.json` - Vocabulary file
  - `merges.txt` - BPE merges
  - `preprocessor_config.json` - Preprocessor config
  - `speech_tokenizer/` - Speech tokenizer (codec encoder/decoder)
  - `speaker_encoder/speaker_config.json` - Speaker information
  - `README.md` - Usage instructions

### Benefits

1. **Better Speaker Adaptation**: New layers can learn speaker-specific patterns without forgetting pre-trained knowledge
2. **Faster Training**: Only training 6 layers instead of 20 reduces training time
3. **Memory Efficient**: Fewer trainable parameters means lower memory usage
4. **Flexible**: Easy to adjust the number of replaced/added layers via command-line arguments

### Layer Initialization

New layers are initialized using Xavier/Glorot uniform initialization:
- Linear layers: `nn.init.xavier_uniform_`
- Embedding layers: `nn.init.normal_(mean=0.0, std=0.02)`
- Bias terms: Initialized to zeros

This ensures the new layers start with reasonable weights and can be trained effectively.

### Loading Fine-Tuned Models

When loading a fine-tuned model with modified layers, ensure you use the correct configuration:

```python
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "output/checkpoint-epoch-0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```

The model will automatically load with 24 layers as specified in the saved configuration.

### Troubleshooting

#### Issue: "Configuration layer count doesn't match actual layers"

The training script automatically detects and fixes this issue. If you see this warning, the configuration will be corrected before saving.

#### Issue: Out of Memory

If you encounter OOM errors:
- Reduce `--batch_size`
- Reduce `--add_new_layers` to decrease trainable parameters
- Use gradient accumulation (already set to 8 by default)

#### Issue: Slow Training

If training is too slow:
- Reduce `--add_new_layers` to decrease trainable parameters
- Increase `--batch_size` if memory allows
- Use mixed precision training (already enabled with bf16)

### Advanced Usage

#### Programmatic Layer Replacement

You can also use the layer replacement utility programmatically:

```python
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from finetuning.layer_utils import replace_and_add_layers, print_model_summary

# Load model
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    torch_dtype=torch.bfloat16
)

# Replace and add layers
model.model = replace_and_add_layers(
    model.model,
    replace_last_n=2,
    add_new_layers=4,
    freeze_original_layers=True,
    verbose=True
)

# Print summary
print_model_summary(model.model)
```

#### Custom Layer Initialization

If you want to use a different initialization scheme, modify the `initialize_decoder_layer` function in `finetuning/layer_utils.py`:

```python
def initialize_decoder_layer(layer: Qwen3TTSTalkerDecoderLayer):
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            # Use Kaiming initialization instead of Xavier
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    layer.apply(_init_weights)
```

### Checkpoint Structure

When training completes, checkpoints are saved with all necessary files for complete model loading.

#### Complete Checkpoint Contents

```
checkpoint-epoch-N/
├── config.json                          # Main model configuration
├── generation_config.json               # Generation parameters (temperature, top_k, etc.)
├── model.safetensors                    # Model weights (all layers except speaker_encoder)
├── tokenizer_config.json                # Text tokenizer configuration
├── vocab.json                           # Vocabulary for text tokenizer
├── merges.txt                           # BPE merges for text tokenizer
├── preprocessor_config.json             # Preprocessor configuration
├── speech_tokenizer/                    # Speech tokenizer directory
│   ├── config.json                      # Speech tokenizer config
│   ├── model.safetensors                # Speech tokenizer weights
│   └── tokenizer_config.json            # Speech tokenizer config
├── speaker_encoder/                     # Speaker encoder directory
│   └── speaker_config.json              # Speaker configuration (speaker name, embedding dim)
└── README.md                            # Usage instructions
```

#### File Descriptions

**Core Model Files:**

| File | Description | Required |
|------|-------------|----------|
| `config.json` | Main model configuration including layer count, hidden size, attention heads, etc. | Yes |
| `generation_config.json` | Default generation parameters (temperature, top_p, top_k, etc.) | Yes |
| `model.safetensors` | All model weights except speaker_encoder (frozen during training) | Yes |

**Text Tokenizer Files:**

| File | Description | Required |
|------|-------------|----------|
| `tokenizer_config.json` | Configuration for the text tokenizer | Yes |
| `vocab.json` | Vocabulary mapping for text tokens | Yes |
| `merges.txt` | BPE merge rules for text tokenization | Yes |
| `preprocessor_config.json` | Preprocessing configuration for text | Yes |

**Speech Tokenizer Files:**

| File | Description | Required |
|------|-------------|----------|
| `speech_tokenizer/config.json` | Speech tokenizer configuration | Yes |
| `speech_tokenizer/model.safetensors` | Speech tokenizer weights (codec encoder/decoder) | Yes |
| `speech_tokenizer/tokenizer_config.json` | Speech tokenizer config | Yes |

**Speaker Information:**

| File | Description | Required |
|------|-------------|----------|
| `speaker_encoder/speaker_config.json` | Speaker name and embedding dimension | Yes |

**Documentation:**

| File | Description | Required |
|------|-------------|----------|
| `README.md` | Usage instructions and model information | No |

#### Why Each File is Needed

**1. config.json**
- Contains the complete model architecture configuration
- Includes layer count (updated after layer replacement)
- Specifies hidden size, attention heads, etc.
- Required for reconstructing the model architecture

**2. generation_config.json**
- Stores default generation parameters
- Ensures consistent generation behavior
- Includes temperature, top_p, top_k, max_new_tokens, etc.

**3. model.safetensors**
- Contains all trainable model weights
- Excludes speaker_encoder (frozen during training)
- Includes the new speaker embedding at index 3000
- Uses safetensors format for security and efficiency

**4. Text Tokenizer Files (tokenizer_config.json, vocab.json, merges.txt)**
- Required for encoding input text
- Converts text to token IDs
- Essential for the model to understand input

**5. preprocessor_config.json**
- Configuration for text preprocessing
- Handles special tokens, padding, etc.

**6. speech_tokenizer/ directory**
- **CRITICAL**: Contains the codec encoder/decoder
- Required for converting audio codes to waveform
- Without this, you cannot generate audio output
- Includes:
  - `config.json`: Speech tokenizer configuration
  - `model.safetensors`: Codec model weights
  - `tokenizer_config.json`: Additional tokenizer config

**7. speaker_encoder/speaker_config.json**
- Stores speaker name and embedding dimension
- Documents which speaker the model was fine-tuned for
- Useful for reference and model management

**8. README.md**
- Provides usage instructions
- Documents model information
- Helpful for users loading the checkpoint

#### Speaker Embedding Storage

The new speaker embedding is stored in the model weights:

```python
# Location: model.safetensors
# Key: talker.model.codec_embedding.weight[3000]
# Shape: [embedding_dim] (typically 1024)
```

This embedding is extracted from the reference audio during training and inserted at index 3000 of the codec embedding layer.

#### Loading a Checkpoint

```python
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# Load the complete checkpoint
model = Qwen3TTSModel.from_pretrained(
    "./output/checkpoint-epoch-0",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# The model will automatically load:
# - Model architecture from config.json
# - Model weights from model.safetensors
# - Text tokenizer from tokenizer files
# - Speech tokenizer from speech_tokenizer/ directory
# - Generation parameters from generation_config.json
```

#### What's NOT Saved

**Speaker Encoder Weights**

The speaker encoder weights are **NOT** saved because:
1. They are frozen during training
2. They are shared across all speakers
3. They can be loaded from the base model
4. Saving them would duplicate large weights unnecessarily

If you need the speaker encoder, load it from the base model:

```python
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# Load base model to get speaker encoder
base_model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
speaker_encoder = base_model.model.speaker_encoder

# Load fine-tuned model
fine_tuned_model = Qwen3TTSModel.from_pretrained("./output/checkpoint-epoch-0")
```

#### Checkpoint Size

Typical checkpoint sizes:

| Component | Size |
|-----------|------|
| model.safetensors | ~3-4 GB |
| speech_tokenizer/ | ~500 MB |
| Config files | ~10 KB |
| **Total** | **~3.5-4.5 GB** |

#### Verification

After saving, the training script will display:

```
============================================================
Checkpoint contents:
============================================================
  - config.json
  - generation_config.json
  - model.safetensors
  - tokenizer_config.json
  - vocab.json
  - merges.txt
  - preprocessor_config.json
  - speech_tokenizer/config.json
  - speech_tokenizer/model.safetensors
  - speech_tokenizer/tokenizer_config.json
  - speaker_encoder/speaker_config.json
  - README.md
============================================================
Total files saved: 12
============================================================
```

#### Troubleshooting

**Issue: Missing speech_tokenizer directory**

**Solution**: Ensure the base model has a speech_tokenizer attribute. Check:

```python
print(hasattr(model, 'speech_tokenizer'))
print(model.speech_tokenizer is not None)
```

**Issue: Cannot load checkpoint**

**Solution**: Verify all required files are present:
- config.json
- model.safetensors
- speech_tokenizer/ directory with all its files

**Issue: Generation fails**

**Solution**: Check that:
1. speech_tokenizer/ directory exists and contains model.safetensors
2. generation_config.json exists
3. All tokenizer files are present

#### Best Practices

1. **Always save the complete checkpoint** - Don't skip any files
2. **Verify checkpoint contents** - Check the file list after saving
3. **Test loading** - Try loading the checkpoint immediately after saving
4. **Document speaker information** - Keep track of which speaker each checkpoint is for
5. **Use consistent naming** - Include speaker name and epoch in checkpoint directory name

## Training Workflows

### Workflow 1: Simple Training - Quick Start

Best for quick experiments and debugging.

```bash
# Step 1: Create .env from template
cp .env.training.example .env

# Step 2: Verify setup
python test_setup.py

# Step 3: Train with defaults
python train.py

# Output: ./output/best/, ./output/last/
```

### Workflow 2: Training with JSONL Files

Best for large datasets or repeated training runs.

```bash
# Step 1: Prepare data to JSONL
python data_processing.py --mode prepare

# Step 2: Train with JSONL mode
DATA_MODE=jsonl python train.py

# Output: ./output/best/, ./output/last/
```

### Workflow 3: Debug with Small Dataset

Best for debugging code and setup issues.

```bash
# Step 1: Prepare small dataset
python data_processing.py --mode prepare --max_samples 10

# Step 2: Train for 1 epoch
NUM_EPOCHS=1 python train.py
```

### Workflow 4: Resume from Checkpoint

```bash
# Training automatically saves checkpoints
# To resume, modify output path or use existing checkpoint
# Currently, resuming requires manual intervention
```

## Monitoring Training

### Console Output

The training script outputs progress to the console:

```
Epoch 0 | Step 10 | Loss: 2.3456 | LR: 1.8e-5
Epoch 0 | Step 20 | Loss: 2.1234 | LR: 1.7e-5
```

### Log Files

The training script creates two log files:

**training_log.jsonl** - Training metrics:
```json
{"step": 100, "epoch": 0, "loss": 2.3456, "learning_rate": 1.8e-5, "timestamp": "2026-02-24T10:30:00"}
{"step": 200, "epoch": 0, "loss": 2.1234, "learning_rate": 1.7e-5, "timestamp": "2026-02-24T10:35:00"}
```

**validation_log.jsonl** - Validation metrics:
```json
{"step": 500, "epoch": 0, "loss": 2.1234, "metrics": {"speaker_embedding_similarity": 0.95}, "timestamp": "2026-02-24T10:35:00"}
```

### WandB Dashboard

When WandB is enabled, comprehensive monitoring is available:

```bash
# 1. Login to WandB (if needed)
wandb login

# 2. After starting training, visit the URL shown in the output
# Or manually:
wandb dashboard
```

**WandB Dashboards show:**
- Training loss over time
- Validation loss
- Learning rate schedule
- Speaker embedding similarity
- Checkpoint information
- Model comparison

### Output Structure

```
output/
├── best/                          # Best model (lowest validation loss)
│   ├── model.safetensors
│   ├── config.json
│   ├── training_state.json
│   └── processor files...
├── last/                          # Last checkpoint
│   ├── model.safetensors
│   ├── config.json
│   ├── training_state.json
│   └── processor files...
├── checkpoint-500/                # Intermediate checkpoints
├── checkpoint-1000/
├── training_log.jsonl             # Training loss logs
└── validation_log.jsonl           # Validation loss and metrics logs
```
│   ├── model.safetensors
│   └── training_state.pt
├── checkpoint-500/                # Intermediate checkpoints
├── checkpoint-1000/
├── epoch-1/
├── epoch-2/
└── epoch-3/
```

## Using the Trained Model

### Load Fine-Tuned Model

```python
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# Load fine-tuned model
model = Qwen3TTSModel.from_pretrained("./output/best")
```

### Generate Speech

```python
import soundfile as sf

# Generate speech
text = "Ka ce, yaya lafiya?"  # Hello, how are you? (Hausa)
ref_audio = "voices/speaker/reference.wav"

# Generate with x-vector mode (faster)
output = model.generate(
    text=text,
    reference_audio=ref_audio,
    use_xvector_only=True
)

# Generate with ICL mode (better quality)
output = model.generate(
    text=text,
    reference_audio=ref_audio,
    ref_text="This is the reference audio content.",
    use_xvector_only=False
)

# Save output
sf.write("output.wav", output.audio, output.sampling_rate)
```

### Batch Generation

```python
texts = [
    "Text 1",
    "Text 2",
    "Text 3"
]

for i, text in enumerate(texts):
    output = model.generate(
        text=text,
        reference_audio=ref_audio,
        use_xvector_only=False
    )
    sf.write(f"output_{i}.wav", output.audio, output.sampling_rate)
```

## Architecture Overview

### Qwen3-TTS Model Architecture

Qwen3-TTS is a transformer-based text-to-speech system using a codec-based approach for audio generation.

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen3-TTS Model                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  Speaker Encoder │      │   Talker Model   │            │
│  │                  │      │                  │            │
│  │  Reference Audio │─────▶│  Text Embedding  │            │
│  │       ↓          │      │  Codec Embedding │            │
│  │  Mel Spectrogram │      │  (with Speaker   │            │
│  │       ↓          │      │   Embedding)     │            │
│  │  Speaker Embed   │      │       ↓          │            │
│  │  (256-dim)       │      │  Transformer     │            │
│  └──────────────────┘      │       ↓          │            │
│                            │  Audio Codes     │            │
│                            │  Prediction      │            │
│                            │       ↓          │            │
│                            │  Sub-Talker      │            │
│                            │  (Auxiliary)     │            │
│                            └──────────────────┘            │
│                                                              │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  Codec Encoder   │      │  Codec Decoder   │            │
│  │                  │      │                  │            │
│  │  Raw Audio       │─────▶│  Audio Codes     │─────▶ Audio│
│  │       ↓          │      │       ↓          │      Output│
│  │  Audio Codes     │      │  Neural Codec    │            │
│  │  (16 channels)   │      │  Vocoder         │            │
│  └──────────────────┘      └──────────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Speaker Encoder**: Extracts 256-dimensional speaker embedding from reference audio
2. **Talker Model**: Core generation component with text and codec embeddings
3. **Codec Encoder**: Encodes raw audio to discrete audio codes (16 channels)
4. **Codec Decoder**: Decodes audio codes back to waveform

### Voice Cloning Modes

| Feature | x-vector Only | ICL (In-Context Learning) |
|---------|---------------|---------------------------|
| **Reference Text** | Not required | Required |
| **Quality** | Good (85-90%) | Excellent (95-98%) |
| **Speed** | Faster (~2-3x) | Slower (~1-1.5x) |
| **Use Case** | Real-time, quick tests | High-quality output |
| **Prosody Matching** | Basic | Advanced |

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size in .env
BATCH_SIZE=1

# Increase gradient accumulation
GRADIENT_ACCUMULATION_STEPS=8

# Use mixed precision
MIXED_PRECISION=bf16
```

### Slow Training

```bash
# Limit samples for testing
MAX_TRAIN_SAMPLES=100

# Reduce epochs
NUM_EPOCHS=1

# Use faster attention (automatically detects)
# Flash Attention 2 -> SDPA fallback
```

### Flash Attention Not Available

The scripts automatically fall back to SDPA (Scaled Dot Product Attention) if Flash Attention is not available. No action needed.

```bash
# Output will show:
# ⚠ Flash attention not available, falling back to SDPA: ...
# ✓ Model loaded with SDPA
```

### No CUDA Available

```bash
# Use CPU in .env
DEVICE=cpu
BATCH_SIZE=1  # Keep small for CPU
```

### SoX Not Found Warning

SoX is only needed for audio processing. If you see this warning and need audio processing:

```bash
# Ubuntu/Debian
sudo apt-get install sox

# macOS
brew install sox

# Otherwise, the warning can be ignored
```

### Dataset Download Issues

```bash
# Check internet connection
# Verify Hugging Face dataset exists
python -c "from datasets import load_dataset; print(load_dataset('vaghawan/hausa-tts-22k'))"

# Use a smaller sample for debugging
MAX_TRAIN_SAMPLES=10
```

### WandB Login Issues

```bash
# Login to WandB
wandb login

# Or disable WandB
USE_WANDB=false
```

### Hugging Face Upload Issues

```bash
# Verify token is valid
python -c "from huggingface_hub import HfApi; api = HfApi(); print(api.whoami())"

# Check repository exists or create it
# The script automatically creates the repo if it doesn't exist
```

### Model Loading Errors

```bash
# Verify model path is correct
python -c "from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel; model = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-Base')"

# Check output directory exists
ls -la ./output/best/
```

## Training Analysis & Best Practices

### Current Training Configuration

#### Model Information

| Parameter | Value |
|-----------|-------|
| **Model Name** | Qwen/Qwen3-TTS-12Hz-1.7B-Base |
| **Model Size** | 1.7 Billion Parameters |
| **Audio Codec** | 12Hz Audio Codec (16 codebooks) |
| **Training Mode** | Custom Voice (Voice Cloning) |
| **Attention Implementation** | SDPA (Scaled Dot Product Attention) |
| **Mixed Precision** | BF16 |

#### Training Hyperparameters

| Parameter | Current Value | Recommended Range | Description |
|-----------|---------------|-------------------|-------------|
| **Batch Size** | 8 | 1-4 | Samples per batch |
| **Learning Rate** | 2e-4 (0.0002) | 1e-5 to 1e-4 | Optimizer learning rate |
| **Gradient Accumulation** | 4 (implicit) | 8-16 | Steps before weight update |
| **Number of Epochs** | 3 | 10-50 | Training iterations |
| **Weight Decay** | 0.01 | 0.001-0.1 | L2 regularization |
| **Warmup Steps** | 100 | 100-500 | LR warm-up period |
| **Max Gradient Norm** | 1.0 | 0.5-1.0 | Gradient clipping |
| **Effective Batch Size** | 32 | 32-64 | batch_size × grad_accum |

#### Dataset Information

| Parameter | Value |
|-----------|-------|
| **Dataset Name** | vaghawan/hausa-tts-22k |
| **Language** | Hausa |
| **Training Samples** | 984 (full) / 100 (limited) |
| **Validation Samples** | Not configured |
| **Text Length** | Short to medium sentences |
| **Audio Codec** | Pre-extracted (16 codebooks × 16 timesteps) |
| **Reference Voice** | English Voice (cross-language cloning) |

### Common Training Issues

#### Issue: Loss Drops to 0.0000 Too Quickly

**WARNING:** Loss dropping to 0.0000 immediately indicates:

1. **Overfitting** - Model memorized small dataset
2. **Loss Computation Issue** - Possibly NaN/Inf masked as 0
3. **Insufficient Training Data** - Only 100 samples
4. **High Learning Rate** - 2e-4 is too aggressive for 1.7B model
5. **Cross-language Mismatch** - English reference for Hausa text

**Root Causes:**

1. **All samples have identical audio_codes**
   - Every sample in your dataset has the EXACT SAME audio codes
   - Model learns to predict the same codes regardless of text
   - Loss becomes zero because target is always identical

2. **Cross-language voice cloning problem**
   - Reference voice: English
   - Target text: Hausa
   - Phonetic mismatch causes poor articulation

3. **Insufficient data diversity**
   - Only 100 samples
   - Limited phonetic coverage
   - Minimal prosodic variation

### Recommended Training Configuration

#### Best Practices Configuration

```bash
# .env file - Recommended Settings

# Model Settings
MODEL_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base
OUTPUT_DIR=./output_v2

# Critical Hyperparameters
BATCH_SIZE=1                    # Per GPU
LR=3e-5                        # Conservative LR
NUM_EPOCHS=30                  # Sufficient training
GRADIENT_ACCUMULATION_STEPS=16 # Effective batch = 16
EFFECTIVE_BATCH_SIZE=16        # 1 × 16

# Regularization
WEIGHT_DECAY=0.01              # Default
MAX_GRAD_NORM=0.5              # Stricter clipping
WARMUP_STEPS=500               # Gentle warmup

# Dataset
DATASET_NAME=vaghawan/hausa-tts-22k
MAX_TRAIN_SAMPLES=5000         # More data
MAX_EVAL_SAMPLES=500           # Validation
TRAIN_JSONL=./data/train_v2.jsonl
VALIDATION_JSONL=./data/val_v2.jsonl

# Reference Audio (TARGET LANGUAGE!)
REF_AUDIO_PATH=./voices/hausa_reference.wav
REF_TEXT="Sanannun za ku iya tattaunawa da bayyana..."  # Hausa text

# Logging
LOGGING_STEPS=5                # Frequent logging
EVAL_STEPS=100                 # Evaluate every 100 steps
SAVE_STEPS=500                 # Save checkpoints
```

#### Alternative: Aggressive Training (More Computation)

```bash
# For best results with more resources:
BATCH_SIZE=2
LR=5e-5
NUM_EPOCHS=50
GRADIENT_ACCUMULATION_STEPS=8
MAX_TRAIN_SAMPLES=10000
```

### Action Plan for Best Voice Cloning Results

#### Phase 1: Immediate Fixes (Priority 1)

**1.1 Fix Training Hyperparameters**

```bash
# Update .env file with these values
BATCH_SIZE=1                    # Reduce for better gradient quality
LR=5e-5                        # Lower learning rate (2.5x lower)
NUM_EPOCHS=20                  # More epochs for better learning
GRADIENT_ACCUMULATION_STEPS=16 # Increase effective batch size
WARMUP_STEPS=300               # Longer warmup
LOGGING_STEPS=5                # More frequent logging
```

**1.2 Increase Dataset Size**

```bash
# Remove sample limit
MAX_TRAIN_SAMPLES=5000         # Use more samples
MAX_EVAL_SAMPLES=500           # Add validation data
```

**1.3 Fix Audio Code Extraction**

**CRITICAL:** Currently all samples have identical codes. Need to extract unique codes per audio file.

Check `dataset_tool.py` line:

```python
# Look for this pattern - codes should vary per sample
audio_codes = tokenizer.encode(audio_data)  # Should extract from actual audio
```

**1.4 Add Validation Set**

```bash
# Create proper validation split
VALIDATION_JSONL=./data/validation.jsonl
MAX_EVAL_SAMPLES=200
```

#### Phase 2: Audio & Data Quality (Priority 2)

**2.1 Language-Specific Reference Audio**

**CURRENT PROBLEM:** Using English reference for Hausa text

**SOLUTION:** Choose reference audio from target language

```bash
# Options:
# 1. Use Hausa speaker's audio (if available)
REF_AUDIO_PATH=./voices/hausa_speaker.wav

# 2. Use multi-lingual speaker
REF_AUDIO_PATH=./voices/multilingual_speaker.wav
```

**Or use the target language data:**
```python
# Use one of the training samples as reference
REF_AUDIO_PATH=/path/to/clear/hausa_sample.wav
REF_TEXT="Sample text in Hausa with good articulation"
```

**2.2 Audio Quality Requirements**

Your reference audio should:

✅ **DO:**
- Be 10-30 seconds long
- Have clear pronunciation
- Contain multiple phonemes
- Have natural prosody
- Be in the target language (Hausa)
- Have consistent volume
- Be recorded at 24kHz

❌ **DON'T:**
- Use background noise
- Use clipped audio
- Use very short (<5s) samples
- Use cross-language for critical tasks
- Use very long (>60s) samples

### Expected Results Analysis

#### Current Configuration (Estimated Quality)

| Metric | Expected | Reality |
|--------|----------|---------|
| **MOS (Mean Opinion Score)** | 1.5-2.0/5.0 | ~1.0/5.0 (Poor) |
| **WER (Word Error Rate)** | >50% | Likely >80% |
| **Voice Similarity** | 30-40% | ~20% |
| **Naturalness** | Very Low | Minimal |
| **Intelligibility** | Poor | Unintelligible |

#### Recommended Configuration (Expected Quality)

| Metric | Expected | Notes |
|--------|----------|-------|
| **MOS** | 3.5-4.2/5.0 | Good to Very Good |
| **WER** | 15-25% | Intelligible |
| **Voice Similarity** | 70-85% | High |
| **Naturalness** | High | Pleasant speech |
| **Intelligibility** | Excellent | Clear articulation |

### Training Monitoring Checklist

#### During Training

- [ ] Loss decreases gradually (not instantly to 0)
- [ ] Loss doesn't plateau too early
- [ ] Learning rate scheduler works correctly
- [ ] No NaN/Inf gradients
- [ ] Validation tracks training reasonably
- [ ] Training logs captured

#### After Training

- [ ] Generate test samples
- [ ] Conduct subjective evaluation (listen)
- [ ] Calculate objective metrics (MOS, WER)
- [ ] Compare with baseline
- [ ] Test on unseen text
- [ ] Test different emotions/styles

### Success Metrics

Define success before training:

| Metric | Minimum Acceptable | Target | Excellent |
|--------|-------------------|--------|-----------|
| **Final Training Loss** | <0.5 | <0.2 | <0.1 |
| **Validation Loss** | <1.0 | <0.5 | <0.3 |
| **MOS Score** | 2.5/5 | 3.5/5 | 4.2/5 |
| **WER** | <40% | <25% | <15% |
| **Voice Similarity** | 50% | 70% | 85% |

### Summary of Key Recommendations

#### Must-Do (Critical)

1. ✅ **Fix audio code extraction** - Get unique codes per sample
2. ✅ **Increase training data** - Use 5000+ samples
3. ✅ **Use target language reference** - Hausa speaker for Hausa text
4. ✅ **Lower learning rate** - 3e-5 to 5e-5
5. ✅ **Train more epochs** - 20-30 epochs
6. ✅ **Add validation** - Monitor overfitting

#### Should-Do (Important)

7. ✅ **Use smaller batch size** - 1-2 for better generalization
8. ✅ **Implement gradient accumulation** - Effective batch 16+
9. ✅ **Add learning rate scheduling** - Cosine annealing
10. ✅ **Monitor training properly** - Log all metrics

#### Nice-to-Do (Optional)

11. 🔄 Use data augmentation
12. 🔄 Implement curriculum learning
13. 🔄 Try adapter/Lora training
14. 🔄 Multi-speaker training

### Troubleshooting Guide

| Problem | Cause | Solution |
|---------|-------|----------|
| **Loss = 0 immediately** | Same codes in all samples | Fix code extraction |
| **Loss = NaN** | LR too high | Reduce LR, add clipping |
| **No voice identity** | Wrong reference audio | Use target language speaker |
| **Poor intelligibility** | Insufficient training | More epochs, more data |
| **Memory OOM** | Batch too big | Reduce batch, increase grad accum |
| **Training too slow** | Inefficient pipeline | Optimize dataloader, use pinning |

## License

Apache-2.0 (same as Qwen3-TTS)

## Additional Resources

- [Qwen3-TTS Documentation](./Qwen3-TTS/README.md)
- [Hugging Face Dataset](https://huggingface.co/datasets/vaghawan/hausa-tts-22k)
- [WandB Documentation](https://docs.wandb.ai/)
- [Qwen3-TTS Models](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)

# Qwen3-TTS

<br>

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/qwen3_tts_logo.png" width="400"/>
<p>

<p align="center">
&nbsp&nbsp🤗 <a href="https://huggingface.co/collections/Qwen/qwen3-tts">Hugging Face</a>&nbsp&nbsp | &nbsp&nbsp🤖 <a href="https://modelscope.cn/collections/Qwen/Qwen3-TTS">ModelScope</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://qwen.ai/blog?id=qwen3tts-0115">Blog</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://arxiv.org/abs/2601.15621">Paper</a>&nbsp&nbsp
<br>
🖥️ <a href="https://huggingface.co/spaces/Qwen/Qwen3-TTS">Hugging Face Demo</a>&nbsp&nbsp | &nbsp&nbsp 🖥️ <a href="https://modelscope.cn/studios/Qwen/Qwen3-TTS">ModelScope Demo</a>&nbsp&nbsp | &nbsp&nbsp💬 <a href="https://github.com/QwenLM/Qwen/blob/main/assets/wechat.png">WeChat (微信)</a>&nbsp&nbsp | &nbsp&nbsp🫨 <a href="https://discord.gg/CV4E9rpNSD">Discord</a>&nbsp&nbsp | &nbsp&nbsp📑 <a href="https://help.aliyun.com/zh/model-studio/qwen-tts-realtime">API</a>

</p>

We release **Qwen3-TTS**, a series of powerful speech generation capabilities developed by Qwen, offering comprehensive support for voice clone, voice design, ultra-high-quality human-like speech generation, and natural language-based voice control. It provides developers and users with the most extensive set of speech generation features available.


## News
* 2026.1.22: 🎉🎉🎉 We have released [Qwen3-TTS](https://huggingface.co/collections/Qwen/qwen3-tts) series (0.6B/1.7B) based on Qwen3-TTS-Tokenizer-12Hz. Please check our [blog](https://qwen.ai/blog?id=qwen3tts-0115)!

## Contents <!-- omit in toc -->

- [Overview](#overview)
  - [Introduction](#introduction)
  - [Model Architecture](#model-architecture)
  - [Released Models Description and Download](#released-models-description-and-download)
- [Quickstart](#quickstart)
  - [Environment Setup](#environment-setup)
  - [Python Package Usage](#python-package-usage)
    - [Custom Voice Generation](#custom-voice-generate)
    - [Voice Design](#voice-design)
    - [Voice Clone](#voice-clone)
    - [Voice Design then Clone](#voice-design-then-clone)
    - [Tokenizer Encode and Decode](#tokenizer-encode-and-decode)
  - [Launch Local Web UI Demo](#launch-local-web-ui-demo)
  - [DashScope API Usage](#dashscope-api-usage)
- [vLLM Usage](#vllm-usage)
- [Fine Tuning](#fine-tuning)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Overview
### Introduction

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/qwen3_tts_introduction.png" width="90%"/>
<p>

Qwen3-TTS covers 10 major languages (Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian) as well as multiple dialectal voice profiles to meet global application needs. In addition, the models feature strong contextual understanding, enabling adaptive control of tone, speaking rate, and emotional expression based on instructions and text semantics, and they show markedly improved robustness to noisy input text. Key features:

* **Powerful Speech Representation**: Powered by the self-developed Qwen3-TTS-Tokenizer-12Hz, it achieves efficient acoustic compression and high-dimensional semantic modeling of speech signals. It fully preserves paralinguistic information and acoustic environmental features, enabling high-speed, high-fidelity speech reconstruction through a lightweight non-DiT architecture.
* **Universal End-to-End Architecture**: Utilizing a discrete multi-codebook LM architecture, it realizes full-information end-to-end speech modeling. This completely bypasses the information bottlenecks and cascading errors inherent in traditional LM+DiT schemes, significantly enhancing the model’s versatility, generation efficiency, and performance ceiling.
* **Extreme Low-Latency Streaming Generation**: Based on the innovative Dual-Track hybrid streaming generation architecture, a single model supports both streaming and non-streaming generation. It can output the first audio packet immediately after a single character is input, with end-to-end synthesis latency as low as 97ms, meeting the rigorous demands of real-time interactive scenarios.
* **Intelligent Text Understanding and Voice Control**: Supports speech generation driven by natural language instructions, allowing for flexible control over multi-dimensional acoustic attributes such as timbre, emotion, and prosody. By deeply integrating text semantic understanding, the model adaptively adjusts tone, rhythm, and emotional expression, achieving lifelike “what you imagine is what you hear” output.


### Model Architecture

<p align="center">
    <img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/overview.png" width="80%"/>
<p>

### Released Models Description and Download

Below is an introduction and download information for the Qwen3-TTS models that have already been released. Other models mentioned in the technical report will be released in the near future. Please select and download the model that fits your needs.

| Tokenizer Name                      | Description |
|---------------------------------|-------------|
| Qwen3-TTS-Tokenizer-12Hz        | The Qwen3-TTS-Tokenizer-12Hz model which can encode the input speech into codes and decode them back into speech. |


| Model | Features | Language Support | Streaming | Instruction Control |
|---|---|---|---|---|
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | Performs voice design based on user-provided descriptions. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-CustomVoice | Provides style control over target timbres via user instructions; supports 9 premium timbres covering various combinations of gender, age, language, and dialect. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ | ✅ |
| Qwen3-TTS-12Hz-1.7B-Base | Base model capable of 3-second rapid voice clone from user audio input; can be used for fine-tuning (FT) other models. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ |  |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | Supports 9 premium timbres covering various combinations of gender, age, language, and dialect. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ |  |
| Qwen3-TTS-12Hz-0.6B-Base | Base model capable of 3-second rapid voice clone from user audio input; can be used for fine-tuning (FT) other models. | Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian | ✅ |  |

During model loading in the qwen-tts package or vLLM, model weights will be automatically downloaded based on the model name. However, if your runtime environment is not conducive to downloading weights during execution, you can refer to the following commands to manually download the model weights to a local directory:

```bash
# Download through ModelScope (recommended for users in Mainland China)
pip install -U modelscope
modelscope download --model Qwen/Qwen3-TTS-Tokenizer-12Hz  --local_dir ./Qwen3-TTS-Tokenizer-12Hz 
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local_dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir ./Qwen3-TTS-12Hz-1.7B-Base
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local_dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
modelscope download --model Qwen/Qwen3-TTS-12Hz-0.6B-Base --local_dir ./Qwen3-TTS-12Hz-0.6B-Base

# Download through Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ./Qwen3-TTS-Tokenizer-12Hz
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --local-dir ./Qwen3-TTS-12Hz-1.7B-VoiceDesign
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ./Qwen3-TTS-12Hz-1.7B-Base
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-0.6B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir ./Qwen3-TTS-12Hz-0.6B-Base
```


## Quickstart

### Environment Setup

The easiest way to quickly use Qwen3-TTS is to install the `qwen-tts` Python package from PyPI. This will pull in the required runtime dependencies and allow you to load any released Qwen3-TTS model. We recommend using a **fresh, isolated environment** to avoid dependency conflicts with existing packages. You can create a clean Python 3.12 environment like this:

```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
```

then run:

```bash
pip install -U qwen-tts
```

If you want to develop or modify the code locally, install from source in editable mode.

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
pip install -e .
```

Additionally, we recommend using FlashAttention 2 to reduce GPU memory usage.

```bash
pip install -U flash-attn --no-build-isolation
```

If your machine has less than 96GB of RAM and lots of CPU cores, run:

```bash
MAX_JOBS=4 pip install -U flash-attn --no-build-isolation
```

Also, you should have hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [FlashAttention repository](https://github.com/Dao-AILab/flash-attention). FlashAttention 2 can only be used when a model is loaded in `torch.float16` or `torch.bfloat16`.


### Python Package Usage

After installation, you can import `Qwen3TTSModel` to run custom voice TTS, voice design, and voice clone. The model weights can be specified either as a Hugging Face model id (recommended) or as a local directory path you downloaded. For all the `generate_*` functions below, besides the parameters shown and explicitly documented, you can also pass generation kwargs supported by Hugging Face Transformers `model.generate`, e.g., `max_new_tokens`, `top_p`, etc.

#### Custom Voice Generate

For custom voice models (`Qwen3-TTS-12Hz-1.7B/0.6B-CustomVoice`), you just need to call `generate_custom_voice`, passing a single string or a batch list, along with `language`, `speaker`, and optional `instruct`. You can also call `model.get_supported_speakers()` and `model.get_supported_languages()` to see which speakers and languages the current model supports.

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# single inference
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
    speaker="Vivian",
    instruct="用特别愤怒的语气说", # Omit if not needed.
)
sf.write("output_custom_voice.wav", wavs[0], sr)

# batch inference
wavs, sr = model.generate_custom_voice(
    text=[
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。", 
        "She said she would be here by noon."
    ],
    language=["Chinese", "English"],
    speaker=["Vivian", "Ryan"],
    instruct=["", "Very happy."]
)
sf.write("output_custom_voice_1.wav", wavs[0], sr)
sf.write("output_custom_voice_2.wav", wavs[1], sr)
```

For `Qwen3-TTS-12Hz-1.7B/0.6B-CustomVoice` models, the supported speaker list and speaker descriptions are provided below. We recommend using each speaker’s native language for the best quality. Of course, each speaker can speak any language supported by the model.

| Speaker | Voice Description  |  Native language |
| --- | --- | --- |
| Vivian | Bright, slightly edgy young female voice. | Chinese |
| Serena | Warm, gentle young female voice. | Chinese |
| Uncle_Fu | Seasoned male voice with a low, mellow timbre. | Chinese |
| Dylan | Youthful Beijing male voice with a clear, natural timbre. | Chinese (Beijing Dialect) |
| Eric | Lively Chengdu male voice with a slightly husky brightness. | Chinese (Sichuan Dialect) |
| Ryan | Dynamic male voice with strong rhythmic drive. | English |
| Aiden | Sunny American male voice with a clear midrange. | English |
| Ono_Anna | Playful Japanese female voice with a light, nimble timbre. | Japanese |
| Sohee | Warm Korean female voice with rich emotion. | Korean |

#### Voice Design

For the voice design model (`Qwen3-TTS-12Hz-1.7B-VoiceDesign`), you can use `generate_voice_design` to provide the target text and a natural-language `instruct` description.

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# single inference
wavs, sr = model.generate_voice_design(
    text="哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
    language="Chinese",
    instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
)
sf.write("output_voice_design.wav", wavs[0], sr)

# batch inference
wavs, sr = model.generate_voice_design(
    text=[
      "哥哥，你回来啦，人家等了你好久好久了，要抱抱！",
      "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
    ],
    language=["Chinese", "English"],
    instruct=[
      "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
      "Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice."
    ]
)
sf.write("output_voice_design_1.wav", wavs[0], sr)
sf.write("output_voice_design_2.wav", wavs[1], sr)
```

#### Voice Clone

For the voice clone model (`Qwen3-TTS-12Hz-1.7B/0.6B-Base`), to clone a voice and synthesize new content, you just need to provide a reference audio clip (`ref_audio`) along with its transcript (`ref_text`). `ref_audio` can be a local file path, a URL, a base64 string, or a `(numpy_array, sample_rate)` tuple. If you set `x_vector_only_mode=True`, only the speaker embedding is used so `ref_text` is not required, but cloning quality may be reduced.

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

ref_audio = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"
ref_text  = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."

wavs, sr = model.generate_voice_clone(
    text="I am solving the equation: x = [-b ± √(b²-4ac)] / 2a? Nobody can — it's a disaster (◍•͈⌔•͈◍), very sad!",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("output_voice_clone.wav", wavs[0], sr)
```

If you need to reuse the same reference prompt across multiple generations (to avoid recomputing prompt features), build it once with `create_voice_clone_prompt` and pass it via `voice_clone_prompt`.

```python
prompt_items = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=ref_text,
    x_vector_only_mode=False,
)
wavs, sr = model.generate_voice_clone(
    text=["Sentence A.", "Sentence B."],
    language=["English", "English"],
    voice_clone_prompt=prompt_items,
)
sf.write("output_voice_clone_1.wav", wavs[0], sr)
sf.write("output_voice_clone_2.wav", wavs[1], sr)
```

For more examples of reusable voice clone prompts, batch cloning, and batch inference, please refer to the [example codes](https://github.com/QwenLM/Qwen3-TTS/blob/main/examples/test_model_12hz_base.py). With those examples and the `generate_voice_clone` function description, you can explore more advanced usage patterns.

#### Voice Design then Clone

If you want a designed voice that you can reuse like a cloned speaker, a practical workflow is: (1) use the **VoiceDesign** model to synthesize a short reference clip that matches your target persona, (2) feed that clip into `create_voice_clone_prompt` to build a reusable prompt, and then (3) call `generate_voice_clone` with `voice_clone_prompt` to generate new content without re-extracting features every time. This is especially useful when you want a consistent character voice across many lines.

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# create a reference audio in the target style using the VoiceDesign model
design_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

ref_text = "H-hey! You dropped your... uh... calculus notebook? I mean, I think it's yours? Maybe?"
ref_instruct = "Male, 17 years old, tenor range, gaining confidence - deeper breath support now, though vowels still tighten when nervous"
ref_wavs, sr = design_model.generate_voice_design(
    text=ref_text,
    language="English",
    instruct=ref_instruct
)
sf.write("voice_design_reference.wav", ref_wavs[0], sr)

# build a reusable clone prompt from the voice design reference
clone_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

voice_clone_prompt = clone_model.create_voice_clone_prompt(
    ref_audio=(ref_wavs[0], sr),   # or "voice_design_reference.wav"
    ref_text=ref_text,
)

sentences = [
    "No problem! I actually... kinda finished those already? If you want to compare answers or something...",
    "What? No! I mean yes but not like... I just think you're... your titration technique is really precise!",
]

# reuse it for multiple single calls
wavs, sr = clone_model.generate_voice_clone(
    text=sentences[0],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write("clone_single_1.wav", wavs[0], sr)

wavs, sr = clone_model.generate_voice_clone(
    text=sentences[1],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write("clone_single_2.wav", wavs[0], sr)

# or batch generate in one call
wavs, sr = clone_model.generate_voice_clone(
    text=sentences,
    language=["English", "English"],
    voice_clone_prompt=voice_clone_prompt,
)
for i, w in enumerate(wavs):
    sf.write(f"clone_batch_{i}.wav", w, sr)
```

#### Tokenizer Encode and Decode

If you only want to encode and decode audio for transport or training and so on, `Qwen3TTSTokenizer` supports encode/decode with paths, URLs, numpy waveforms, and dict/list payloads, for example:

```python
import soundfile as sf
from qwen_tts import Qwen3TTSTokenizer

tokenizer = Qwen3TTSTokenizer.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    device_map="cuda:0",
)

enc = tokenizer.encode("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/tokenizer_demo_1.wav")
wavs, sr = tokenizer.decode(enc)
sf.write("decode_output.wav", wavs[0], sr)
```

For more tokenizer examples (including different input formats and batch usage), please refer to the [example codes](https://github.com/QwenLM/Qwen3-TTS/blob/main/examples/test_tokenizer_12hz.py). With those examples and the description for `Qwen3TTSTokenizer`, you can explore more advanced usage patterns.

### Launch Local Web UI Demo

To launch the Qwen3-TTS web ui demo, simply install the `qwen-tts` package and run `qwen-tts-demo`. Use the command below for help:

```bash
qwen-tts-demo --help
```

To launch the demo, you can use the following commands:

```bash
# CustomVoice model
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --ip 0.0.0.0 --port 8000
# VoiceDesign model
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign --ip 0.0.0.0 --port 8000
# Base model
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base --ip 0.0.0.0 --port 8000
```

And then open `http://<your-ip>:8000`, or access it via port forwarding in tools like VS Code.

#### Base Model HTTPS Notes

To avoid browser microphone permission issues after deploying the server, for Base model deployments, it is recommended/required to run the gradio service over **HTTPS** (especially when accessed remotely or behind modern browsers/gateways). Use `--ssl-certfile` and `--ssl-keyfile` to enable HTTPS. First we need to generate a private key and a self-signed cert (valid for 365 days):

```bash
openssl req -x509 -newkey rsa:2048 \
  -keyout key.pem -out cert.pem \
  -days 365 -nodes \
  -subj "/CN=localhost"
```

Then run the demo with HTTPS:

```bash
qwen-tts-demo Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --ip 0.0.0.0 --port 8000 \
  --ssl-certfile cert.pem \
  --ssl-keyfile key.pem \
  --no-ssl-verify
```

And open `https://<your-ip>:8000` to experience it. If your browser shows a warning, it’s expected for self-signed certificates. For production, use a real certificate.

### DashScope API Usage

To further explore Qwen3-TTS, we encourage you to try our DashScope API for a faster and more efficient experience. For detailed API information and documentation, please refer to the following:

| API Description | API Documentation (Mainland China) | API Documentation (International) |
|------------------|-----------------------------------|------------------------------------|
| Real-time API for Qwen3-TTS of custom voice model. | [https://help.aliyun.com/zh/model-studio/qwen-tts-realtime](https://help.aliyun.com/zh/model-studio/qwen-tts-realtime) | [https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-realtime) |
| Real-time API for Qwen3-TTS of voice clone model. | [https://help.aliyun.com/zh/model-studio/qwen-tts-voice-cloning](https://help.aliyun.com/zh/model-studio/qwen-tts-voice-cloning) | [https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-cloning](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-cloning) |
| Real-time API for Qwen3-TTS of voice design model. | [https://help.aliyun.com/zh/model-studio/qwen-tts-voice-design](https://help.aliyun.com/zh/model-studio/qwen-tts-voice-design) | [https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-design](https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-design) |


## vLLM Usage

vLLM officially provides day-0 support for Qwen3-TTS! Welcome to use vLLM-Omni for Qwen3-TTS deployment and inference. For installation and more details, please check [vLLM-Omni official documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/getting_started/quickstart/#installation). Now only offline inference is supported. Online serving will be supported later, and vLLM-Omni will continue to offer support and optimization for Qwen3-TTS in areas such as inference speed and streaming capabilities.

### Offline Inference
You can use vLLM-Omni to inference Qwen3-TTS locally, we provide examples in [vLLM-Omni repo](https://github.com/vllm-project/vllm-omni/tree/main/examples/offline_inference/qwen3_tts) which can generate audio output:
```bash
# git clone https://github.com/vllm-project/vllm-omni.git

# cd vllm-omni/examples/offline_inference/qwen3_tts

# Run a single sample with CustomVoice task
python end2end.py --query-type CustomVoice

# Batch sample (multiple prompts in one run) with CustomVoice task:
python end2end.py --query-type CustomVoice --use-batch-sample

# Run a single sample with VoiceDesign task
python end2end.py --query-type VoiceDesign

# Batch sample (multiple prompts in one run) with VoiceDesign task:
python end2end.py --query-type VoiceDesign --use-batch-sample

# Run a single sample with Base task in icl mode-tag
python end2end.py --query-type Base --mode-tag icl
```

## Fine Tuning

Please refer to [Qwen3-TTS-Finetuning](finetuning/) for detailed instructions on fine-tuning Qwen3-TTS.

## Evaluation

During evaluation, we ran inference for all models with `dtype=torch.bfloat16` and set `max_new_tokens=2048`. All other sampling parameters used the defaults from the checkpoint’s `generate_config.json`. For the Seed-Test and InstructTTS-Eval test sets, we set `language="auto"`, while for all other test sets we explicitly passed the corresponding `language`. The detailed results are shown below.


<details>
<summary>Speech Generation Benchmarks</summary>

*Zero-shot speech generation on the Seed-TTS test set. Performance is measured by Word Error Rate (WER, ↓), where lower is better.*

<table>
  <thead>
    <tr>
      <th style="text-align: center;">Datasets</th>
      <th style="text-align: left;">Model</th>
      <th colspan="2" style="text-align: center;">Performance</th>
    </tr>
    <tr style="border-bottom: 1px solid #ddd; border-top: 1px solid #ddd;">
      <td colspan="4" style="text-align: center;"><em>Content Consistency</em></td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="14" style="text-align: center; vertical-align: middle;">SEED<br><em>test-zh</em> | <em>test-en</em></td>
      <td style="text-align: left;">Seed-TTS (Anastassiou et al., 2024)</td>
      <td style="text-align: center;">1.12</td>
      <td style="text-align: center;">2.25</td>
    </tr>
    <tr>
      <td style="text-align: left;">MaskGCT (Wang et al., 2024)</td>
      <td style="text-align: center;">2.27</td>
      <td style="text-align: center;">2.62</td>
    </tr>
    <tr>
      <td style="text-align: left;">E2 TTS (Eskimez et al., 2024)</td>
      <td style="text-align: center;">1.97</td>
      <td style="text-align: center;">2.19</td>
    </tr>
    <tr>
      <td style="text-align: left;">F5-TTS (Chen et al., 2024)</td>
      <td style="text-align: center;">1.56</td>
      <td style="text-align: center;">1.83</td>
    </tr>
    <tr>
      <td style="text-align: left;">Spark TTS (Wang et al., 2025)</td>
      <td style="text-align: center;">1.20</td>
      <td style="text-align: center;">1.98</td>
    </tr>
    <tr>
      <td style="text-align: left;">Llasa-8B (Ye et al., 2025b)</td>
      <td style="text-align: center;">1.59</td>
      <td style="text-align: center;">2.97</td>
    </tr>
    <tr>
      <td style="text-align: left;">KALL-E (Xia et al., 2024)</td>
      <td style="text-align: center;">0.96</td>
      <td style="text-align: center;">1.94</td>
    </tr>
    <tr>
      <td style="text-align: left;">FireRedTTS 2 (Xie et al., 2025)</td>
      <td style="text-align: center;">1.14</td>
      <td style="text-align: center;">1.95</td>
    </tr>
    <tr>
      <td style="text-align: left;">CosyVoice 3 (Du et al., 2025)</td>
      <td style="text-align: center;"><strong>0.71</strong></td>
      <td style="text-align: center;">1.45</td>
    </tr>
    <tr>
      <td style="text-align: left;">MiniMax-Speech (Zhang et al., 2025a)</td>
      <td style="text-align: center;">0.83</td>
      <td style="text-align: center;">1.65</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3-TTS-25Hz-0.6B-Base</td>
      <td style="text-align: center;">1.18</td>
      <td style="text-align: center;">1.64</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3-TTS-25Hz-1.7B-Base</td>
      <td style="text-align: center;">1.10</td>
      <td style="text-align: center;">1.49</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3-TTS-12Hz-0.6B-Base</td>
      <td style="text-align: center;">0.92</td>
      <td style="text-align: center;">1.32</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3-TTS-12Hz-1.7B-Base</td>
      <td style="text-align: center;">0.77</td>
      <td style="text-align: center;"><strong>1.24</strong></td>
    </tr>
  </tbody>
</table>

<br>

*Multilingual speech generation on the TTS multilingual test set. Performance is measured by Word Error Rate (WER, ↓) for content consistency and Cosine Similarity (SIM, ↑) for speaker similarity.*

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align: left; vertical-align: bottom;">Language</th>
      <th colspan="2" style="text-align: center;">Qwen3-TTS-25Hz</th>
      <th colspan="2" style="text-align: center;">Qwen3-TTS-12Hz</th>
      <th rowspan="2" style="text-align: center; vertical-align: bottom;">MiniMax</th>
      <th rowspan="2" style="text-align: center; vertical-align: bottom;">ElevenLabs</th>
    </tr>
    <tr>
      <th style="text-align: center;">0.6B-Base</th>
      <th style="text-align: center;">1.7B-Base</th>
      <th style="text-align: center;">0.6B-Base</th>
      <th style="text-align: center;">1.7B-Base</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="7" style="text-align: center; border-top: 1px solid #ddd; border-bottom: 1px solid #ddd;"><em>Content Consistency</em></td>
    </tr>
    <tr>
      <td style="text-align: left;">Chinese</td>
      <td style="text-align: center;">1.108</td>
      <td style="text-align: center;"><strong>0.777</strong></td>
      <td style="text-align: center;">1.145</td>
      <td style="text-align: center;">0.928</td>
      <td style="text-align: center;">2.252</td>
      <td style="text-align: center;">16.026</td>
    </tr>
    <tr>
      <td style="text-align: left;">English</td>
      <td style="text-align: center;">1.048</td>
      <td style="text-align: center;">1.014</td>
      <td style="text-align: center;"><strong>0.836</strong></td>
      <td style="text-align: center;">0.934</td>
      <td style="text-align: center;">2.164</td>
      <td style="text-align: center;">2.339</td>
    </tr>
    <tr>
      <td style="text-align: left;">German</td>
      <td style="text-align: center;">1.501</td>
      <td style="text-align: center;">0.960</td>
      <td style="text-align: center;">1.089</td>
      <td style="text-align: center;">1.235</td>
      <td style="text-align: center;">1.906</td>
      <td style="text-align: center;"><strong>0.572</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Italian</td>
      <td style="text-align: center;">1.169</td>
      <td style="text-align: center;">1.105</td>
      <td style="text-align: center;">1.534</td>
      <td style="text-align: center;"><strong>0.948</strong></td>
      <td style="text-align: center;">1.543</td>
      <td style="text-align: center;">1.743</td>
    </tr>
    <tr>
      <td style="text-align: left;">Portuguese</td>
      <td style="text-align: center;">2.046</td>
      <td style="text-align: center;">1.778</td>
      <td style="text-align: center;">2.254</td>
      <td style="text-align: center;">1.526</td>
      <td style="text-align: center;">1.877</td>
      <td style="text-align: center;"><strong>1.331</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Spanish</td>
      <td style="text-align: center;">2.031</td>
      <td style="text-align: center;">1.491</td>
      <td style="text-align: center;">1.491</td>
      <td style="text-align: center;">1.126</td>
      <td style="text-align: center;"><strong>1.029</strong></td>
      <td style="text-align: center;">1.084</td>
    </tr>
    <tr>
      <td style="text-align: left;">Japanese</td>
      <td style="text-align: center;">4.189</td>
      <td style="text-align: center;">5.121</td>
      <td style="text-align: center;">6.404</td>
      <td style="text-align: center;">3.823</td>
      <td style="text-align: center;"><strong>3.519</strong></td>
      <td style="text-align: center;">10.646</td>
    </tr>
    <tr>
      <td style="text-align: left;">Korean</td>
      <td style="text-align: center;">2.852</td>
      <td style="text-align: center;">2.631</td>
      <td style="text-align: center;"><strong>1.741</strong></td>
      <td style="text-align: center;">1.755</td>
      <td style="text-align: center;">1.747</td>
      <td style="text-align: center;">1.865</td>
    </tr>
    <tr>
      <td style="text-align: left;">French</td>
      <td style="text-align: center;">2.852</td>
      <td style="text-align: center;"><strong>2.631</strong></td>
      <td style="text-align: center;">2.931</td>
      <td style="text-align: center;">2.858</td>
      <td style="text-align: center;">4.099</td>
      <td style="text-align: center;">5.216</td>
    </tr>
    <tr>
      <td style="text-align: left;">Russian</td>
      <td style="text-align: center;">5.957</td>
      <td style="text-align: center;">4.535</td>
      <td style="text-align: center;">4.458</td>
      <td style="text-align: center;"><strong>3.212</strong></td>
      <td style="text-align: center;">4.281</td>
      <td style="text-align: center;">3.878</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td colspan="7" style="text-align: center; border-bottom: 1px solid #ddd;"><em>Speaker Similarity</em></td>
    </tr>
    <tr>
      <td style="text-align: left;">Chinese</td>
      <td style="text-align: center;">0.797</td>
      <td style="text-align: center;">0.796</td>
      <td style="text-align: center;"><strong>0.811</strong></td>
      <td style="text-align: center;">0.799</td>
      <td style="text-align: center;">0.780</td>
      <td style="text-align: center;">0.677</td>
    </tr>
    <tr>
      <td style="text-align: left;">English</td>
      <td style="text-align: center;">0.811</td>
      <td style="text-align: center;">0.815</td>
      <td style="text-align: center;"><strong>0.829</strong></td>
      <td style="text-align: center;">0.775</td>
      <td style="text-align: center;">0.756</td>
      <td style="text-align: center;">0.613</td>
    </tr>
    <tr>
      <td style="text-align: left;">German</td>
      <td style="text-align: center;">0.749</td>
      <td style="text-align: center;">0.737</td>
      <td style="text-align: center;">0.769</td>
      <td style="text-align: center;"><strong>0.775</strong></td>
      <td style="text-align: center;">0.733</td>
      <td style="text-align: center;">0.614</td>
    </tr>
    <tr>
      <td style="text-align: left;">Italian</td>
      <td style="text-align: center;">0.722</td>
      <td style="text-align: center;">0.718</td>
      <td style="text-align: center;">0.792</td>
      <td style="text-align: center;"><strong>0.817</strong></td>
      <td style="text-align: center;">0.699</td>
      <td style="text-align: center;">0.579</td>
    </tr>
    <tr>
      <td style="text-align: left;">Portuguese</td>
      <td style="text-align: center;">0.790</td>
      <td style="text-align: center;">0.783</td>
      <td style="text-align: center;">0.794</td>
      <td style="text-align: center;"><strong>0.817</strong></td>
      <td style="text-align: center;">0.805</td>
      <td style="text-align: center;">0.711</td>
    </tr>
    <tr>
      <td style="text-align: left;">Spanish</td>
      <td style="text-align: center;">0.732</td>
      <td style="text-align: center;">0.731</td>
      <td style="text-align: center;">0.812</td>
      <td style="text-align: center;"><strong>0.814</strong></td>
      <td style="text-align: center;">0.762</td>
      <td style="text-align: center;">0.615</td>
    </tr>
    <tr>
      <td style="text-align: left;">Japanese</td>
      <td style="text-align: center;"><strong>0.810</strong></td>
      <td style="text-align: center;">0.807</td>
      <td style="text-align: center;">0.798</td>
      <td style="text-align: center;">0.788</td>
      <td style="text-align: center;">0.776</td>
      <td style="text-align: center;">0.738</td>
    </tr>
    <tr>
      <td style="text-align: left;">Korean</td>
      <td style="text-align: center;"><strong>0.824</strong></td>
      <td style="text-align: center;">0.814</td>
      <td style="text-align: center;">0.812</td>
      <td style="text-align: center;">0.799</td>
      <td style="text-align: center;">0.779</td>
      <td style="text-align: center;">0.700</td>
    </tr>
    <tr>
      <td style="text-align: left;">French</td>
      <td style="text-align: center;">0.698</td>
      <td style="text-align: center;">0.703</td>
      <td style="text-align: center;">0.700</td>
      <td style="text-align: center;"><strong>0.714</strong></td>
      <td style="text-align: center;">0.628</td>
      <td style="text-align: center;">0.535</td>
    </tr>
    <tr>
      <td style="text-align: left;">Russian</td>
      <td style="text-align: center;">0.734</td>
      <td style="text-align: center;">0.744</td>
      <td style="text-align: center;">0.781</td>
      <td style="text-align: center;"><strong>0.792</strong></td>
      <td style="text-align: center;">0.761</td>
      <td style="text-align: center;">0.676</td>
    </tr>
  </tbody>
</table>

<br>

*Cross-lingual speech generation on the Cross-Lingual benchmark. Performance is measured by Mixed Error Rate (WER for English, CER for others, ↓).*

<table>
  <thead>
    <tr>
      <th style="text-align: left;">Task</th>
      <th style="text-align: center;">Qwen3-TTS-25Hz-1.7B-Base</th>
      <th style="text-align: center;">Qwen3-TTS-12Hz-1.7B-Base</th>
      <th style="text-align: center;">CosyVoice3</th>
      <th style="text-align: center;">CosyVoice2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">en-to-zh</td>
      <td style="text-align: center;">5.66</td>
      <td style="text-align: center;"><strong>4.77</strong></td>
      <td style="text-align: center;">5.09</td>
      <td style="text-align: center;">13.5</td>
    </tr>
    <tr>
      <td style="text-align: left;">ja-to-zh</td>
      <td style="text-align: center;">3.92</td>
      <td style="text-align: center;">3.43</td>
      <td style="text-align: center;"><strong>3.05</strong></td>
      <td style="text-align: center;">48.1</td>
    </tr>
    <tr>
      <td style="text-align: left;">ko-to-zh</td>
      <td style="text-align: center;">1.14</td>
      <td style="text-align: center;">1.08</td>
      <td style="text-align: center;"><strong>1.06</strong></td>
      <td style="text-align: center;">7.70</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td style="text-align: left;">zh-to-en</td>
      <td style="text-align: center;">2.91</td>
      <td style="text-align: center;"><strong>2.77</strong></td>
      <td style="text-align: center;">2.98</td>
      <td style="text-align: center;">6.47</td>
    </tr>
    <tr>
      <td style="text-align: left;">ja-to-en</td>
      <td style="text-align: center;">3.95</td>
      <td style="text-align: center;"><strong>3.04</strong></td>
      <td style="text-align: center;">4.20</td>
      <td style="text-align: center;">17.1</td>
    </tr>
    <tr>
      <td style="text-align: left;">ko-to-en</td>
      <td style="text-align: center;">3.48</td>
      <td style="text-align: center;"><strong>3.09</strong></td>
      <td style="text-align: center;">4.19</td>
      <td style="text-align: center;">11.2</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td style="text-align: left;">zh-to-ja</td>
      <td style="text-align: center;">9.29</td>
      <td style="text-align: center;">8.40</td>
      <td style="text-align: center;"><strong>7.08</strong></td>
      <td style="text-align: center;">13.1</td>
    </tr>
    <tr>
      <td style="text-align: left;">en-to-ja</td>
      <td style="text-align: center;">7.74</td>
      <td style="text-align: center;">7.21</td>
      <td style="text-align: center;"><strong>6.80</strong></td>
      <td style="text-align: center;">14.9</td>
    </tr>
    <tr>
      <td style="text-align: left;">ko-to-ja</td>
      <td style="text-align: center;">4.17</td>
      <td style="text-align: center;"><strong>3.67</strong></td>
      <td style="text-align: center;">3.93</td>
      <td style="text-align: center;">5.86</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td style="text-align: left;">zh-to-ko</td>
      <td style="text-align: center;">8.12</td>
      <td style="text-align: center;"><strong>4.82</strong></td>
      <td style="text-align: center;">14.4</td>
      <td style="text-align: center;">24.8</td>
    </tr>
    <tr>
      <td style="text-align: left;">en-to-ko</td>
      <td style="text-align: center;">6.83</td>
      <td style="text-align: center;"><strong>5.14</strong></td>
      <td style="text-align: center;">5.87</td>
      <td style="text-align: center;">21.9</td>
    </tr>
    <tr>
      <td style="text-align: left;">ja-to-ko</td>
      <td style="text-align: center;">6.86</td>
      <td style="text-align: center;"><strong>5.59</strong></td>
      <td style="text-align: center;">7.92</td>
      <td style="text-align: center;">21.5</td>
    </tr>
  </tbody>
</table>

<br>

*Controllable speech generation on InstructTTSEval. Performance is measured by Attribute Perception and Synthesis accuracy (APS), Description-Speech Consistency (DSD), and Response Precision (RP).*

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align: left; vertical-align: bottom;">Type</th>
      <th rowspan="2" style="text-align: left; vertical-align: bottom;">Model</th>
      <th colspan="3" style="text-align: center;">InstructTTSEval-ZH</th>
      <th colspan="3" style="text-align: center;">InstructTTSEval-EN</th>
    </tr>
    <tr>
      <th style="text-align: center;">APS (↑)</th>
      <th style="text-align: center;">DSD (↑)</th>
      <th style="text-align: center;">RP (↑)</th>
      <th style="text-align: center;">APS (↑)</th>
      <th style="text-align: center;">DSD (↑)</th>
      <th style="text-align: center;">RP (↑)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" style="text-align: left; vertical-align: middle;"><em>Target<br>Speaker</em></td>
      <td style="text-align: left;">Gemini-flash</td>
      <td style="text-align: center;">88.2</td>
      <td style="text-align: center;"><strong>90.9</strong></td>
      <td style="text-align: center;"><strong>77.3</strong></td>
      <td style="text-align: center;"><strong>92.3</strong></td>
      <td style="text-align: center;"><strong>93.8</strong></td>
      <td style="text-align: center;"><strong>80.1</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Gemini-pro</td>
      <td style="text-align: center;"><strong>89.0</strong></td>
      <td style="text-align: center;">90.1</td>
      <td style="text-align: center;">75.5</td>
      <td style="text-align: center;">87.6</td>
      <td style="text-align: center;">86.0</td>
      <td style="text-align: center;">67.2</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3TTS-25Hz-1.7B-CustomVoice</td>
      <td style="text-align: center;">83.1</td>
      <td style="text-align: center;">75.0</td>
      <td style="text-align: center;">63.0</td>
      <td style="text-align: center;">79.0</td>
      <td style="text-align: center;">82.8</td>
      <td style="text-align: center;">69.3</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3TTS-12Hz-1.7B-CustomVoice</td>
      <td style="text-align: center;">83.0</td>
      <td style="text-align: center;">77.8</td>
      <td style="text-align: center;">61.2</td>
      <td style="text-align: center;">77.3</td>
      <td style="text-align: center;">77.1</td>
      <td style="text-align: center;">63.7</td>
    </tr>
    <tr>
      <td style="text-align: left;">GPT-4o-mini-tts</td>
      <td style="text-align: center;">54.9</td>
      <td style="text-align: center;">52.3</td>
      <td style="text-align: center;">46.0</td>
      <td style="text-align: center;">76.4</td>
      <td style="text-align: center;">74.3</td>
      <td style="text-align: center;">54.8</td>
    </tr>
    <tr style="border-top: 1px solid #ddd;">
      <td rowspan="9" style="text-align: left; vertical-align: middle;"><em>Voice<br>Design</em></td>
      <td style="text-align: left;">Qwen3TTS-12Hz-1.7B-VD</td>
      <td style="text-align: center;"><strong>85.2</strong></td>
      <td style="text-align: center;"><strong>81.1</strong></td>
      <td style="text-align: center;"><strong>65.1</strong></td>
      <td style="text-align: center;">82.9</td>
      <td style="text-align: center;"><strong>82.4</strong></td>
      <td style="text-align: center;"><strong>68.4</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Mimo-Audio-7B-Instruct (Zhang et al., 2025b)</td>
      <td style="text-align: center;">75.7</td>
      <td style="text-align: center;">74.3</td>
      <td style="text-align: center;">61.5</td>
      <td style="text-align: center;">80.6</td>
      <td style="text-align: center;">77.6</td>
      <td style="text-align: center;">59.5</td>
    </tr>
    <tr>
      <td style="text-align: left;">VoiceSculptor (Hu et al., 2026)</td>
      <td style="text-align: center;">75.7</td>
      <td style="text-align: center;">64.7</td>
      <td style="text-align: center;">61.5</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
    </tr>
    <tr>
      <td style="text-align: left;">Hume</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;"><strong>83.0</strong></td>
      <td style="text-align: center;">75.3</td>
      <td style="text-align: center;">54.3</td>
    </tr>
    <tr>
      <td style="text-align: left;">VoxInstruct (Zhou et al., 2024)</td>
      <td style="text-align: center;">47.5</td>
      <td style="text-align: center;">52.3</td>
      <td style="text-align: center;">42.6</td>
      <td style="text-align: center;">54.9</td>
      <td style="text-align: center;">57.0</td>
      <td style="text-align: center;">39.3</td>
    </tr>
    <tr>
      <td style="text-align: left;">Parler-tts-mini (Lyth & King, 2024)</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">63.4</td>
      <td style="text-align: center;">48.7</td>
      <td style="text-align: center;">28.6</td>
    </tr>
    <tr>
      <td style="text-align: left;">Parler-tts-large (Lyth & King, 2024)</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">60.0</td>
      <td style="text-align: center;">45.9</td>
      <td style="text-align: center;">31.2</td>
    </tr>
    <tr>
      <td style="text-align: left;">PromptTTS (Guo et al., 2023)</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">64.3</td>
      <td style="text-align: center;">47.2</td>
      <td style="text-align: center;">31.4</td>
    </tr>
    <tr>
      <td style="text-align: left;">PromptStyle (Liu et al., 2023)</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">57.4</td>
      <td style="text-align: center;">46.4</td>
      <td style="text-align: center;">30.9</td>
    </tr>
  </tbody>
</table>

<br>

*Target-Speaker Multilingual Speech Generation on the TTS multilingual test set. Performance is measured by Word Error Rate (WER, ↓).*

<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align: left; vertical-align: bottom;">Language</th>
      <th colspan="2" style="text-align: center;">Qwen3-TTS-25Hz</th>
      <th colspan="2" style="text-align: center;">Qwen3-TTS-12Hz</th>
      <th rowspan="2" style="text-align: center; vertical-align: bottom;">GPT-4o-Audio<br>Preview</th>
    </tr>
    <tr>
      <th style="text-align: center;">0.6B-CustomVoice</th>
      <th style="text-align: center;">1.7B-CustomVoice</th>
      <th style="text-align: center;">0.6B-CustomVoice</th>
      <th style="text-align: center;">1.7B-CustomVoice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">Chinese</td>
      <td style="text-align: center;">0.874</td>
      <td style="text-align: center;"><strong>0.708</strong></td>
      <td style="text-align: center;">0.944</td>
      <td style="text-align: center;">0.903</td>
      <td style="text-align: center;">3.519</td>
    </tr>
    <tr>
      <td style="text-align: left;">English</td>
      <td style="text-align: center;">1.332</td>
      <td style="text-align: center;">0.936</td>
      <td style="text-align: center;">1.188</td>
      <td style="text-align: center;"><strong>0.899</strong></td>
      <td style="text-align: center;">2.197</td>
    </tr>
    <tr>
      <td style="text-align: left;">German</td>
      <td style="text-align: center;">0.990</td>
      <td style="text-align: center;"><strong>0.634</strong></td>
      <td style="text-align: center;">2.722</td>
      <td style="text-align: center;">1.057</td>
      <td style="text-align: center;">1.161</td>
    </tr>
    <tr>
      <td style="text-align: left;">Italian</td>
      <td style="text-align: center;">1.861</td>
      <td style="text-align: center;">1.271</td>
      <td style="text-align: center;">2.545</td>
      <td style="text-align: center;">1.362</td>
      <td style="text-align: center;"><strong>1.194</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Portuguese</td>
      <td style="text-align: center;">1.728</td>
      <td style="text-align: center;">1.854</td>
      <td style="text-align: center;">3.219</td>
      <td style="text-align: center;">2.681</td>
      <td style="text-align: center;"><strong>1.504</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Spanish</td>
      <td style="text-align: center;">1.309</td>
      <td style="text-align: center;">1.284</td>
      <td style="text-align: center;"><strong>1.154</strong></td>
      <td style="text-align: center;">1.330</td>
      <td style="text-align: center;">4.000</td>
    </tr>
    <tr>
      <td style="text-align: left;">Japanese</td>
      <td style="text-align: center;"><strong>3.875</strong></td>
      <td style="text-align: center;">4.518</td>
      <td style="text-align: center;">6.877</td>
      <td style="text-align: center;">4.924</td>
      <td style="text-align: center;">5.001</td>
    </tr>
    <tr>
      <td style="text-align: left;">Korean</td>
      <td style="text-align: center;">2.202</td>
      <td style="text-align: center;">2.274</td>
      <td style="text-align: center;">3.053</td>
      <td style="text-align: center;"><strong>1.741</strong></td>
      <td style="text-align: center;">2.763</td>
    </tr>
    <tr>
      <td style="text-align: left;">French</td>
      <td style="text-align: center;">3.865</td>
      <td style="text-align: center;"><strong>3.080</strong></td>
      <td style="text-align: center;">3.841</td>
      <td style="text-align: center;">3.781</td>
      <td style="text-align: center;">3.605</td>
    </tr>
    <tr>
      <td style="text-align: left;">Russian</td>
      <td style="text-align: center;">6.529</td>
      <td style="text-align: center;"><strong>4.444</strong></td>
      <td style="text-align: center;">5.809</td>
      <td style="text-align: center;">4.734</td>
      <td style="text-align: center;">5.250</td>
    </tr>
  </tbody>
</table>

<br>

*Long speech generation results. Performance is measured by Word Error Rate (WER, ↓).*

<table>
  <thead>
    <tr>
      <th style="text-align: center;">Datasets</th>
      <th style="text-align: left;">Model</th>
      <th colspan="2" style="text-align: center;">Performance</th>
    </tr>
    <tr style="border-bottom: 1px solid #ddd; border-top: 1px solid #ddd;">
      <td colspan="4" style="text-align: center;"><em>Content Consistency</em></td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5" style="text-align: center; vertical-align: middle;"><em>long-zh</em> | <em>long-en</em></td>
      <td style="text-align: left;">Higgs-Audio-v2 (chunk) (Boson AI, 2025)</td>
      <td style="text-align: center;">5.505</td>
      <td style="text-align: center;">6.917</td>
    </tr>
    <tr>
      <td style="text-align: left;">VibeVoice (Peng et al., 2025)</td>
      <td style="text-align: center;">22.619</td>
      <td style="text-align: center;">1.780</td>
    </tr>
    <tr>
      <td style="text-align: left;">VoxCPM (Zhou et al., 2025)</td>
      <td style="text-align: center;">4.835</td>
      <td style="text-align: center;">7.474</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3-TTS-25Hz-1.7B-CustomVoice</td>
      <td style="text-align: center;"><strong>1.517</strong></td>
      <td style="text-align: center;"><strong>1.225</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen3-TTS-12Hz-1.7B-CustomVoice</td>
      <td style="text-align: center;">2.356</td>
      <td style="text-align: center;">2.812</td>
    </tr>
  </tbody>
</table>
</details>


<details>
<summary>Speech Tokenizer Benchmarks</summary>

*Comparison between different supervised semantic speech tokenizers on ASR Task.*

<table>
  <thead>
    <tr>
      <th style="text-align: left;">Model</th>
      <th style="text-align: center;">Codebook Size</th>
      <th style="text-align: center;">FPS</th>
      <th style="text-align: center;">C.V. EN</th>
      <th style="text-align: center;">C.V. CN</th>
      <th style="text-align: center;">Fluers EN</th>
      <th style="text-align: center;">Fluers CN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">S3 Tokenizer(VQ) (Du et al., 2024a)</td>
      <td style="text-align: center;">4096</td>
      <td style="text-align: center;">50</td>
      <td style="text-align: center;">12.06</td>
      <td style="text-align: center;">15.38</td>
      <td style="text-align: center;">-</td>
      <td style="text-align: center;">-</td>
    </tr>
    <tr>
      <td style="text-align: left;">S3 Tokenizer(VQ) (Du et al., 2024a)</td>
      <td style="text-align: center;">4096</td>
      <td style="text-align: center;">25</td>
      <td style="text-align: center;">11.56</td>
      <td style="text-align: center;">18.26</td>
      <td style="text-align: center;">7.65</td>
      <td style="text-align: center;">5.03</td>
    </tr>
    <tr>
      <td style="text-align: left;">S3 Tokenizer(FSQ) (Du et al., 2024a)</td>
      <td style="text-align: center;">6561</td>
      <td style="text-align: center;">25</td>
      <td style="text-align: center;">10.67</td>
      <td style="text-align: center;"><strong>7.29</strong></td>
      <td style="text-align: center;">6.58</td>
      <td style="text-align: center;">4.43</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen-TTS-Tokenizer-25Hz (Stage 1)</td>
      <td style="text-align: center;">32768</td>
      <td style="text-align: center;">25</td>
      <td style="text-align: center;"><strong>7.51</strong></td>
      <td style="text-align: center;">10.73</td>
      <td style="text-align: center;"><strong>3.07</strong></td>
      <td style="text-align: center;"><strong>4.23</strong></td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen-TTS-Tokenizer-25Hz (Stage 2)</td>
      <td style="text-align: center;">32768</td>
      <td style="text-align: center;">25</td>
      <td style="text-align: center;">10.40</td>
      <td style="text-align: center;">14.99</td>
      <td style="text-align: center;">4.14</td>
      <td style="text-align: center;">4.67</td>
    </tr>
  </tbody>
</table>

<br>

*Comparison between different semantic-related speech tokenizers.*

<table>
  <thead>
    <tr>
      <th style="text-align: left;">Model</th>
      <th style="text-align: center;">NQ</th>
      <th style="text-align: center;">Codebook Size</th>
      <th style="text-align: center;">FPS</th>
      <th style="text-align: center;">PESQ_WB</th>
      <th style="text-align: center;">PESQ_NB</th>
      <th style="text-align: center;">STOI</th>
      <th style="text-align: center;">UTMOS</th>
      <th style="text-align: center;">SIM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left;">SpeechTokenizer (Zhang et al., 2023a)</td>
      <td style="text-align: center;">8</td>
      <td style="text-align: center;">1024</td>
      <td style="text-align: center;">50</td>
      <td style="text-align: center;">2.60</td>
      <td style="text-align: center;">3.05</td>
      <td style="text-align: center;">0.92</td>
      <td style="text-align: center;">3.90</td>
      <td style="text-align: center;">0.85</td>
    </tr>
    <tr>
      <td style="text-align: left;">X-codec (Ye et al., 2025a)</td>
      <td style="text-align: center;">2</td>
      <td style="text-align: center;">1024</td>
      <td style="text-align: center;">50</td>
      <td style="text-align: center;">2.68</td>
      <td style="text-align: center;">3.27</td>
      <td style="text-align: center;">0.86</td>
      <td style="text-align: center;">4.11</td>
      <td style="text-align: center;">0.84</td>
    </tr>
    <tr>
      <td style="text-align: left;">X-codec 2 (Ye et al., 2025b)</td>
      <td style="text-align: center;">1</td>
      <td style="text-align: center;">65536</td>
      <td style="text-align: center;">50</td>
      <td style="text-align: center;">2.43</td>
      <td style="text-align: center;">3.04</td>
      <td style="text-align: center;">0.92</td>
      <td style="text-align: center;">4.13</td>
      <td style="text-align: center;">0.82</td>
    </tr>
    <tr>
      <td style="text-align: left;">XY-Tokenizer (Gong et al., 2025)</td>
      <td style="text-align: center;">8</td>
      <td style="text-align: center;">1024</td>
      <td style="text-align: center;">12.5</td>
      <td style="text-align: center;">2.41</td>
      <td style="text-align: center;">3.00</td>
      <td style="text-align: center;">0.91</td>
      <td style="text-align: center;">3.98</td>
      <td style="text-align: center;">0.83</td>
    </tr>
    <tr>
      <td style="text-align: left;">Mimi (Défossez et al., 2024)</td>
      <td style="text-align: center;">16</td>
      <td style="text-align: center;">2048</td>
      <td style="text-align: center;">12.5</td>
      <td style="text-align: center;">2.88</td>
      <td style="text-align: center;">3.42</td>
      <td style="text-align: center;">0.94</td>
      <td style="text-align: center;">3.87</td>
      <td style="text-align: center;">0.87</td>
    </tr>
    <tr>
      <td style="text-align: left;">FireredTTS 2 Tokenizer (Xie et al., 2025)</td>
      <td style="text-align: center;">16</td>
      <td style="text-align: center;">2048</td>
      <td style="text-align: center;">12.5</td>
      <td style="text-align: center;">2.73</td>
      <td style="text-align: center;">3.28</td>
      <td style="text-align: center;">0.94</td>
      <td style="text-align: center;">3.88</td>
      <td style="text-align: center;">0.87</td>
    </tr>
    <tr>
      <td style="text-align: left;">Qwen-TTS-Tokenizer-12Hz</td>
      <td style="text-align: center;">16</td>
      <td style="text-align: center;">2048</td>
      <td style="text-align: center;">12.5</td>
      <td style="text-align: center;"><strong>3.21</strong></td>
      <td style="text-align: center;"><strong>3.68</strong></td>
      <td style="text-align: center;"><strong>0.96</strong></td>
      <td style="text-align: center;"><strong>4.16</strong></td>
      <td style="text-align: center;"><strong>0.95</strong></td>
    </tr>
  </tbody>
</table>

</details>


## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=QwenLM/Qwen3-TTS&type=Date)](https://star-history.com/#QwenLM/Qwen3-TTS&Date)


<br>