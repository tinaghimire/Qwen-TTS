# Qwen3-TTS Training Pipeline

Comprehensive training pipeline for Qwen3-TTS voice cloning with Hausa TTS fine-tuning. This project provides two training pipelines and a unified dataset tool.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Environment Configuration](#environment-configuration)
- [Project Structure](#project-structure)
- [Dataset Tool](#dataset-tool)
- [Training Scripts](#training-scripts)
- [Training Workflows](#training-workflows)
- [Monitoring Training](#monitoring-training)
- [Using the Trained Model](#using-the-trained-model)
- [Architecture Overview](#architecture-overview)
- [Troubleshooting](#troubleshooting)

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
```

## Quick Start

```bash
# Create .env from template
cp .env.training.example .env

# Run simple training
python train_using_sft.py

# OR run advanced training with validation
python train_wandb_validation.py
```

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
| `TRAIN_JSONL` | `./data/hausa_train.jsonl` | Training data |
| `VALIDATION_JSONL` | `./data/hausa_validation.jsonl` | Validation data |

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
| `REF_TEXT` | `""` | Reference transcription (for ICL) |
| `SPEAKER_NAME` | `reference_speaker` | Speaker name |

### Dataset Limits

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_TRAIN_SAMPLES` | - | Max training samples (empty = all) |
| `MAX_EVAL_SAMPLES` | - | Max evaluation samples (empty = all) |

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
| `SKIP_PREPARE` | `false` | Skip data preparation |
| `PREPARE_ONLY` | `false` | Only prepare data, don't train |

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
├── dataset_tool.py               # Unified dataset loading and preparation tool
├── train_using_sft.py            # Simple training pipeline (uses sft_12hz.py)
├── train_wandb_validation.py     # Advanced training pipeline (with validation, metrics, WandB)
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
        └── sft_12hz.py            # Base training script
```

## Dataset Tool

The `dataset_tool.py` provides a unified interface for dataset loading and preparation from Hugging Face datasets.

### Prepare Dataset from Hugging Face

```bash
# Prepare training data (reads config from .env)
python dataset_tool.py

# Or with explicit parameters
python dataset_tool.py --dataset_name vaghawan/hausa-tts-22k --split train --output_jsonl ./data/train.jsonl

# Prepare validation data
python dataset_tool.py --split validation --output_jsonl ./data/validation.jsonl

# Limit number of samples (for debugging)
python dataset_tool.py --max_samples 100

# Use CPU instead of GPU
python dataset_tool.py --device cpu
```

### Get Dataset Information

```bash
# Get info about a prepared dataset
python dataset_tool.py --info ./data/train.jsonl
```

### Use Dataset Tool in Python

```python
from dataset_tool import prepare_dataset, load_jsonl_dataset, get_dataset_info

# Prepare dataset from Hugging Face
prepare_dataset(
    dataset_name="vaghawan/hausa-tts-22k",
    split="train",
    output_jsonl="./data/train.jsonl",
    max_samples=1000
)

# Load dataset from JSONL
dataset = load_jsonl_dataset("./data/train.jsonl")
print(f"Loaded {len(dataset)} samples")

# Get dataset information
info = get_dataset_info("./data/train.jsonl")
print(f"Languages: {info['languages']}")
print(f"Speakers: {info['speakers']}")
```

### Dataset Tool Output Format

Each prepared sample in the JSONL file contains:

```json
{
  "audio": "sample_0.wav",
  "text": "This is the target text to synthesize.",
  "audio_codes": [[...], [...], ...],
  "ref_audio": "voices/speaker/reference.wav",
  "ref_text": "This is the reference audio content.",
  "language": "ha",
  "speaker_id": "speaker_001",
  "gender": "male",
  "age_range": "25-35",
  "phase": "unknown"
}
```

## Training Scripts

### 1. Simple Training (`train_using_sft.py`)

Simplest training pipeline using `sft_12hz.py` directly. Perfect for quick experiments.

**Features:**
- Uses base `sft_12hz.py` script
- Minimal configuration required
- Fast setup
- Good for prototyping

**Usage:**
```bash
# Train with default settings
python train_using_sft.py

# Skip data preparation if already done
echo "SKIP_PREPARE=true" >> .env
python train_using_sft.py

# Only prepare data, don't train
echo "PREPARE_ONLY=true" >> .env
python train_using_sft.py
```

**How it works:**
1. Calls `dataset_tool.py` to prepare training data
2. Calls `Qwen3-TTS/finetuning/sft_12hz.py` with parameters from `.env`
3. Saves model checkpoints to `OUTPUT_MODEL_PATH`

### 2. Advanced Training (`train_wandb_validation.py`)

Comprehensive training pipeline with validation, metrics, WandB logging, and model checkpointing.

**Features:**
- Validation during training
- WandB logging for detailed metrics
- Checkpoint saving with optimizer and scheduler states
- Best model tracking based on validation loss
- Model upload to Hugging Face Hub
- Mixed precision training support (BF16)
- Gradient accumulation for larger effective batch sizes

**Usage:**
```bash
# Train with default settings (includes WandB)
python train_wandb_validation.py

# Skip data preparation if already done
echo "SKIP_PREPARE=true" >> .env
python train_wandb_validation.py

# Disable WandB
echo "USE_WANDB=false" >> .env
python train_wandb_validation.py

# Enable model upload
echo "UPLOAD_TO_HUB=true" >> .env
echo "HF_TOKEN=hf_your_token_here" >> .env
python train_wandb_validation.py
```

**How it works:**
1. (Optional) Calls `dataset_tool.py` to prepare training and validation data
2. Initializes WandB tracker (if enabled)
3. Loads model and datasets
4. Sets up optimizer, scheduler, and accelerator
5. Trains model with validation at regular intervals
6. Saves best model (lowest validation loss) and last checkpoint
7. (Optional) Uploads models to Hugging Face Hub
8. Ends WandB tracking

## Training Workflows

### Workflow 1: Simple Training - Quick Start

Best for quick experiments and debugging.

```bash
# Step 1: Create .env from template
cp .env.training.example .env

# Step 2: Verify setup
python test_setup.py

# Step 3: Train with defaults
python train_using_sft.py

# Output: ./output/checkpoint-epoch-*/
```

### Workflow 2: Advanced Training - Production

Best for production models with proper validation and monitoring.

```bash
# Step 1: Create .env from template
cp .env.training.example .env
nano .env  # Edit configuration

# Step 2: Verify setup
python test_setup.py

# Step 3: Train
python train_wandb_validation.py

# Output: ./output/best/, ./output/last/
```

### Workflow 3: Debug with Small Dataset

Best for debugging code and setup issues.

```bash
# Step 1: Prepare small dataset
python dataset_tool.py --max_samples 10

# Step 2: Train for 1 epoch
python train_using_sft.py --num_epochs 1
```

### Workflow 4: Resume from Checkpoint

```bash
# Training automatically saves checkpoints
# To resume, modify output path or use existing checkpoint
# Currently, resuming requires manual intervention
```

## Monitoring Training

### Console Output

Both training scripts output progress to the console:

```
Epoch 0 | Step 10 | Loss: 2.3456
Epoch 0 | Step 20 | Loss: 2.1234
```

### WandB Dashboard (Advanced Training)

For advanced training, WandB provides comprehensive monitoring:

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
- Gradient norms
- Checkpoint information
- Model comparison

### Output Structure

#### Simple Training Output

```
output/
├── checkpoint-epoch-0/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
├── checkpoint-epoch-1/
├── checkpoint-epoch-2/
```

#### Advanced Training Output

```
output/
├── best/                          # Best model (lowest validation loss)
│   ├── config.json
│   ├── model.safetensors
│   └── training_state.pt         # Optimizer and scheduler states
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

## License

Apache-2.0 (same as Qwen3-TTS)

## Additional Resources

- [Qwen3-TTS Documentation](./Qwen3-TTS/README.md)
- [Hugging Face Dataset](https://huggingface.co/datasets/vaghawan/hausa-tts-22k)
- [WandB Documentation](https://docs.wandb.ai/)
- [Qwen3-TTS Models](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)