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
| `REF_TEXT` | `""` | Reference transcription (for ICL) |
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

The `dataset_tool.py` provides a unified interface to load data directly from HuggingFace and create PyTorch DataLoaders for training. **No intermediate JSONL files are needed!**

### Get Dataset Information

```bash
# Get info about a dataset (without loading it)
python dataset_tool.py --action info

# Get info for a specific split
python dataset_tool.py --action info --split validation
```

### Test DataLoaders

```bash
# Test training dataloader
python dataset_tool.py --action train --batch_size 4

# Test evaluation dataloader
python dataset_tool.py --action eval --batch_size 4

# Limit number of samples for testing
python dataset_tool.py --action train --batch_size 2 --max_samples 10

# Use multiple workers (for faster loading)
python dataset_tool.py --action train --batch_size 4 --num_workers 4
```

### Use Dataset Tool in Python

```python
from dataset_tool import (
    get_train_dataloader,
    get_eval_dataloader,
    get_dataloader,
    QwenTTSDataset,
)

# Get training dataloader with defaults from .env
train_loader = get_train_dataloader(batch_size=4, num_workers=4)

# Get evaluation dataloader
val_loader = get_eval_dataloader(batch_size=4, num_workers=2)

# Create custom dataloader
custom_loader = get_dataloader(
    dataset_name="vaghawan/hausa-tts-22k",
    split="train",
    batch_size=8,
    max_samples=1000,
    shuffle=True,
)

# Iterate through batches
for batch in train_loader:
    # batch is a list of samples
    for sample in batch:
        audio_array = sample["audio"]
        text = sample["text"]
        audio_codes = sample["audio_codes"]
        sr = sample["sr"]
        # ... process for training

# Create PyTorch Dataset (no DataLoader)
dataset = QwenTTSDataset(
    dataset_name="vaghawan/hausa-tts-22k",
    split="train",
    max_samples=1000
)
print(f"Dataset size: {len(dataset)}")

# Get single sample
sample = dataset[0]
print(f"Text: {sample['text']}")
print(f"Audio codes shape: {len(sample['audio_codes'])}")
```

### DataLoader Output Format

Each sample in the batch contains:

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

## Training Scripts

### 1. Simple Training (`train_using_sft.py`)

Direct training pipeline that loads data from HuggingFace on-the-fly. Perfect for quick experiments.

**Features:**
- Loads data directly from HuggingFace (no intermediate JSONL files)
- On-the-fly audio processing
- Uses PyTorch DataLoader for efficient batch loading
- Minimal configuration required
- Fast setup with no data preparation step

**Usage:**
```bash
# Train with default settings
python train_using_sft.py

# Test DataLoader creation only
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