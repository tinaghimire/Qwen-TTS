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
    einops sox onnxruntime datasets>=2.14.0

# Install flash-attn (optional, for faster inference)
pip install flash-attn --no-build-isolation
```

## 🚀 Quick Start

### 1. Start the WebSocket Server

```bash
# Default settings (port 8000, model size 1.7B)
uv run qwen_tts_server.py

# Custom settings
uv run qwen_tts_server.py --host 0.0.0.0 --port 8001 --model-size 0.6B
```

### 2. Test the Server

```bash
# Run all tests
uv run test_qwen_websocket.py --voice english_voice

# Run specific test
uv run test_qwen_websocket.py --voice english_voice --test basic
```

### 3. Train on Hausa TTS Data

```bash
# Simple training (recommended for beginners)
python train_simple.py --batch_size 2 --lr 2e-5 --num_epochs 3

# Advanced training (with validation and WandB)
python train_advanced.py --batch_size 2 --lr 2e-5 --num_epochs 3 --use_wandb
```

## 🌐 WebSocket Server

### Voice Setup

Create a directory for each voice in the `voices/` folder:

```bash
mkdir -p voices/english_voice
cp /path/to/reference.wav voices/english_voice/
```

**Supported audio formats**: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`, `.aac`

### Endpoints

#### 1. Standard WebSocket Endpoint

```
ws://localhost:8000/ws/voice-clone/{voice_id}
```

**Protocol:**
```json
// Client sends config (optional)
{"type": "config", "language": "Auto", "use_xvector_only": false, "model_size": "1.7B"}

// Client sends text to synthesize
{"type": "text", "text": "Hello world", "ref_text": "optional reference text"}

// Client sends end signal
{"type": "end"}

// Server responds with audio
{
  "type": "audio",
  "audio": "<base64_wav>",
  "sample_rate": 24000,
  "generation_time": 1.23,
  "audio_duration": 2.45,
  "real_time_factor": 1.99
}
```

#### 2. ElevenLabs-Compatible Endpoint

```
ws://localhost:8000/v1/text-to-speech/{voice_id}/stream-input
```

**Protocol:**
```json
// Initialization
{"text": " ", "generation_config": {"language": "Auto", "use_xvector_only": true}}

// Stream text
{"text": "Hello world."}

// Close
{"text": ""}

// Server response
{"audio": "<base64_pcm_wav>", "isFinal": false}
// Final
{"isFinal": true}
```

#### 3. HTTP Endpoint

```
POST /voice-clone/{voice_id}
Content-Type: application/json

{
  "target_text": "Hello world",
  "ref_text": "optional reference text",
  "language": "Auto",
  "use_xvector_only": false,
  "model_size": "1.7B"
}
```

**Response:** WAV audio file with timing headers:
- `X-Generation-Time`: Time taken to generate audio (seconds)
- `X-Audio-Duration`: Duration of generated audio (seconds)
- `X-Sample-Rate`: Sample rate (Hz)

#### 4. List Voices Endpoint

```
GET /voices
```

**Response:**
```json
{
  "voices": [
    {"id": "english_voice", "audio_file": "english_voice/english_voice.wav"}
  ],
  "count": 1
}
```

## 🎯 Training Pipeline

### Overview

Train Qwen3-TTS on the Hausa TTS dataset from Hugging Face for voice cloning in Hausa language.

### Dataset

- **Source**: `vaghawan/hausa-tts-22k`
- **Splits**: train, validation, test
- **Features**: audio, text, speaker_id, language, gender, age_range, phase

### Training Options

#### Option 1: Simple Training (Beginner Friendly)

Uses `sft_12hz.py` directly - perfect for quick experiments.

```bash
# Train with default settings
python train_simple.py

# Custom settings
python train_simple.py --batch_size 4 --lr 1e-5 --num_epochs 5

# Skip data preparation if already done
python train_simple.py --skip_prepare

# Only prepare data, don't train
python train_simple.py --prepare_only
```

#### Option 2: Advanced Training (Production Ready)

Includes validation, metrics, WandB logging, and model checkpointing.

```bash
# Train with default settings (includes WandB)
python train_advanced.py

# Train with custom settings
python train_advanced.py --batch_size 4 --lr 1e-5 --num_epochs 5 --use_wandb

# Train without WandB
python train_advanced.py --use_wandb False

# Upload models to Hugging Face Hub
python train_advanced.py --upload_to_hub --hub_token YOUR_TOKEN

# Skip data preparation if already done
python train_advanced.py --skip_prepare
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

### Command Line Arguments

#### Dataset Tool (`dataset_tool.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_name` | str | `vaghawan/hausa-tts-22k` | Hugging Face dataset name |
| `--split` | str | `train` | Dataset split (train, validation, test) |
| `--output_jsonl` | str | `./data/{split}.jsonl` | Output JSONL file path |
| `--model_path` | str | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Path to Qwen3-TTS model |
| `--ref_audio_path` | str | `voices/english_voice/english_voice.wav` | Reference audio path |
| `--ref_text` | str | MTN Entertainment... | Reference text |
| `--max_samples` | int | None | Maximum samples to process |
| `--device` | str | `cuda` | Device (cuda/cpu) |
| `--info` | str | None | Get info about JSONL dataset |

#### Simple Training (`train_simple.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_name` | str | `vaghawan/hausa-tts-22k` | Hugging Face dataset name |
| `--train_jsonl` | str | `./data/train.jsonl` | Training data JSONL |
| `--validation_jsonl` | str | `None` | Validation data JSONL |
| `--init_model_path` | str | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Initial model path |
| `--output_model_path` | str | `./output` | Output directory |
| `--batch_size` | int | `2` | Batch size |
| `--lr` | float | `2e-5` | Learning rate |
| `--num_epochs` | int | `3` | Number of epochs |
| `--speaker_name` | str | `hausa_speaker` | Speaker name |
| `--max_train_samples` | int | None | Max training samples |
| `--max_eval_samples` | int | None | Max evaluation samples |
| `--skip_prepare` | flag | False | Skip data preparation |
| `--prepare_only` | flag | False | Only prepare data |

#### Advanced Training (`train_advanced.py`)

All simple training options plus:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--gradient_accumulation_steps` | int | `4` | Gradient accumulation steps |
| `--weight_decay` | float | `0.01` | Weight decay |
| `--warmup_steps` | int | `100` | Warmup steps |
| `--max_grad_norm` | float | `1.0` | Max gradient norm |
| `--logging_steps` | int | `10` | Logging frequency |
| `--save_steps` | int | `500` | Checkpoint save frequency |
| `--eval_steps` | int | `500` | Evaluation frequency |
| `--use_wandb` | flag | True | Use WandB logging |
| `--wandb_project` | str | `qwen3-tts-hausa` | WandB project name |
| `--wandb_run_name` | str | None | WandB run name |
| `--upload_to_hub` | flag | False | Upload to Hugging Face |
| `--hub_model_id_best` | str | `vaghawan/tts-best` | Best model repo ID |
| `--hub_model_id_last` | str | `vaghawan/tts-last` | Last model repo ID |
| `--hub_token` | str | None | Hugging Face token |
| `--mixed_precision` | str | `bf16` | Mixed precision mode |

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

## 📚 API Documentation

Once the server is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🧪 Testing

### Run All Tests

```bash
uv run test_qwen_websocket.py --voice english_voice
```

### Run Specific Tests

```bash
# Basic connection test
uv run test_qwen_websocket.py --voice english_voice --test basic

# Streaming test
uv run test_qwen_websocket.py --voice english_voice --test streaming

# Audio similarity test
uv run test_qwen_websocket.py --voice english_voice --test similarity

# ElevenLabs-compatible test
uv run test_qwen_websocket.py --voice english_voice --test elevenlabs
```

### Test Output

All test outputs are saved to the `test_output/` directory:

- `test1_basic_output.wav` - Basic test output
- `test2_chunk_*.wav` - Streaming test chunks
- `test3_generated.wav` - Audio similarity test output
- `test4_el_chunk_*.wav` - ElevenLabs-compatible test chunks
- `test_results.json` - Comprehensive test results

### Understanding Test Results

#### Timing Statistics

- **Generation Time**: Time taken to generate audio
- **Audio Duration**: Length of generated audio
- **Real-Time Factor (RTF)**: `audio_duration / generation_time`
  - RTF > 1.0: Faster than real-time (good)
  - RTF < 1.0: Slower than real-time

#### Audio Similarity Scores

- **MFCC Similarity**: Based on Mel-frequency cepstral coefficients
- **Spectral Similarity**: Based on chroma spectral features

Both scores range from 0 (dissimilar) to 1 (identical).

Quality interpretation:
- **Excellent**: > 0.7
- **Good**: 0.5 - 0.7
- **Fair**: 0.3 - 0.5
- **Poor**: < 0.3

## 🔍 Troubleshooting

### Server won't start
- Check if port is already in use: `lsof -i :8000`
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`

### WebSocket connection fails
- Verify server is running: `curl http://localhost:8000/health`
- Check firewall settings
- Ensure voice directory exists in `voices/` directory
- List available voices: `curl http://localhost:8000/voices`

### Audio generation errors
- Check reference audio format (should be WAV, mono)
- Verify reference text matches audio content (for ICL mode)
- Check GPU memory: `nvidia-smi`

### Training issues
- **Out of Memory**: Reduce `--batch_size` or use `--max_samples` for testing
- **Slow Training**: Increase `--batch_size` if memory allows
- **Dataset Not Found**: Ensure you have internet connection and `vaghawan/hausa-tts-22k` exists on Hugging Face

### Similarity scores are low
- Ensure reference audio is clear and has good quality
- Use longer reference audio (5-10 seconds recommended)
- Check that reference text accurately transcribes the audio

## 📄 License

This code follows the same license as Qwen3-TTS (Apache-2.0).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions, please open an issue on GitHub.