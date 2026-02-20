# Hausa TTS Training with sft_12hz.py

This guide shows how to train Qwen3-TTS on the Hausa TTS dataset using the existing `sft_12hz.py` training script.

## Overview

The training pipeline consists of two steps:

1. **Data Preparation**: Load Hausa dataset from Hugging Face, process audio codes, and save to JSONL format
2. **Training**: Use `sft_12hz.py` to train the model on the prepared data

## Files

- `prepare_hausa_data.py` - Script to prepare Hausa TTS data
- `train_hausa_sft.py` - Main training script that orchestrates data preparation and training
- `Qwen3-TTS/finetuning/sft_12hz.py` - Original training script (used as-is)

## Quick Start

### Option 1: One-Command Training

```bash
cd Qwen3-TTS-finetuning

python train_hausa_sft.py \
    --train_jsonl ./data/hausa_train.jsonl \
    --output_model_path ./hausa_output \
    --batch_size 2 \
    --lr 2e-5 \
    --num_epochs 3 \
    --speaker_name hausa_speaker
```

### Option 2: Step-by-Step

#### Step 1: Prepare Training Data

```bash
python prepare_hausa_data.py \
    --split train \
    --output_jsonl ./data/hausa_train.jsonl \
    --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --device cuda
```

#### Step 2: Train Model

```bash
python Qwen3-TTS/finetuning/sft_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --output_model_path ./hausa_output \
    --train_jsonl ./data/hausa_train.jsonl \
    --batch_size 2 \
    --lr 2e-5 \
    --num_epochs 3 \
    --speaker_name hausa_speaker
```

## Command Line Arguments

### Data Preparation (`prepare_hausa_data.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--split` | str | `train` | Dataset split (train, validation, test) |
| `--output_jsonl` | str | required | Output JSONL file path |
| `--model_path` | str | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Path to Qwen3-TTS model |
| `--ref_audio_path` | str | `voices/english_voice/english_voice.wav` | Reference audio path |
| `--ref_text` | str | MTN Entertainment... | Reference text |
| `--max_samples` | int | None | Maximum samples to process |
| `--device` | str | `cuda` | Device (cuda/cpu) |

### Training Pipeline (`train_hausa_sft.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--train_jsonl` | str | `./data/hausa_train.jsonl` | Training data JSONL |
| `--validation_jsonl` | str | None | Validation data JSONL (optional) |
| `--ref_audio_path` | str | `voices/english_voice/english_voice.wav` | Reference audio path |
| `--ref_text` | str | MTN Entertainment... | Reference text |
| `--max_train_samples` | int | None | Max training samples |
| `--max_eval_samples` | int | None | Max evaluation samples |
| `--device` | str | `cuda` | Device for data prep |
| `--init_model_path` | str | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Initial model path |
| `--output_model_path` | str | `./output` | Output directory |
| `--batch_size` | int | `2` | Batch size |
| `--lr` | float | `2e-5` | Learning rate |
| `--num_epochs` | int | `3` | Number of epochs |
| `--speaker_name` | str | `hausa_speaker` | Speaker name |
| `--skip_prepare` | flag | False | Skip data preparation |
| `--prepare_only` | flag | False | Only prepare data |

### Training (`sft_12hz.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--init_model_path` | str | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Initial model path |
| `--output_model_path` | str | `./output` | Output directory |
| `--train_jsonl` | str | required | Training data JSONL |
| `--batch_size` | int | `2` | Batch size |
| `--lr` | float | `2e-5` | Learning rate |
| `--num_epochs` | int | `3` | Number of epochs |
| `--speaker_name` | str | `speaker_test` | Speaker name |

## Examples

### Debug Mode (Small Dataset)

```bash
python train_hausa_sft.py \
    --train_jsonl ./data/hausa_train_debug.jsonl \
    --output_model_path ./hausa_output_debug \
    --max_train_samples 100 \
    --batch_size 2 \
    --num_epochs 1
```

### Prepare All Splits

```bash
# Train split
python prepare_hausa_data.py \
    --split train \
    --output_jsonl ./data/hausa_train.jsonl

# Validation split
python prepare_hausa_data.py \
    --split validation \
    --output_jsonl ./data/hausa_val.jsonl

# Test split
python prepare_hausa_data.py \
    --split test \
    --output_jsonl ./data/hausa_test.jsonl
```

### Custom Reference Audio and Text

```bash
python train_hausa_sft.py \
    --train_jsonl ./data/hausa_train.jsonl \
    --ref_audio_path ./my_reference.wav \
    --ref_text "This is my custom reference text for voice cloning." \
    --output_model_path ./hausa_output \
    --num_epochs 5
```

### Skip Data Preparation (Already Prepared)

```bash
python train_hausa_sft.py \
    --train_jsonl ./data/hausa_train.jsonl \
    --output_model_path ./hausa_output \
    --skip_prepare \
    --num_epochs 3
```

### Prepare Data Only (No Training)

```bash
python train_hausa_sft.py \
    --train_jsonl ./data/hausa_train.jsonl \
    --prepare_only
```

## Output Structure

After training, the output directory will contain:

```
hausa_output/
├── checkpoint-epoch-0/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files...
├── checkpoint-epoch-1/
├── checkpoint-epoch-2/
└── ...
```

Each checkpoint contains:
- `config.json`: Model configuration with speaker settings
- `model.safetensors`: Model weights
- Tokenizer and processor files from the original model

## Using the Trained Model

```python
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# Load the fine-tuned model
model = Qwen3TTSModel.from_pretrained("./hausa_output/checkpoint-epoch-2")

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

## Data Format

The prepared JSONL files contain one line per sample:

```json
{
  "audio": "sample_0.wav",
  "text": "Sample text in Hausa",
  "audio_codes": [[...], [...], ...],  # 16 layers of codec codes
  "language": "ha",
  "ref_audio": "voices/english_voice/english_voice.wav",
  "ref_text": "MTN Entertainment and Lifestyle...",
  "speaker_id": "speaker_001",
  "gender": "male",
  "age_range": "25-35",
  "phase": "train"
}
```

## Troubleshooting

### Out of Memory During Data Preparation
- Reduce `--max_samples` for testing
- Use `--device cpu` if GPU memory is limited

### Out of Memory During Training
- Reduce `--batch_size`
- Use gradient accumulation (modify `sft_12hz.py`)

### Slow Data Preparation
- The first run loads the model, which takes time
- Subsequent runs are faster if model is cached

### Dataset Not Found
- Ensure you have internet connection
- Check that `vaghawan/hausa-tts-22k` exists on Hugging Face

### Reference Audio Not Found
- Check the path to `voices/english_voice/english_voice.wav`
- Provide absolute path if needed

## Notes

- The reference audio is used for speaker embedding extraction
- The reference text is not directly used in `sft_12hz.py` but is included in the data for future use
- Audio codes are computed using the model's codec encoder
- The training script `sft_12hz.py` is used as-is without modifications

## License

Apache License 2.0
