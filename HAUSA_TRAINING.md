# Hausa TTS Fine-tuning with Qwen3-TTS

This guide explains how to fine-tune Qwen3-TTS on the Hausa TTS dataset from Hugging Face.

## Overview

The training setup includes:
- **Dataset**: `vaghawan/hausa-tts-22k` - A Hausa TTS dataset with train/validation/test splits
- **Reference Audio**: `voices/english_voice/english_voice.wav` - Used for voice cloning
- **Training Script**: `Qwen3-TTS/finetuning/train_hausa.py` - Flexible trainer with evaluation
- **Dataset Class**: `Qwen3-TTS/finetuning/hausa_dataset.py` - Custom dataset for Hausa data

## Features

✅ **Flexible Training**
- Trainer class with training and evaluation loops
- Gradient accumulation for larger effective batch sizes
- Learning rate scheduling with warmup
- Mixed precision training (BF16/FP16)

✅ **Monitoring & Logging**
- Weights & Biases (WandB) integration
- TensorBoard support via Accelerate
- Progress bars with tqdm

✅ **Model Management**
- Automatic checkpoint saving
- Best model tracking based on validation loss
- Full training state saving (optimizer, scheduler, global step)
- Upload to Hugging Face Hub

✅ **Evaluation**
- Validation loss computation
- Periodic evaluation during training
- Best model selection

## Prerequisites

```bash
# Install required packages
pip install torch transformers accelerate datasets librosa safetensors wandb huggingface_hub tqdm
```

## Dataset Structure

The Hausa TTS dataset contains:
- `audio`: Audio waveform and sampling rate
- `text`: Transcription text
- `speaker_id`: Speaker identifier
- `language`: Language code (ha for Hausa)
- `gender`: Speaker gender
- `age_range`: Speaker age range
- `phase`: Dataset split (train/validation/test)

## Quick Start

### 1. Basic Training

```bash
cd Qwen3-TTS-finetuning

python Qwen3-TTS/finetuning/train_hausa.py \
    --output_dir ./hausa_output \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5
```

### 2. Training with Custom Settings

```bash
python Qwen3-TTS/finetuning/train_hausa.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --output_dir ./hausa_output \
    --ref_audio_path voices/english_voice/english_voice.wav \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --num_epochs 5 \
    --warmup_steps 200 \
    --speaker_name hausa_speaker \
    --logging_steps 10 \
    --save_steps 500 \
    --eval_steps 500 \
    --use_wandb \
    --wandb_project qwen3-tts-hausa \
    --wandb_run_name hausa_experiment_1 \
    --upload_to_hub \
    --hub_model_id_best vaghawan/tts-best \
    --hub_model_id_last vaghawan/tts-last
```

### 3. Debug Mode (Limited Samples)

```bash
python Qwen3-TTS/finetuning/train_hausa.py \
    --output_dir ./hausa_output_debug \
    --max_train_samples 100 \
    --max_eval_samples 20 \
    --num_epochs 1 \
    --batch_size 2
```

## Command Line Arguments

### Model & Data Paths
- `--init_model_path`: Path to pre-trained model (default: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`)
- `--output_dir`: Output directory for checkpoints (default: `./output`)
- `--ref_audio_path`: Path to reference audio (default: `voices/english_voice/english_voice.wav`)

### Training Hyperparameters
- `--batch_size`: Batch size per device (default: `2`)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: `4`)
- `--learning_rate`: Learning rate (default: `2e-5`)
- `--num_epochs`: Number of training epochs (default: `3`)
- `--weight_decay`: Weight decay (default: `0.01`)
- `--warmup_steps`: Number of warmup steps (default: `100`)
- `--max_grad_norm`: Maximum gradient norm for clipping (default: `1.0`)

### Dataset Settings
- `--train_split`: Training split name (default: `train`)
- `--validation_split`: Validation split name (default: `validation`)
- `--test_split`: Test split name (default: `test`)
- `--max_train_samples`: Maximum training samples (for debugging)
- `--max_eval_samples`: Maximum evaluation samples (for debugging)

### Speaker Settings
- `--speaker_name`: Speaker name for the model (default: `hausa_speaker`)

### Logging & Checkpointing
- `--logging_steps`: Log every N steps (default: `10`)
- `--save_steps`: Save checkpoint every N steps (default: `500`)
- `--eval_steps`: Evaluate every N steps (default: `500`)
- `--save_total_limit`: Maximum number of checkpoints to keep (default: `3`)

### WandB Settings
- `--use_wandb`: Enable WandB logging (default: `True`)
- `--wandb_project`: WandB project name (default: `qwen3-tts-hausa`)
- `--wandb_run_name`: WandB run name (optional)

### Hugging Face Upload Settings
- `--upload_to_hub`: Upload models to Hugging Face Hub (default: `True`)
- `--hub_model_id_best`: Hub repository ID for best model (default: `vaghawan/tts-best`)
- `--hub_model_id_last`: Hub repository ID for last model (default: `vaghawan/tts-last`)
- `--hub_token`: Hugging Face API token (optional, can be set via `HF_TOKEN` env var)

### Mixed Precision
- `--mixed_precision`: Mixed precision mode (default: `bf16`, options: `no`, `fp16`, `bf16`)

## Output Structure

After training, the output directory will contain:

```
hausa_output/
├── best/                          # Best model (lowest validation loss)
│   ├── config.json
│   ├── model.safetensors
│   ├── training_state.pt          # Optimizer, scheduler, training state
│   └── tokenizer files...
├── last/                          # Last checkpoint (for resuming training)
│   ├── config.json
│   ├── model.safetensors
│   ├── training_state.pt
│   └── tokenizer files...
├── checkpoint-500/                # Intermediate checkpoints
├── checkpoint-1000/
├── epoch-1/
├── epoch-2/
└── epoch-3/
```

## Resuming Training

To resume training from a checkpoint:

```python
# The training_state.pt file contains:
# - optimizer_state_dict: Optimizer state
# - scheduler_state_dict: Scheduler state
# - global_step: Current global step
# - best_eval_loss: Best validation loss seen so far

# You can modify the training script to load these states
```

## Using the Trained Model

After training, you can use the model for inference:

```python
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# Load the fine-tuned model
model = Qwen3TTSModel.from_pretrained("./hausa_output/best")

# Generate speech
text = "Hello, this is a test."
ref_audio = "voices/english_voice/english_voice.wav"

output = model.generate(
    text=text,
    reference_audio=ref_audio,
    language="ha"  # Hausa
)

# Save output
import soundfile as sf
sf.write("output.wav", output.audio, output.sampling_rate)
```

## Hugging Face Hub Upload

The training script automatically uploads models to Hugging Face Hub:

### Best Model (`vaghawan/tts-best`)
- Contains the model with lowest validation loss
- Includes all files needed for inference
- Suitable for production use

### Last Model (`vaghawan/tts-last`)
- Contains the final checkpoint after all epochs
- Includes `training_state.pt` with optimizer and scheduler states
- Suitable for resuming training

### Setting Up Hugging Face Token

```bash
# Option 1: Set environment variable
export HF_TOKEN="your_token_here"

# Option 2: Pass as argument
python train_hausa.py --hub_token "your_token_here"

# Option 3: Login via CLI
huggingface-cli login
```

## Monitoring Training

### WandB Dashboard

If `--use_wandb` is enabled, you can monitor training at:
```
https://wandb.ai/<your_username>/qwen3-tts-hausa
```

Metrics logged:
- `train/loss`: Training loss per step
- `train/avg_loss`: Average training loss
- `train/learning_rate`: Current learning rate
- `eval/loss`: Validation loss
- `eval/epoch_loss`: Validation loss at end of epoch

### TensorBoard

```bash
tensorboard --logdir ./hausa_output
```

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Increase `--gradient_accumulation_steps`
- Use `--mixed_precision bf16` or `--mixed_precision fp16`

### Slow Training
- Increase `--batch_size` if memory allows
- Reduce `--logging_steps` and `--eval_steps`
- Use multiple GPUs with Accelerate

### Dataset Issues
- Check that the reference audio path is correct
- Verify the dataset splits exist in the Hugging Face dataset
- Use `--max_train_samples` for quick testing

### Upload Issues
- Verify your Hugging Face token has write permissions
- Check that the repository names are correct
- Ensure you have internet connectivity

## Advanced Usage

### Custom Dataset

You can create a custom dataset by extending `HausaTTSDataset`:

```python
from hausa_dataset import HausaTTSDataset

class CustomDataset(HausaTTSDataset):
    def _prepare_data_list(self):
        # Custom data preparation logic
        pass
```

### Custom Loss Function

Modify the `compute_loss` method in `HausaTTSTrainer`:

```python
def compute_loss(self, batch):
    # Your custom loss computation
    loss = ...
    return loss
```

### Custom Scheduler

Change the scheduler in `HausaTTSTrainer.__init__`:

```python
from torch.optim.lr_scheduler import OneCycleLR

self.scheduler = OneCycleLR(
    self.optimizer,
    max_lr=args.learning_rate,
    total_steps=num_training_steps
)
```

## Citation

If you use this code, please cite:

```bibtex
@software{qwen3_tts_2026,
  title={Qwen3-TTS: A High-Quality Text-to-Speech System},
  author={Qwen Team},
  year={2026},
  publisher={Alibaba Cloud}
}
```

## License

Apache License 2.0

## Contact

For issues and questions, please open an issue on GitHub.
