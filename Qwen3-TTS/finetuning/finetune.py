#!/usr/bin/env python3
# coding=utf-8
"""
Unified Training Script for Qwen3-TTS Fine-tuning.

Data flow (only supported path):
1. Upload data once: python data_processing.py prepare --speaker both --upload --repo_id "vaghawan/qwen3-tts-multi-speaker"
2. Training loads from HuggingFace (vaghawan/qwen3-tts-multi-speaker) by subset/split.
   ref_audio comes from voices/ (hausa_speaker.wav, english_speaker.wav); audio_codes from tokenizer on the fly.
3. Combined train/val dataloaders via get_multispeaker_finetune_dataloader (batch- and CUDA-friendly).

Multi-GPU: accelerate launch --num_processes=N train.py
"""

import json
import os
import subprocess
import sys
import datetime
import multiprocessing
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, fields

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from datasets import load_dataset
from huggingface_hub import HfApi, login
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoConfig, get_cosine_schedule_with_warmup
from tqdm import tqdm
from dotenv import load_dotenv
import wandb

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    

# Add paths: ensure project root is on path so "finetuning" and "qwen_tts" are importable
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts import Qwen3TTSTokenizer
from safetensors.torch import save_file
from finetuning.dataset import TTSDataset
from finetuning.data_processing import SPEAKER_LANGUAGE
from finetuning.layer_utils import replace_and_add_layers, print_model_summary, get_trainable_params
from finetuning.dual_loss_trainer import DualLossTrainer
from finetuning.quality_metrics import EVALUATION_METRICS, QualityMetricsCalculator
from transformers import AutoProcessor
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

# Load environment variables
load_dotenv(override=True)

# Global ref_audio and ref_mel per speaker (loaded once, used for all batches)
REF_AUDIO_CACHE = {}
REF_MEL_CACHE = {}
# Device cache: (tuple(speaker_names), device_str) -> ref_mels tensor to avoid repeated CPU->GPU transfer
REF_MEL_DEVICE_CACHE = {}


def _voices_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voices")


def _get_speech_tokenizer_codebook_size(speech_tokenizer) -> Optional[int]:
    """Return the codebook size of the speech tokenizer (12Hz or 25Hz) for clamping predicted codes."""
    if speech_tokenizer is None:
        return None
    inner = getattr(speech_tokenizer, "model", None)
    if inner is None:
        return None
    config = getattr(inner, "config", None)
    if config is None:
        return None
    # 12Hz: decoder_config.codebook_size (default 2048)
    dec = getattr(config, "decoder_config", None)
    if dec is not None and hasattr(dec, "codebook_size"):
        return int(dec.codebook_size)
    # 25Hz: audio_vq_codebook_size (default 32768)
    if hasattr(config, "audio_vq_codebook_size"):
        return int(config.audio_vq_codebook_size)
    # Fallback for 12Hz when decoder_config is not populated
    if getattr(config, "model_type", "") == "qwen3_tts_tokenizer_12hz":
        return 2048
    return None


def _training_summary(config: "TrainingConfig", train_dataloader: Any) -> None:
    """Print data used and training duration summary at startup."""
    import math
    speakers = [s.strip() for s in config.train_speakers.split(",") if s.strip()]
    if not speakers:
        speakers = ["hausa_speaker", "english_speaker"]
    n_gpus = int(os.getenv("WORLD_SIZE", "1"))
    if n_gpus < 1:
        n_gpus = 1
    effective_batch = config.train_batch_size * n_gpus * config.gradient_accumulation_steps

    max_train = config.max_train_samples
    max_val = config.max_eval_samples
    try:
        steps_per_epoch = len(train_dataloader)
        train_samples_used = steps_per_epoch * config.train_batch_size * n_gpus
    except TypeError:
        steps_per_epoch = (
            math.ceil(max_train / (config.train_batch_size * n_gpus))
            if (max_train and config.train_batch_size)
            else None
        )
        train_samples_used = max_train if max_train else "full dataset"

    total_steps = (steps_per_epoch * config.num_epochs) if steps_per_epoch else None

    print("\n" + "=" * 60)
    print("Data used & training duration")
    print("=" * 60)
    print(f"  HF dataset:        {config.hf_dataset_repo}")
    print(f"  Speakers:          {', '.join(speakers)}")
    print(f"  Train samples:     {train_samples_used} (cap: {max_train or 'none'})")
    print(f"  Val samples cap:   {max_val or 'none'}")
    print(f"  Batch size:        {config.train_batch_size} per GPU × {n_gpus} GPU(s) × grad_accum {config.gradient_accumulation_steps} = {effective_batch} effective")
    if steps_per_epoch is not None:
        print(f"  Steps per epoch:   {steps_per_epoch}")
        print(f"  Epochs:            {config.num_epochs}")
        if total_steps is not None:
            print(f"  Total steps:       {total_steps}")
    print()
    print("  Recommendation:    Train 3–5 epochs for this setup. With 150k train")
    print("                     samples and batch 128, expect ~1.2k steps/epoch (~6k steps")
    print("                     for 5 epochs). Monitor validation loss; stop early if")
    print("                     it plateaus or increases.")
    print("=" * 60 + "\n")


def _ensure_config_has_speaker_languages(config_obj: Any, train_speakers_str: str) -> None:
    """Ensure talker_config.codec_language_id has entries for each speaker's language (e.g. hausa, english)
    so training uses correct language conditioning. Adds missing languages using english's token id."""
    speakers = [s.strip() for s in (train_speakers_str or "").split(",") if s.strip()]
    if not speakers:
        return
    languages = list({SPEAKER_LANGUAGE.get(s, "english").strip().lower() for s in speakers})
    tc = getattr(config_obj, "talker_config", None)
    if tc is None:
        return
    codec_lang = getattr(tc, "codec_language_id", None) if not isinstance(tc, dict) else tc.get("codec_language_id")
    if not isinstance(codec_lang, dict):
        codec_lang = dict(codec_lang) if codec_lang else {}
    english_id = codec_lang.get("english")
    if english_id is None and codec_lang:
        english_id = next(iter(codec_lang.values()), None)
    updated = False
    for lang in languages:
        if lang not in codec_lang and english_id is not None:
            codec_lang[lang] = english_id
            updated = True
    if not updated:
        return
    if isinstance(tc, dict):
        tc["codec_language_id"] = codec_lang
    else:
        tc.codec_language_id = codec_lang


def load_speaker_refs(speakers: List[str], voices_dir: Optional[str] = None) -> None:
    """Load ref_audio and ref_mel for each speaker into REF_AUDIO_CACHE and REF_MEL_CACHE."""
    import librosa
    voices_dir = voices_dir or _voices_dir()
    for speaker in speakers:
        if speaker in REF_MEL_CACHE:
            continue
        path = os.path.join(voices_dir, f"{speaker}.wav")
        if not os.path.exists(path):
            logger.warning(f"Ref audio not found: {path}, skipping speaker {speaker}")
            continue
        audio, sr = librosa.load(path, sr=None, mono=True)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = audio.astype(np.float32)
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
        REF_AUDIO_CACHE[speaker] = audio
        with torch.inference_mode():
            ref_mel = mel_spectrogram(
                torch.from_numpy(audio).unsqueeze(0),
                n_fft=1024, num_mels=128, sampling_rate=24000,
                hop_size=256, win_size=1024, fmin=0, fmax=12000,
            ).transpose(1, 2)
        REF_MEL_CACHE[speaker] = ref_mel
    logger.info(f"Loaded ref_audio/ref_mel for speakers: {', '.join(REF_MEL_CACHE)}")


def ref_mels_for_speakers(speakers: List[str], device: torch.device) -> torch.Tensor:
    """Build (B, T, 128) ref_mels tensor from list of speaker ids using REF_MEL_CACHE.
    Caches result per (speakers, device) to avoid repeated CPU->GPU transfer (full GPU/CPU use).
    """
    device_str = str(device)
    key = (tuple(speakers), device_str)
    if key in REF_MEL_DEVICE_CACHE:
        return REF_MEL_DEVICE_CACHE[key]
    ref_list = [REF_MEL_CACHE[s] for s in speakers]
    max_t = max(m.shape[1] for m in ref_list)
    padded = torch.stack([
        m if m.shape[1] == max_t else torch.nn.functional.pad(m, (0, 0, 0, max_t - m.shape[1]))
        for m in ref_list
    ], dim=0).to(device)
    REF_MEL_DEVICE_CACHE[key] = padded
    return padded


def _run_save_checkpoint_io(
    state_dict: Dict[str, torch.Tensor],
    output_dir: str,
    model: Any,
    trainer: Any,
    checkpoint_type: str,
    step: int,
    epoch: int,
) -> None:
    """Run checkpoint file I/O in a background thread (batch-compatible, does not block training)."""
    save_path = os.path.join(output_dir, "model.safetensors")
    save_file(state_dict, save_path)
    print(f"✓ Saved model.safetensors")
    config_dict = model.model.config.to_dict()
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    if not isinstance(talker_config, dict):
        talker_config = {}
    train_speakers_list = [s.strip().lower() for s in trainer.config.train_speakers.split(",") if s.strip()]
    if not train_speakers_list:
        train_speakers_list = [trainer.config.speaker_name.lower()]
    talker_config["spk_id"] = {name: 3000 + i for i, name in enumerate(train_speakers_list)}
    talker_config["spk_is_dialect"] = {name: False for name in train_speakers_list}
    tc_obj = getattr(model.model.config, "talker_config", None)
    codec_lang = talker_config.get("codec_language_id")
    if isinstance(tc_obj, object) and getattr(tc_obj, "codec_language_id", None) and isinstance(tc_obj.codec_language_id, dict):
        codec_lang = dict(tc_obj.codec_language_id)
    if not isinstance(codec_lang, dict):
        codec_lang = {}
    english_id = codec_lang.get("english")
    if english_id is None and codec_lang:
        english_id = next(iter(codec_lang.values()), None)
    if english_id is not None:
        if "english" not in codec_lang:
            codec_lang["english"] = english_id
        if "hausa" not in codec_lang:
            codec_lang["hausa"] = english_id
    talker_config["codec_language_id"] = codec_lang
    config_dict["talker_config"] = talker_config
    saved_num_layers = config_dict["talker_config"]["num_hidden_layers"]
    actual_num_layers = len(model.model.talker.model.layers)
    if saved_num_layers != actual_num_layers:
        config_dict["talker_config"]["num_hidden_layers"] = actual_num_layers
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved config.json")
    if hasattr(model.model, "generate_config") and model.model.generate_config is not None:
        with open(os.path.join(output_dir, "generation_config.json"), "w", encoding="utf-8") as f:
            json.dump(model.model.generate_config, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved generation_config.json")
    _processor = getattr(model, "processor", None) or getattr(model.model, "processor", None)
    if _processor is not None:
        _processor.save_pretrained(output_dir)
        print(f"✓ Saved processor and tokenizer files")
    if hasattr(model.model, "speech_tokenizer") and model.model.speech_tokenizer is not None:
        speech_tokenizer_dir = os.path.join(output_dir, "speech_tokenizer")
        os.makedirs(speech_tokenizer_dir, exist_ok=True)
        model.model.speech_tokenizer.model.save_pretrained(speech_tokenizer_dir)
        model.model.speech_tokenizer.feature_extractor.save_pretrained(speech_tokenizer_dir)
        print(f"✓ Saved speech_tokenizer to {speech_tokenizer_dir}")
    if hasattr(model.model, "speaker_encoder") and model.model.speaker_encoder is not None:
        speaker_encoder_dir = os.path.join(output_dir, "speaker_encoder")
        os.makedirs(speaker_encoder_dir, exist_ok=True)
        speaker_encoder_config = {
            "model_type": "qwen3_tts_speaker_encoder",
            "speaker_name": trainer.config.speaker_name,
            "speaker_embedding_dim": trainer.target_speaker_embedding.shape[-1] if trainer.target_speaker_embedding is not None else 1024,
        }
        with open(os.path.join(speaker_encoder_dir, "speaker_config.json"), "w", encoding="utf-8") as f:
            json.dump(speaker_encoder_config, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved speaker encoder config for speaker: {trainer.config.speaker_name}")
    config_snapshot = asdict(trainer.config)
    config_snapshot.pop("hf_token", None)
    training_state = {
        "step": step,
        "epoch": epoch,
        "best_val_loss": trainer.best_val_loss,
        "best_val_metrics": trainer.best_val_metrics,
        "config": config_snapshot,
    }
    with open(os.path.join(output_dir, "training_state.json"), "w", encoding="utf-8") as f:
        json.dump(training_state, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved training_state.json")
    readme_content = f"""# Fine-Tuned Qwen3-TTS Model Checkpoint

## Model Information
- **Speaker Name**: {trainer.config.speaker_name}
- **Base Model**: {trainer.config.init_model_path}
- **Number of Layers**: {config_dict.get('talker_config', {}).get('num_hidden_layers', 'N/A')}
- **Hidden Size**: {config_dict.get('talker_config', {}).get('hidden_size', 'N/A')}
- **Training Epoch**: {epoch}
- **Training Step**: {step}
- **Best Validation Loss**: {trainer.best_val_loss:.4f}

## Training Configuration
- **Learning Rate**: {trainer.config.learning_rate}
- **Train Batch Size**: {trainer.config.train_batch_size}
- **Validation Batch Size**: {trainer.config.validation_batch_size}
- **Gradient Accumulation Steps**: {trainer.config.gradient_accumulation_steps}
- **Weight Decay**: {trainer.config.weight_decay}
- **Warmup Steps**: {trainer.config.warmup_steps}
- **Speaker Encoder Frozen**: {trainer.config.freeze_speaker_encoder}
- **Layer Replacement**: {trainer.config.replace_last_n_layers} layers replaced, {trainer.config.add_new_layers} layers added
- **Original Layers Frozen**: {trainer.config.freeze_original_layers}

## Files Included
- `config.json` - Model configuration
- `generation_config.json` - Generation parameters
- `model.safetensors` - Model weights (includes speaker encoder weights)
- `tokenizer_config.json` - Tokenizer configuration
- `vocab.json` - Vocabulary file
- `merges.txt` - BPE merges file
- `preprocessor_config.json` - Preprocessor configuration
- `speech_tokenizer/` - Speech tokenizer model and config
- `speaker_encoder/speaker_config.json` - Speaker encoder configuration
- `training_state.json` - Training state and configuration

## Loading the Model

```python
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# Load the fine-tuned model
model = Qwen3TTSModel.from_pretrained(
    "{output_dir}",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# Generate speech with the new speaker
text = "Your text here"
ref_audio = "path/to/reference_audio.wav"

wavs, sr = model.generate_voice_clone(
    text=text,
    language="Auto",
    ref_audio=ref_audio,
    ref_text="Reference text for ICL mode",
    x_vector_only_mode=False
)
```

## Speaker Information
The model has been fine-tuned for speaker: **{trainer.config.speaker_name}**
Speaker embedding is stored at index 3000 in the codec embedding layer.
Speaker encoder weights are included in the checkpoint and have been fine-tuned.

## Notes
- This model uses the Qwen3-TTS tokenizer
- The model supports streaming generation
- For best results, use reference audio from the same speaker used during training
- The speaker encoder has been fine-tuned to better capture speaker characteristics
"""
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"✓ Created README.md with usage instructions")
    print(f"✓ Saved {checkpoint_type} checkpoint to {output_dir}")
    if checkpoint_type in ["best", "last"] and getattr(trainer.config, "upload_to_hf", False) and getattr(trainer.config, "hf_token", None):
        trainer._upload_checkpoint_to_hf(output_dir, trainer.config.hf_best_model_repo if checkpoint_type == "best" else trainer.config.hf_last_model_repo, checkpoint_type)


@dataclass
class TrainingConfig:
    """Training configuration loaded from .env file."""
    
    
    # Data: HuggingFace combined multi-speaker only
    hf_dataset_repo: str = os.getenv("HF_DATASET_REPO", "vaghawan/qwen3-tts-multi-speaker")
    train_speakers: str = os.getenv("TRAIN_SPEAKERS", "hausa_speaker,english_speaker")

    # Model Configuration - Paths to models and tokenizer
    init_model_path: str = os.getenv("INIT_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")  # Base model path (12Hz or 25Hz model)
    tokenizer_path: str = os.getenv("TOKENIZER_PATH", "Qwen/Qwen3-TTS-Tokenizer-12Hz")  # Tokenizer path (12Hz or 25Hz)
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")  # Directory for checkpoints and logs
    speaker_name: str = os.getenv("SPEAKER_NAME", "reference_speaker")  # Speaker name for fine-tuning
    
    # Training Hyperparameters - Learning rate, epochs, etc.
    batch_size: int = int(os.getenv("BATCH_SIZE", 2))  # Batch size for training (depends on GPU memory)
    train_batch_size: int = int(os.getenv("TRAIN_BATCH_SIZE", os.getenv("BATCH_SIZE", 2)))
    validation_batch_size: int = int(os.getenv("VALIDATION_BATCH_SIZE", os.getenv("BATCH_SIZE", 2)))
    learning_rate: float = float(os.getenv("LEARNING_RATE", 2e-5))  # Learning rate for optimizer
    num_epochs: int = int(os.getenv("NUM_EPOCHS", 3))  # Number of training epochs
    gradient_accumulation_steps: int = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 8))  # Gradient accumulation for larger effective batch size
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", 0.01))  # Weight decay for regularization
    warmup_steps: int = int(os.getenv("WARMUP_STEPS", 200))  # Number of warmup steps for learning rate schedule
    max_grad_norm: float = float(os.getenv("MAX_GRAD_NORM", 1.0))  # Maximum gradient norm for clipping
    sub_talker_loss_weight: float = float(os.getenv("SUB_TALKER_LOSS_WEIGHT", 0.3))  # Weight for auxiliary sub-talker loss

    # Auxiliary losses (reconstruction, voice consistency, prosody)
    use_auxiliary_losses: bool = os.getenv("USE_AUXILIARY_LOSSES", "true").lower() == "true"
    mel_reconstruction_weight: float = float(os.getenv("MEL_RECONSTRUCTION_WEIGHT", 0.2))
    reconstruction_weight: float = float(os.getenv("RECONSTRUCTION_WEIGHT", 0.15))
    voice_consistency_weight: float = float(os.getenv("VOICE_CONSISTENCY_WEIGHT", 0.2))
    prosody_weight: float = float(os.getenv("PROSODY_WEIGHT", 0.15))
    audio_loss_every_n_steps: int = int(os.getenv("AUDIO_LOSS_EVERY_N_STEPS", "4"))

    # Layer Replacement Configuration - How many layers to replace/add
    replace_last_n_layers: int = int(os.getenv("REPLACE_LAST_N_LAYERS", 2))  # Number of last layers to replace with new ones
    add_new_layers: int = int(os.getenv("ADD_NEW_LAYERS", 4))  # Number of additional layers to add after replacement
    freeze_original_layers: bool = os.getenv("FREEZE_ORIGINAL_LAYERS", "true").lower() == "true"  # Whether to freeze original (non-replaced) layers
    freeze_speaker_encoder: bool = os.getenv("FREEZE_SPEAKER_ENCODER", "false").lower() == "true"  # Whether to freeze speaker encoder (default: False for finetuning)
    
    # Logging and Checkpointing - Frequency of logging and checkpointing
    logging_steps: int = int(os.getenv("LOGGING_STEPS", 10))  # Log training metrics every N steps
    save_steps: int = int(os.getenv("SAVE_STEPS", 500))  # Save checkpoint every N steps
    eval_steps: int = int(os.getenv("EVAL_STEPS", 500))  # Evaluate validation set every N steps
    save_total_limit: int = int(os.getenv("SAVE_TOTAL_LIMIT", 3))  # Maximum number of checkpoints to keep
    
    
    # WandB Configuration - Project tracking and metrics
    
    use_wandb: bool = os.getenv("USE_WANDB", "true").lower() == "true"  # Enable WandB tracking
    wandb_project: str = os.getenv("WANDB_PROJECT", "qwen3-tts-training")  # WandB project name
    wandb_run_name: Optional[str] = os.getenv("WANDB_RUN_NAME", None)  # WandB run name (optional)
    wandb_entity: Optional[str] = os.getenv("WANDB_ENTITY", None)  # WandB entity/team name (optional)
    
    
    # HuggingFace Configuration - Model upload configuration
    
    hf_token: Optional[str] = os.getenv("HF_TOKEN", None)  # HuggingFace authentication token (required for upload)
    upload_to_hf: bool = os.getenv("UPLOAD_TO_HF", "false").lower() == "true"  # Enable HuggingFace Hub upload
    hf_best_model_repo: str = os.getenv("HF_BEST_MODEL_REPO", "your-username/tts-best")  # Repo for best model (lowest val loss)
    hf_last_model_repo: str = os.getenv("HF_LAST_MODEL_REPO", "your-username/tts-last")  # Repo for last checkpoint
    
    # Device and Precision - Training device setup
    
    device: str = os.getenv("DEVICE", "cuda")  # Device: cuda (GPU) or cpu
    mixed_precision: str = os.getenv("MIXED_PRECISION", "bf16")  # Mixed precision: bf16, fp16, or no
    # Load model directly onto GPU from the start (max GPU use). e.g. cuda:0 or cuda. Empty = let Accelerator place (default for multi-GPU).
    model_device_map: Optional[str] = os.getenv("MODEL_DEVICE_MAP") or None

    # Tokenizer and audio codes: CPU (safe with fork) or CUDA (requires spawn for DataLoader workers)
    use_cpu_for_tokenizer_audio_codes: bool = os.getenv("USE_CPU_FOR_TOKENIZER_AUDIO_CODES", "true").lower() == "true"
    # Only speech_tokenizer.decoder on CPU (saves VRAM; encoder stays on GPU). Decode() moves data to CPU and back.
    speech_tokenizer_decoder_on_cpu: bool = os.getenv("SPEECH_TOKENIZER_DECODER_ON_CPU", "true").lower() == "true"
    # Multiprocessing start method for DataLoader: auto (spawn when using CUDA for tokenizer, else fork), fork, or spawn
    dataloader_multiprocessing_start_method: str = os.getenv("DATALOADER_MULTIPROCESSING_START_METHOD", "auto").lower()
    # DataLoader workers and prefetch (faster training: more workers + prefetch + persistent workers)
    dataloader_num_workers: Optional[int] = int(os.getenv("DATALOADER_NUM_WORKERS")) if os.getenv("DATALOADER_NUM_WORKERS") else None  # None = auto from GPU count
    dataloader_prefetch_factor: int = int(os.getenv("DATALOADER_PREFETCH_FACTOR", "4"))  # Batches to prefetch per worker (only if num_workers > 0)
    dataloader_persistent_workers: bool = os.getenv("DATALOADER_PERSISTENT_WORKERS", "true").lower() == "true"  # Keep workers alive between epochs (faster)
    
    # Data Limits - Max samples to use for testing
    
    max_train_samples: Optional[int] = int(os.getenv("MAX_TRAIN_SAMPLES")) if os.getenv("MAX_TRAIN_SAMPLES") else None  # Max training samples (for debugging)
    max_eval_samples: Optional[int] = int(os.getenv("MAX_VAL_SAMPLES")) if os.getenv("MAX_VAL_SAMPLES") else None  # Max validation samples (for debugging)
    cache_dir: str = os.getenv("CACHE_DIR", "./cache")  # Cache directory for streaming mode
    use_streaming_dataset: bool = os.getenv("USE_STREAMING_DATASET", "false").lower() == "true"  # Stream from HF (low RAM)
    shuffle_buffer_size: int = int(os.getenv("SHUFFLE_BUFFER_SIZE", "1000"))  # Buffer size for streaming shuffle
    
    
    # Reference Audio - Reference audio for speaker cloning
    
    ref_audio_path: Optional[str] = os.getenv("REF_AUDIO_PATH", None)  # Path to reference audio file (optional)


class Logger:
    """Logger for training and validation metrics."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.training_log_file = os.path.join(output_dir, "training_log.jsonl")
        self.validation_log_file = os.path.join(output_dir, "validation_log.jsonl")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def log_training(self, step: int, epoch: int, loss: float, learning_rate: float, **kwargs):
        """Log training metrics to training_log.jsonl."""
        log_entry = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "learning_rate": float(f"{learning_rate:.10e}"),  # Ensure full scientific notation
            "timestamp": datetime.datetime.now().isoformat(),
            "progress_pct": kwargs.get("progress_pct", 0),
            "epoch_progress": kwargs.get("epoch_progress", 0),
            **{k: v for k, v in kwargs.items() if k not in ["progress_pct", "epoch_progress"]}
        }
        # Include main_loss, sub_talker_loss, grad_norm when provided
        for key in ("main_loss", "sub_talker_loss", "grad_norm"):
            if key in kwargs:
                log_entry[key] = kwargs[key]
        
        with open(self.training_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_validation(self, step: int, epoch: int, loss: float, metrics: Dict[str, float], **kwargs):
        """Log validation metrics to validation_log.jsonl. main_loss and sub_talker_loss at top level for separate charting."""
        log_entry = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "main_loss": metrics.get("main_loss"),
            "sub_talker_loss": metrics.get("sub_talker_loss"),
            "metrics": metrics,
            "timestamp": datetime.datetime.now().isoformat(),
            **kwargs
        }
        
        with open(self.validation_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')


class DataProcessor:
    """Handle data loading and preparation."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.train_preprocessor = None
        self.eval_preprocessor = None
    
    def prepare_data(self):
        """Data is loaded from HuggingFace in get_train_dataloader / get_eval_dataloader."""
        print("="*60)
        print("Step 1: Data Preparation")
        print("="*60)
        print("Loading from HuggingFace (combined multi-speaker)")
        print(f"   Repo: {self.config.hf_dataset_repo}")
        print(f"   Speakers: {self.config.train_speakers}")
        print("="*60)
    
    def get_train_dataloader(self, model, config) -> DataLoader:
        """Get training dataloader from HuggingFace (combined multi-speaker).
        Uses model.processor (Qwen3TTSProcessor from processing_qwen3_tts): text tokenizer + chat template
        for encoding prompts; same processor is saved in checkpoints so inference can load from checkpoint.
        """
        from finetuning.data_processing import get_multispeaker_finetune_dataloader
        speakers = [s.strip() for s in self.config.train_speakers.split(",") if s.strip()]
        if not speakers:
            speakers = ["hausa_speaker", "english_speaker"]
        return get_multispeaker_finetune_dataloader(
            repo_id=self.config.hf_dataset_repo,
            speakers=speakers,
            split="train",
            processor=model.processor,
            config=config,
            batch_size=self.config.train_batch_size,
            max_samples=self.config.max_train_samples,
            tokenizer_path=self.config.tokenizer_path,
            tokenizer_device="cpu" if self.config.use_cpu_for_tokenizer_audio_codes else "cuda",
            cache_dir=self.config.cache_dir,
            num_workers=self.config.dataloader_num_workers,
            prefetch_factor=self.config.dataloader_prefetch_factor,
            persistent_workers=self.config.dataloader_persistent_workers,
            use_streaming=self.config.use_streaming_dataset,
            shuffle_buffer_size=self.config.shuffle_buffer_size,
        )
    
    def get_eval_dataloader(self, model, config) -> Optional[DataLoader]:
        """Get validation dataloader from HuggingFace (combined multi-speaker).
        Uses model.processor (Qwen3TTSProcessor) same as train.
        """
        from finetuning.data_processing import get_multispeaker_finetune_dataloader
        speakers = [s.strip() for s in self.config.train_speakers.split(",") if s.strip()]
        if not speakers:
            speakers = ["hausa_speaker", "english_speaker"]
        return get_multispeaker_finetune_dataloader(
            repo_id=self.config.hf_dataset_repo,
            speakers=speakers,
            split="validation",
            processor=model.processor,
            config=config,
            batch_size=self.config.validation_batch_size,
            max_samples=self.config.max_eval_samples,
            tokenizer_path=self.config.tokenizer_path,
            tokenizer_device="cpu" if self.config.use_cpu_for_tokenizer_audio_codes else "cuda",
            cache_dir=self.config.cache_dir,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers,
            prefetch_factor=self.config.dataloader_prefetch_factor,
            persistent_workers=self.config.dataloader_persistent_workers,
            use_streaming=self.config.use_streaming_dataset,
            shuffle_buffer_size=self.config.shuffle_buffer_size,
        )


class Trainer:
    """Main trainer class."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = Logger(config.output_dir)
        self.target_speaker_embedding = None
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        self.dual_loss_trainer = None
        self._last_aux = None
        self._save_checkpoint_thread = None
        
        # Initialize accelerator (use bf16/fp16 from config for full GPU throughput)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision if config.mixed_precision in ("bf16", "fp16", "no") else "no",
            log_with="wandb" if config.use_wandb else None,
        )
        
        # Initialize WandB
        if config.use_wandb:
            self.accelerator.init_trackers(
                project_name=config.wandb_project,
                config=asdict(config),
                init_kwargs={"wandb": {"name": config.wandb_run_name, "entity": config.wandb_entity}}
            )
    
    def load_model(self):
        """Load model and apply layer replacement."""
        print("="*60)
        print("Step 2: Loading Model")
        print("="*60)
        print(f"Model path: {self.config.init_model_path}")
        
        # Try to load model with different attention implementations
        load_kwargs: Dict[str, Any] = {"torch_dtype": torch.float32}
        if self.config.model_device_map:
            load_kwargs["device_map"] = self.config.model_device_map
            print(f"  Model device_map: {self.config.model_device_map} (load on GPU from start)")
        try:
            model = Qwen3TTSModel.from_pretrained(
                self.config.init_model_path,
                attn_implementation="flash_attention_2",
                **load_kwargs,
            )
            print(f"✓ Model loaded with flash_attention_2")
        except ImportError as e:
            print(f"⚠ Flash attention not available, falling back to SDPA")
            print(f"   (You can install flash-attn for potentially faster training)")
            model = Qwen3TTSModel.from_pretrained(
                self.config.init_model_path,
                attn_implementation="sdpa",
                **load_kwargs,
            )
            print(f"✓ Model loaded with SDPA (Scaled Dot Product Attention)")
        except Exception as e:
            print(f"⚠ Error loading model with flash_attention_2: {e}")
            print(f"   Trying SDPA fallback...")
            model = Qwen3TTSModel.from_pretrained(
                self.config.init_model_path,
                attn_implementation="sdpa",
                **load_kwargs,
            )
            print(f"✓ Model loaded with SDPA")
        
        # Model will be cast to bf16 by accelerator, no manual casting needed
        print(f"✓ Model ready for bf16 mixed precision training")
        
        # Freeze or unfreeze speaker encoder
        if self.config.freeze_speaker_encoder:
            print(f"\nFreezing speaker encoder...")
            for param in model.model.speaker_encoder.parameters():
                param.requires_grad = False
            print(f"✓ Speaker encoder frozen")
        else:
            print(f"\nUnfreezing speaker encoder for finetuning...")
            for param in model.model.speaker_encoder.parameters():
                param.requires_grad = True
            print(f"✓ Speaker encoder unfrozen (gradients will flow)")
        
        # Print trainable parameters summary
        self._print_trainable_params(model)
        
        # Apply layer replacement if configured
        if self.config.replace_last_n_layers > 0 or self.config.add_new_layers > 0:
            print(f"\nApplying layer replacement and addition...")
            model.model = replace_and_add_layers(
                model.model,
                replace_last_n=self.config.replace_last_n_layers,
                add_new_layers=self.config.add_new_layers,
                freeze_original_layers=self.config.freeze_original_layers,
                verbose=True
            )
            print_model_summary(model.model)
        
        if self.config.use_auxiliary_losses:
            config_obj = AutoConfig.from_pretrained(self.config.init_model_path)
            self.dual_loss_trainer = DualLossTrainer(model, config_obj)
            model.model.mel_head = self.dual_loss_trainer.mel_head
            logger.info("✓ Auxiliary losses enabled (mel reconstruction, voice consistency, prosody)")
        
        return model
    
    def _print_trainable_params(self, model):
        """Print summary of trainable parameters."""
        print(f"\n{'='*60}")
        print("Trainable Parameters Summary")
        print(f"{'='*60}")
        
        total_params = 0
        trainable_params = 0
        
        # Count parameters by component
        components = {
            'speaker_encoder': 0,
            'talker.model.layers': 0,
            'talker.model.text_embedding': 0,
            'talker.model.codec_embedding': 0,
            'talker.code_predictor': 0,
            'other': 0
        }
        
        for name, param in model.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
                # Categorize by component
                if 'speaker_encoder' in name:
                    components['speaker_encoder'] += param.numel()
                elif 'talker.model.layers' in name:
                    components['talker.model.layers'] += param.numel()
                elif 'talker.model.text_embedding' in name:
                    components['talker.model.text_embedding'] += param.numel()
                elif 'talker.model.codec_embedding' in name:
                    components['talker.model.codec_embedding'] += param.numel()
                elif 'talker.code_predictor' in name:
                    components['talker.code_predictor'] += param.numel()
                else:
                    components['other'] += param.numel()
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
        print(f"\nTrainable parameters by component:")
        for comp, count in components.items():
            if count > 0:
                print(f"  {comp}: {count:,} ({100 * count / trainable_params:.2f}%)")
        print(f"{'='*60}\n")
    
    def train(self, model, train_dataloader, eval_dataloader=None):
        """Main training loop."""
        print("="*60)
        print("Step 4: Training")
        print("="*60)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        try:
            steps_per_epoch = len(train_dataloader)
        except TypeError:
            import math
            n = max(1, getattr(self.accelerator, "num_processes", 1))
            steps_per_epoch = (
                math.ceil(self.config.max_train_samples / (self.config.train_batch_size * n))
                if (self.config.max_train_samples and self.config.train_batch_size)
                else 10000
            )
            print(f"  (streaming/iterable dataset: using steps_per_epoch={steps_per_epoch})")
        total_steps = steps_per_epoch * self.config.num_epochs
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Prepare with accelerator
        model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        
        if eval_dataloader is not None:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        model.model.train()
        global_step = 0
        try:
            steps_per_epoch = len(train_dataloader)
        except TypeError:
            import math
            n = max(1, self.accelerator.num_processes)
            steps_per_epoch = (
                math.ceil(self.config.max_train_samples / (self.config.train_batch_size * n))
                if (self.config.max_train_samples and self.config.train_batch_size)
                else 10000
            )
            pass
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            # Create progress bar for this epoch (total=steps_per_epoch when known for accurate %)
            try:
                _total = len(train_dataloader)
            except TypeError:
                _total = None
            pbar = tqdm(
                train_dataloader,
                desc=f"Train Epoch {epoch + 1}/{self.config.num_epochs}",
                total=_total,
                unit="batch",
                dynamic_ncols=True,
                disable=not self.accelerator.is_main_process,
            )
            
            for step, batch in enumerate(pbar):
                with self.accelerator.accumulate(model):
                    loss, loss_components = self.training_step(model, batch, global_step=global_step)
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    grad_norm = None
                    if self.accelerator.sync_gradients:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.model.parameters(), self.config.max_grad_norm)
                        if grad_norm is not None and not isinstance(grad_norm, float):
                            grad_norm = grad_norm.item()
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                global_step += 1
                
                # Calculate progress percentage
                progress_pct = (global_step / total_steps) * 100
                epoch_progress = ((step + 1) / steps_per_epoch) * 100
                
                # Update progress bar (batch index + overall % when total known)
                current_lr = scheduler.get_last_lr()[0]
                postfix = {
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                }
                if _total is not None:
                    postfix['batch'] = f'{step + 1}/{_total}'
                postfix['progress'] = f'{progress_pct:.1f}%'
                pbar.set_postfix(postfix)
                
                # Log to WandB at every step for clear graph visualization
                if self.config.use_wandb:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/main_loss": loss_components["main_loss"],
                        "train/sub_talker_loss": loss_components["sub_talker_loss"],
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                        "train/progress_pct": progress_pct,
                        "train/epoch_progress": epoch_progress
                    }
                    if grad_norm is not None:
                        log_dict["train/grad_norm"] = grad_norm
                    # Aux losses: mel is computed every step; waveform/voice/prosody only when step % audio_loss_every_n_steps == 0 (else 0)
                    if getattr(self, "_last_aux", None) is not None:
                        for k, v in self._last_aux.items():
                            log_dict[f"train/aux_{k}"] = v
                    self.accelerator.log(log_dict, step=global_step)
                
                # Log to file and print progress at configured intervals
                if global_step % self.config.logging_steps == 0:
                    self.logger.log_training(
                        step=global_step,
                        epoch=epoch,
                        loss=loss.item(),
                        learning_rate=current_lr,
                        progress_pct=progress_pct,
                        epoch_progress=epoch_progress,
                        main_loss=loss_components["main_loss"],
                        sub_talker_loss=loss_components["sub_talker_loss"],
                        grad_norm=grad_norm,
                    )
                    # Mirror same metrics to WandB at logging_steps so file and WandB stay in sync
                    if self.config.use_wandb:
                        step_log = {
                            "train/loss": loss.item(),
                            "train/main_loss": loss_components["main_loss"],
                            "train/sub_talker_loss": loss_components["sub_talker_loss"],
                            "train/learning_rate": current_lr,
                            "train/progress_pct": progress_pct,
                            "train/epoch_progress": epoch_progress,
                        }
                        if grad_norm is not None:
                            step_log["train/grad_norm"] = grad_norm
                        if getattr(self, "_last_aux", None) is not None:
                            for k, v in self._last_aux.items():
                                step_log[f"train/aux_{k}"] = v
                        self.accelerator.log(step_log, step=global_step)
                    
                    if self.accelerator.is_main_process:
                        print(f"\n{'='*60}")
                        print(f"📊 Training Progress - Step {global_step}/{total_steps}")
                        print(f"{'='*60}")
                        print(f"  Overall Progress: {progress_pct:.2f}%")
                        print(f"  Epoch {epoch + 1} Progress: {epoch_progress:.2f}% ({step + 1}/{steps_per_epoch} steps)")
                        print(f"  Loss: {loss.item():.4f} (main: {loss_components['main_loss']:.4f}, sub_talker: {loss_components['sub_talker_loss']:.4f})")
                        if grad_norm is not None:
                            print(f"  Grad norm: {grad_norm:.4f}")
                        print(f"  Learning Rate: {current_lr:.10e}")
                        print(f"{'='*60}")
                
                # Evaluation
                if eval_dataloader is not None and global_step % self.config.eval_steps == 0:
                    print(f"\n{'='*60}")
                    print(f"🔍 Running Validation at Step {global_step}")
                    print(f"{'='*60}")
                    val_loss, val_metrics = self.evaluate(model, eval_dataloader, global_step, epoch)
                    
                    if self.accelerator.is_main_process:
                        print(f"\n{'='*60}")
                        print(f"📈 Validation Results - Step {global_step}")
                        print(f"{'='*60}")
                        print(f"  Validation Loss: {val_loss:.4f}")
                        print(f"  Main Loss: {val_metrics.get('main_loss', val_loss):.4f}")
                        print(f"  Sub-talker Loss: {val_metrics.get('sub_talker_loss', 0):.4f}")
                        for k in EVALUATION_METRICS:
                            v = val_metrics.get(k)
                            if v is not None:
                                print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: N/A")
                        print(f"  Best Validation Loss: {self.best_val_loss:.4f}")
                        print(f"{'='*60}")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        print(f"\n✨ New best model found! Saving...")
                        self.best_val_loss = val_loss
                        self.best_val_metrics = val_metrics
                        self.save_checkpoint(model, optimizer, scheduler, global_step, epoch, "best")
                        if self.accelerator.is_main_process:
                            print(f"✅ Best model saved at step {global_step}")
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    print(f"\n💾 Saving checkpoint at step {global_step}...")
                    self.save_checkpoint(model, optimizer, scheduler, global_step, epoch, "checkpoint")
                    if self.accelerator.is_main_process:
                        print(f"✅ Checkpoint saved at step {global_step}")
            
            # Close progress bar for this epoch
            pbar.close()
            
            # End of epoch evaluation
            if eval_dataloader is not None:
                print(f"\n{'='*60}")
                print(f"🔍 End of Epoch {epoch + 1} Validation")
                print(f"{'='*60}")
                val_loss, val_metrics = self.evaluate(model, eval_dataloader, global_step, epoch)
                
                if self.accelerator.is_main_process:
                    print(f"\n{'='*60}")
                    print(f"📈 Epoch {epoch + 1} Validation Results")
                    print(f"{'='*60}")
                    print(f"  Validation Loss: {val_loss:.4f}")
                    print(f"  Main Loss: {val_metrics.get('main_loss', val_loss):.4f}")
                    print(f"  Sub-talker Loss: {val_metrics.get('sub_talker_loss', 0):.4f}")
                    for k in EVALUATION_METRICS:
                        v = val_metrics.get(k)
                        if v is not None:
                            print(f"  {k}: {v:.4f}" if np.isfinite(v) else f"  {k}: N/A")
                    print(f"  Best Validation Loss: {self.best_val_loss:.4f}")
                    print(f"{'='*60}")
                
                if val_loss < self.best_val_loss:
                    print(f"\n✨ New best model found at end of epoch {epoch + 1}! Saving...")
                    self.best_val_loss = val_loss
                    self.best_val_metrics = val_metrics
                    self.save_checkpoint(model, optimizer, scheduler, global_step, epoch, "best")
                    if self.accelerator.is_main_process:
                        print(f"✅ Best model saved at end of epoch {epoch + 1}")
        
        # Save last checkpoint
        self.save_checkpoint(model, optimizer, scheduler, global_step, epoch, "last")
        
        print("\n" + "="*60)
        print("Training complete!")
        print("="*60)
    
    def training_step(self, model, batch, global_step=None):
        """Perform one training step."""
        # Move all tensors to the model's device (accelerator will handle dtype)
        device = model.model.device
        if "speaker_ids" in batch:
            speaker_list = [s.strip() for s in self.config.train_speakers.split(",") if s.strip()]
            if not speaker_list:
                speaker_list = ["hausa_speaker", "english_speaker"]
            ids = batch["speaker_ids"].tolist()
            speakers = [speaker_list[i] if i < len(speaker_list) else speaker_list[0] for i in ids]
            ref_mels = ref_mels_for_speakers(speakers, device)
            batch = {**batch, "ref_mels": ref_mels}
        elif "speakers" in batch:
            ref_mels = ref_mels_for_speakers(batch["speakers"], device)
            batch = {**batch, "ref_mels": ref_mels}
        else:
            ref_mels = batch["ref_mels"].to(device)
        input_ids = batch['input_ids'].to(device)
        codec_ids = batch['codec_ids'].to(device)
        text_embedding_mask = batch['text_embedding_mask'].to(device)
        codec_embedding_mask = batch['codec_embedding_mask'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        codec_0_labels = batch['codec_0_labels'].to(device)
        codec_mask = batch['codec_mask'].to(device)
        
        # Get speaker embedding
        speaker_embedding = model.model.speaker_encoder(ref_mels)
        if self.target_speaker_embedding is None:
            self.target_speaker_embedding = speaker_embedding.clone().detach() # Create a new tensor with the same data and detach it from the computational graph
        
        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]
        
        # Let accelerator handle dtype conversion automatically
        input_text_embedding = model.model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
        input_codec_embedding = model.model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
        # Use talker's codec embedding for speaker (spk_id in input_codec_ids at position 6); do not overwrite
        # so that each speaker's embedding (3000, 3001, ...) is trained and distinct at inference.
        
        input_embeddings = input_text_embedding + input_codec_embedding
        
        for i in range(1, 16):
            codec_i_embedding = model.model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_i_embedding

        # logger.info(f"Input embeddings shape: {input_embeddings.shape}")
        # logger.info(f"Input embeddings: {input_embeddings}")
        # logger.info(f"Attention mask shape: {attention_mask.shape}")
        # logger.info(f"Attention mask: {attention_mask}")
        # logger.info(f"Codec 0 labels shape: {codec_0_labels.shape}")
        # logger.info(f"Codec 0 labels: {codec_0_labels}")
        # logger.info(f"Codec mask shape: {codec_mask.shape}")
        # logger.info(f"Codec mask: {codec_mask}")
        # logger.info(f"Speaker embedding shape: {speaker_embedding.shape}")

        # logger.info(f"Input text embedding data type: {input_text_embedding.dtype}")
        # logger.info(f"Input codec embedding data type: {input_codec_embedding.dtype}")
        # logger.info(f"Speaker embedding data type: {speaker_embedding.dtype}")

        # logger.info(f"Input embeddings data type: {input_embeddings.dtype}")
        # logger.info(f"Attention mask data type: {attention_mask.dtype}")
        # logger.info(f"Codec 0 labels data type: {codec_0_labels.dtype}")
        # logger.info(f"Codec mask data type: {codec_mask.dtype}")
        
        outputs = model.model.talker(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=codec_0_labels[:, 1:],
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[0][-1]
        talker_hidden_states = hidden_states[codec_mask[:, 1:]]
        talker_codec_ids = codec_ids[codec_mask]
        
        sub_talker_logits, sub_talker_loss = model.model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
        
        main_loss = outputs.loss
        loss = main_loss + self.config.sub_talker_loss_weight * sub_talker_loss
        loss_components = {
            "main_loss": main_loss.item(),
            "sub_talker_loss": sub_talker_loss.item(),
        }

        if self.config.use_auxiliary_losses and self.dual_loss_trainer is not None and "target_mel" in batch and "target_audio" in batch:
            # Every audio_loss_every_n_steps, generate waveforms and include waveform/voice/prosody losses
            generated_audio_tensor = None
            if global_step is not None and self.config.audio_loss_every_n_steps > 0 and global_step % self.config.audio_loss_every_n_steps == 0:
                was_training = model.model.training
                try:
                    if was_training:
                        model.model.eval()
                    with torch.no_grad():
                        gen_list = self._generate_audio_from_batch(model, batch)
                    if gen_list is not None:
                        generated_audio_tensor = self._pad_wav_list_to_tensor(gen_list, device)
                    if generated_audio_tensor is None:
                        # Fallback: use target_audio so we still compute and backprop waveform/voice/prosody.
                        # Waveform and prosody will be 0 (target vs target); voice_consistency is ref_emb vs target_mel emb (non-zero).
                        generated_audio_tensor = batch["target_audio"].to(device)
                        if self.accelerator.is_main_process:
                            logger.debug(
                                f"Step {global_step}: using target_audio as fallback for aux losses (generation failed or skipped)."
                            )
                finally:
                    if was_training:
                        model.model.train()
                if gen_list is not None and generated_audio_tensor is None and self.accelerator.is_main_process:
                    logger.warning(f"Step {global_step}: gen_list had {len(gen_list)} items but _pad_wav_list_to_tensor returned None.")

            aux = self.dual_loss_trainer.compute_auxiliary_losses(
                batch=batch,
                model=model,
                outputs=outputs,
                sample_rate=24000,
                generated_audio=generated_audio_tensor,
            )
            # All auxiliary losses are always added (weight * loss) so they are used in backward when non-zero.
            loss = loss + self.config.mel_reconstruction_weight * aux["mel_reconstruction_loss"]
            loss = loss + self.config.reconstruction_weight * aux["waveform_reconstruction_loss"]
            loss = loss + self.config.voice_consistency_weight * aux["voice_consistency_loss"]
            loss = loss + self.config.prosody_weight * aux["prosody_loss"]

            self._last_aux = {k: v.item() for k, v in aux.items() if torch.is_tensor(v)}
            loss_components.update(self._last_aux)
        else:
            self._last_aux = None
        return loss, loss_components

    def _generate_audio_from_batch(self, model, batch):
        """Generate waveform from one batch using predicted codec_0 and teacher-forced codec 1..15.
        Returns list of numpy arrays (one per sample) or None on failure.
        """
        device = model.model.device
        input_ids = batch["input_ids"].to(device)
        codec_ids = batch["codec_ids"].to(device)
        ref_mels = batch["ref_mels"].to(device)
        codec_mask = batch["codec_mask"].to(device)
        codec_0_labels = batch["codec_0_labels"].to(device)
        text_embedding_mask = batch["text_embedding_mask"].to(device)
        codec_embedding_mask = batch["codec_embedding_mask"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        speaker_embedding = model.model.speaker_encoder(ref_mels)
        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]
        input_text_embedding = model.model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
        input_codec_embedding = model.model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
        input_codec_embedding[:, 6, :] = speaker_embedding
        input_embeddings = input_text_embedding + input_codec_embedding
        for i in range(1, 16):
            codec_i_embedding = model.model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_i_embedding

        outputs = model.model.talker(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=codec_0_labels[:, 1:],
            output_hidden_states=False,
        )
        logits = outputs.logits
        if logits is None:
            return None
        pred_next = logits.argmax(dim=-1)
        pred_codec_0_full = torch.cat([codec_0_labels[:, 0:1], pred_next], dim=1)

        B = codec_ids.shape[0]
        codes_list = []
        for i in range(B):
            indices = codec_mask[i].nonzero(as_tuple=True)[0]
            if len(indices) == 0:
                codes_list.append(None)
                continue
            codes_i = codec_ids[i, indices].clone()
            codes_i[:, 0] = pred_codec_0_full[i, indices]
            codes_list.append(codes_i)

        valid_codes = []
        valid_indices = []
        for i in range(B):
            c = codes_list[i]
            if c is not None and c.shape[0] > 0:
                valid_codes.append(c)
                valid_indices.append(i)
        if not valid_codes:
            logger.debug("_generate_audio_from_batch: no valid_codes (codec_mask may be empty or indices mismatch)")
            return None
        if not hasattr(model.model, "speech_tokenizer") or model.model.speech_tokenizer is None:
            logger.warning(
                "Audio aux losses skipped: model.model.speech_tokenizer is missing. "
                "Load the base TTS model with speech_tokenizer (e.g. from_pretrained(init_model_path))."
            )
            return None
        # Clamp codes to tokenizer codebook range to avoid "index out of range" in decode
        codebook_size = _get_speech_tokenizer_codebook_size(model.model.speech_tokenizer)
        if codebook_size is not None:
            valid_codes = [torch.clamp(c.long(), 0, codebook_size - 1) for c in valid_codes]
        # Pass codes on same device as tokenizer (decode does .to(self.device)); avoids CPU sync
        try:
            wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": c} for c in valid_codes])
        except Exception as e:
            logger.warning(f"speech_tokenizer.decode failed: {e}")
            return None
        out = [None] * B
        for j, i in enumerate(valid_indices):
            if j < len(wavs):
                w = wavs[j]
                out[i] = w if isinstance(w, np.ndarray) else np.array(w, dtype=np.float32)
        return out

    def _pad_wav_list_to_tensor(self, wav_list, device):
        """Convert list of variable-length numpy wavs (or None) to padded tensor [B, max_len] on device."""
        if not wav_list:
            return None
        valid = []
        max_len = 0
        for i, w in enumerate(wav_list):
            if w is not None and len(w) > 0:
                valid.append(i)
                if len(w) > max_len:
                    max_len = len(w)
        if not valid:
            return None
        B = len(wav_list)
        out = np.zeros((B, max_len), dtype=np.float32)
        for i in valid:
            w = np.asarray(wav_list[i], dtype=np.float32).flatten()
            L = len(w)
            out[i, :L] = w[:L]
        return torch.from_numpy(out).to(device)

    def evaluate(self, model, eval_dataloader, step, epoch):
        """Evaluate model on validation set. Reports EVALUATION_METRICS."""
        print(f"\nEvaluating at step {step}...")
        model.model.eval()

        total_loss = 0
        total_samples = 0
        total_main_loss = 0
        total_sub_talker_loss = 0
        speaker_embeddings = []
        sample_batch_for_quality = None  # one batch with target_audio for quality_metrics

        with torch.no_grad():
            try:
                eval_total = len(eval_dataloader)
            except TypeError:
                eval_total = None
            eval_pbar = tqdm(
                eval_dataloader,
                desc=f"Eval Epoch {epoch + 1}/{self.config.num_epochs}",
                total=eval_total,
                unit="batch",
                dynamic_ncols=True,
                disable=not self.accelerator.is_main_process,
            )
            for batch in eval_pbar:
                loss, loss_components = self.training_step(model, batch)
                batch_size = batch["input_ids"].size(0)
                total_loss += loss.item() * batch_size
                total_main_loss += loss_components["main_loss"] * batch_size
                total_sub_talker_loss += loss_components["sub_talker_loss"] * batch_size
                total_samples += batch_size
                eval_pbar.set_postfix({
                    "loss": f"{total_loss / total_samples:.4f}",
                    "samples": total_samples,
                })

                if "speaker_ids" in batch:
                    speaker_list = [s.strip() for s in self.config.train_speakers.split(",") if s.strip()]
                    if not speaker_list:
                        speaker_list = ["hausa_speaker", "english_speaker"]
                    ids = batch["speaker_ids"].tolist()
                    speakers = [speaker_list[i] if i < len(speaker_list) else speaker_list[0] for i in ids]
                    ref_mels = ref_mels_for_speakers(speakers, model.model.device)
                elif "speakers" in batch:
                    ref_mels = ref_mels_for_speakers(batch["speakers"], model.model.device)
                else:
                    ref_mels = batch["ref_mels"].to(model.model.device)
                speaker_emb = model.model.speaker_encoder(ref_mels)
                speaker_embeddings.append(speaker_emb)

                if sample_batch_for_quality is None and "target_audio" in batch and "target_audio_lengths" in batch:
                    sample_batch_for_quality = batch

        avg_loss = total_loss / total_samples
        avg_main_loss = total_main_loss / total_samples
        avg_sub_talker_loss = total_sub_talker_loss / total_samples

        # Speaker embedding consistency (similarity within batch)
        if len(speaker_embeddings) > 1:
            all_embeddings = torch.cat(speaker_embeddings, dim=0)
            embeddings_norm = torch.nn.functional.normalize(all_embeddings, p=2, dim=-1)
            similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
            upper_tri = similarity_matrix.triu(diagonal=1)
            avg_similarity = upper_tri.mean().item()
        else:
            avg_similarity = 1.0

        # Build metrics with all five EVALUATION_METRICS plus loss breakdown
        metrics = {
            "perplexity": torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 20 else float("inf"),
            "speaker_embedding_consistency": avg_similarity,
            "pronunciation_accuracy": 0.0,
            "tonal_accuracy": 0.0,
            "prosody_accuracy": 0.0,
            "main_loss": avg_main_loss,
            "sub_talker_loss": avg_sub_talker_loss,
        }

        # When we have target audio, generate audio from the model and compute all five
        # metrics with QualityMetricsCalculator(generated_audio, target_audio).
        if sample_batch_for_quality is not None and self.accelerator.is_main_process:
            try:
                with torch.no_grad():
                    gen_wavs = self._generate_audio_from_batch(model, sample_batch_for_quality)
                if gen_wavs is None:
                    logger.warning(
                        "Quality metrics skipped: _generate_audio_from_batch returned None "
                        "(check codec_mask / valid_codes or speech_tokenizer availability)."
                    )
                else:
                    qcalc = QualityMetricsCalculator(sample_rate=24000)
                    target_audio = sample_batch_for_quality["target_audio"]
                    target_audio_lengths = sample_batch_for_quality["target_audio_lengths"]
                    B = min(len(gen_wavs), target_audio.shape[0])
                    sums = {k: 0.0 for k in EVALUATION_METRICS}
                    counts = {k: 0 for k in EVALUATION_METRICS}
                    for i in range(B):
                        L = int(target_audio_lengths[i].item())
                        if L < 100 or i >= len(gen_wavs) or gen_wavs[i] is None or len(gen_wavs[i]) < 100:
                            continue
                        ref_np = target_audio[i, :L].float().cpu().numpy()
                        gen_np = gen_wavs[i] if isinstance(gen_wavs[i], np.ndarray) else np.array(gen_wavs[i], dtype=np.float32)
                        m = qcalc.calculate_all_metrics(gen_np, ref_np)
                        for k in EVALUATION_METRICS:
                            sums[k] += m[k]
                            counts[k] += 1
                    for k in EVALUATION_METRICS:
                        if counts[k] > 0:
                            metrics[k] = sums[k] / counts[k]
            except Exception as e:
                logger.warning("Quality metrics (generated vs target_audio) failed: %s", e)

        # Log validation
        self.logger.log_validation(
            step=step,
            epoch=epoch,
            loss=avg_loss,
            metrics=metrics,
        )

        if self.config.use_wandb:
            log_dict = {
                "val/loss": avg_loss,
                "val/main_loss": avg_main_loss,
                "val/sub_talker_loss": avg_sub_talker_loss,
                "val/step": step,
                "val/epoch": epoch,
            }
            for k in EVALUATION_METRICS:
                log_dict[f"val/{k}"] = metrics[k]
            self.accelerator.log(log_dict, step=step)

        if self.accelerator.is_main_process:
            print(f"Validation Loss: {avg_loss:.4f} (main: {avg_main_loss:.4f}, sub_talker: {avg_sub_talker_loss:.4f})")
            for k in EVALUATION_METRICS:
                v = metrics[k]
                if isinstance(v, float) and not np.isfinite(v):
                    print(f"  {k}: N/A")
                else:
                    print(f"  {k}: {v:.4f}")

        model.model.train()
        return avg_loss, metrics

    def _upload_checkpoint_to_hf(self, checkpoint_dir: str, repo_id: str, checkpoint_type: str):
        """Upload a checkpoint directory to HuggingFace."""
        if not self.config.upload_to_hf or not self.config.hf_token:
            return

        if not self.accelerator.is_main_process:
            return

        if not os.path.exists(checkpoint_dir):
            print(f"⚠ Checkpoint directory {checkpoint_dir} does not exist, skipping upload")
            return

        try:
            print(f"\n{'='*60}")
            print(f"📤 Uploading {checkpoint_type} checkpoint to HuggingFace")
            print(f"{'='*60}")
            print(f"  Repository: {repo_id}")
            print(f"  Local path: {checkpoint_dir}")
            print(f"{'='*60}")

            login(token=self.config.hf_token)
            api = HfApi()

            api.upload_folder(
                folder_path=checkpoint_dir,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {checkpoint_type} checkpoint - Step {self.best_val_loss if checkpoint_type == 'best' else 'last'}, Speaker: {self.config.speaker_name}"
            )

            print(f"✓ Successfully uploaded {checkpoint_type} checkpoint to {repo_id}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"⚠ Error uploading {checkpoint_type} checkpoint to HuggingFace: {e}")
            print(f"{'='*60}\n")

    def save_checkpoint(self, model, optimizer, scheduler, step, epoch, checkpoint_type):
        """Save model checkpoint (I/O runs in background thread so training is not blocked)."""
        if not self.accelerator.is_main_process:
            return

        print(f"\n{'='*60}")
        print(f"💾 Saving {checkpoint_type.upper()} Checkpoint")
        print(f"{'='*60}")
        print(f"  Step: {step}")
        print(f"  Epoch: {epoch + 1}/{self.config.num_epochs}")
        print(f"  Type: {checkpoint_type}")
        print(f"{'='*60}")

        output_dir = os.path.join(self.config.output_dir, checkpoint_type)
        os.makedirs(output_dir, exist_ok=True)

        # Copy state to CPU so training can continue (batch-compatible, GPU-free for save)
        state_dict = {k: v.detach().to("cpu").clone() for k, v in model.model.state_dict().items()}
        if self.target_speaker_embedding is not None:
            weight = state_dict["talker.model.codec_embedding.weight"]
            state_dict["talker.model.codec_embedding.weight"][3000] = self.target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)

        # Count the number of layer weights being saved (single pass)
        unique_layers = set()
        for k in state_dict:
            if "talker.model.layers" not in k:
                continue
            parts = k.split(".")
            if "layers" in parts:
                unique_layers.add(parts[parts.index("layers") + 1])
        print(f"  Saving {len(unique_layers)} layers in checkpoint")

        # Wait for previous async save to finish so we don't overlap writes
        if self._save_checkpoint_thread is not None and self._save_checkpoint_thread.is_alive():
            self._save_checkpoint_thread.join()
        self._save_checkpoint_thread = threading.Thread(
            target=_run_save_checkpoint_io,
            args=(state_dict, output_dir, model, self, checkpoint_type, step, epoch),
            daemon=False,
        )
        self._save_checkpoint_thread.start()

        # Upload to HuggingFace is done inside _run_save_checkpoint_io after files are written
    
    def upload_to_huggingface(self):
        """Upload models to HuggingFace."""
        if not self.config.upload_to_hf or not self.config.hf_token:
            print("Skipping HuggingFace upload")
            return
        
        if not self.accelerator.is_main_process:
            return
        
        print("\n" + "="*60)
        print("Uploading to HuggingFace")
        print("="*60)
        
        # Login
        login(token=self.config.hf_token)
        api = HfApi()
        
        # Upload best model
        best_dir = os.path.join(self.config.output_dir, "best")
        if os.path.exists(best_dir):
            api.upload_folder(
                folder_path=best_dir,
                repo_id=self.config.hf_best_model_repo,
                repo_type="model",
                commit_message=f"Upload best model - {self.config.speaker_name}"
            )
            print(f"✓ Uploaded best model to {self.config.hf_best_model_repo}")
        
        # Upload last model
        last_dir = os.path.join(self.config.output_dir, "last")
        if os.path.exists(last_dir):
            api.upload_folder(
                folder_path=last_dir,
                repo_id=self.config.hf_last_model_repo,
                repo_type="model",
                commit_message=f"Upload last model - {self.config.speaker_name}"
            )
            print(f"✓ Uploaded last model to {self.config.hf_last_model_repo}")

    def cleanup(self):
        """Cleanup background preprocessors and wait for any async checkpoint save to finish."""
        print("\nCleaning up background preprocessors...")
        
        if self.train_preprocessor:
            self.train_preprocessor.stop()
        
        if self.eval_preprocessor:
            self.eval_preprocessor.stop()

        if self._save_checkpoint_thread is not None and self._save_checkpoint_thread.is_alive():
            self._save_checkpoint_thread.join()
            self._save_checkpoint_thread = None
        
        print("✓ Cleanup complete")


def _clear_all_caches() -> None:
    """Clear GPU and in-process caches to free memory (avoids OOM from fragmentation or previous runs)."""
    REF_MEL_DEVICE_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _move_speech_tokenizer_decoder_to_cpu(model: Any) -> None:
    """Move only speech_tokenizer.decoder to CPU to save VRAM; decode() handles device transfer."""
    if not getattr(model, "model", None):
        return
    st = getattr(model.model, "speech_tokenizer", None)
    if st is None or getattr(st, "model", None) is None:
        return
    dec = getattr(st.model, "decoder", None)
    if dec is not None:
        dec.to("cpu")
        logger.info("speech_tokenizer.decoder moved to CPU (encoder stays on GPU)")


def main():
    """Main training function."""
    # Reduce CUDA fragmentation (PyTorch recommendation when OOM with "reserved but unallocated")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("="*60)
    print("Qwen3-TTS Training Pipeline")
    print("="*60)

    # Clear all caches so training starts with a clean state (avoids OOM)
    _clear_all_caches()
    print("  All caches cleared (REF_MEL_DEVICE_CACHE + GPU)")

    # Load configuration
    config = TrainingConfig()
    
    # Set DataLoader multiprocessing start method before any workers are created.
    # Use spawn when tokenizer/audio codes run on CUDA (fork + CUDA is unsafe); otherwise fork is fine.
    _effective = config.dataloader_multiprocessing_start_method
    if _effective == "auto":
        _effective = "spawn" if (torch.cuda.is_available() and not config.use_cpu_for_tokenizer_audio_codes) else "fork"
    if _effective not in ("fork", "spawn", "forkserver"):
        _effective = "fork"
    try:
        multiprocessing.set_start_method(_effective, force=False)
        print(f"  DataLoader multiprocessing start method: {_effective}")
    except RuntimeError:
        pass  # already set
    
    # Print configuration (iterate fields without building full dict)
    print("\nConfiguration:")
    for f in fields(config):
        print(f"  {f.name}: {getattr(config, f.name)}")
    
    # Step 1: Prepare data
    data_processor = DataProcessor(config)
    data_processor.prepare_data()
    
    # Step 2: Load model
    trainer = Trainer(config)
    model = trainer.load_model()

    # Optionally move only speech_tokenizer.decoder to CPU (saves VRAM; max workers with data tokenizer on CPU)
    if config.speech_tokenizer_decoder_on_cpu:
        _move_speech_tokenizer_decoder_to_cpu(model)
    
    # Step 3: Get dataloaders
    config_obj = AutoConfig.from_pretrained(config.init_model_path)
    _ensure_config_has_speaker_languages(config_obj, config.train_speakers)
    train_dataloader = data_processor.get_train_dataloader(model, config_obj)
    eval_dataloader = data_processor.get_eval_dataloader(model, config_obj)

    # Load ref audio/mel for each train speaker (required when batch has "speakers")
    speakers_list = [s.strip() for s in config.train_speakers.split(",") if s.strip()]
    if speakers_list:
        load_speaker_refs(speakers_list)

    # Print data used and how long to train
    _training_summary(config, train_dataloader)

    # Step 4: Train
    trainer.train(model, train_dataloader, eval_dataloader)

    # Step 5: Upload to HuggingFace
    trainer.upload_to_huggingface()
    trainer.cleanup()

    # Finish WandB
    if config.use_wandb:
        trainer.accelerator.end_training()
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()
