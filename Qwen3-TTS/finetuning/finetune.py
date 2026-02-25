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
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

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

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS", "finetuning"))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts import Qwen3TTSTokenizer
from safetensors.torch import save_file
from finetuning.dataset import TTSDataset
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


def _voices_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voices")


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
    logger.info(f"Loaded ref_audio/ref_mel for speakers: {list(REF_MEL_CACHE.keys())}")


def ref_mels_for_speakers(speakers: List[str], device: torch.device) -> torch.Tensor:
    """Build (B, T, 128) ref_mels tensor from list of speaker ids using REF_MEL_CACHE."""
    ref_list = [REF_MEL_CACHE[s] for s in speakers]
    max_t = max(m.shape[1] for m in ref_list)
    padded = []
    for m in ref_list:
        if m.shape[1] < max_t:
            m = torch.nn.functional.pad(m, (0, 0, 0, max_t - m.shape[1]))
        padded.append(m)
    return torch.cat(padded, dim=0).to(device)


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
    
    
    # Data Limits - Max samples to use for testing
    
    max_train_samples: Optional[int] = int(os.getenv("MAX_TRAIN_SAMPLES")) if os.getenv("MAX_TRAIN_SAMPLES") else None  # Max training samples (for debugging)
    max_eval_samples: Optional[int] = int(os.getenv("MAX_VAL_SAMPLES")) if os.getenv("MAX_EVAL_SAMPLES") else None  # Max validation samples (for debugging)
    cache_dir: str = os.getenv("CACHE_DIR", "./cache")  # Cache directory for streaming mode
    
    
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
        
        with open(self.training_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_validation(self, step: int, epoch: int, loss: float, metrics: Dict[str, float], **kwargs):
        """Log validation metrics to validation_log.jsonl."""
        log_entry = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
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
        """Get training dataloader from HuggingFace (combined multi-speaker)."""
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
            batch_size=self.config.batch_size,
            max_samples=self.config.max_train_samples,
            tokenizer_path=self.config.tokenizer_path,
            cache_dir=self.config.cache_dir,
            num_workers=None,
        )
    
    def get_eval_dataloader(self, model, config) -> Optional[DataLoader]:
        """Get validation dataloader from HuggingFace (combined multi-speaker)."""
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
            batch_size=self.config.batch_size,
            max_samples=self.config.max_eval_samples,
            tokenizer_path=self.config.tokenizer_path,
            cache_dir=self.config.cache_dir,
            shuffle=False,
            num_workers=None,
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
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision="no",  # Disable mixed precision - use model's native dtype
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
        try:
            model = Qwen3TTSModel.from_pretrained(
                self.config.init_model_path,
                torch_dtype=torch.float32,
                attn_implementation="flash_attention_2",
            )
            print(f"✓ Model loaded with flash_attention_2")
        except ImportError as e:
            print(f"⚠ Flash attention not available, falling back to SDPA")
            print(f"   (You can install flash-attn for potentially faster training)")
            model = Qwen3TTSModel.from_pretrained(
                self.config.init_model_path,
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
            )
            print(f"✓ Model loaded with SDPA (Scaled Dot Product Attention)")
        except Exception as e:
            print(f"⚠ Error loading model with flash_attention_2: {e}")
            print(f"   Trying SDPA fallback...")
            model = Qwen3TTSModel.from_pretrained(
                self.config.init_model_path,
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
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
        
        total_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Prepare with accelerator
        model, optimizer, train_dataloader, scheduler = self.accelerator.prepare(
            model, optimizer, train_dataloader, scheduler
        )
        
        if eval_dataloader:
            eval_dataloader = self.accelerator.prepare(eval_dataloader)
        
        model.model.train()
        global_step = 0
        total_steps = len(train_dataloader) * self.config.num_epochs
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            # Create progress bar for this epoch
            pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.config.num_epochs}",
                unit="batch",
                disable=not self.accelerator.is_main_process
            )
            
            for step, batch in enumerate(pbar):
                with self.accelerator.accumulate(model):
                    loss = self.training_step(model, batch, global_step=global_step)
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.model.parameters(), self.config.max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                global_step += 1
                
                # Calculate progress percentage
                progress_pct = (global_step / total_steps) * 100
                epoch_progress = ((step + 1) / len(train_dataloader)) * 100
                
                # Update progress bar
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{current_lr:.2e}',
                    'progress': f'{progress_pct:.1f}%'
                })
                
                # Log to WandB at every step for clear graph visualization
                if self.config.use_wandb:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch,
                        "train/global_step": global_step,
                        "train/progress_pct": progress_pct,
                        "train/epoch_progress": epoch_progress
                    }
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
                        epoch_progress=epoch_progress
                    )
                    
                    if self.accelerator.is_main_process:
                        print(f"\n{'='*60}")
                        print(f"📊 Training Progress - Step {global_step}/{total_steps}")
                        print(f"{'='*60}")
                        print(f"  Overall Progress: {progress_pct:.2f}%")
                        print(f"  Epoch {epoch + 1} Progress: {epoch_progress:.2f}% ({step + 1}/{len(train_dataloader)} steps)")
                        print(f"  Loss: {loss.item():.4f}")
                        print(f"  Learning Rate: {current_lr:.10e}")
                        print(f"{'='*60}")
                
                # Evaluation
                if eval_dataloader and global_step % self.config.eval_steps == 0:
                    print(f"\n{'='*60}")
                    print(f"🔍 Running Validation at Step {global_step}")
                    print(f"{'='*60}")
                    val_loss, val_metrics = self.evaluate(model, eval_dataloader, global_step, epoch)
                    
                    if self.accelerator.is_main_process:
                        print(f"\n{'='*60}")
                        print(f"📈 Validation Results - Step {global_step}")
                        print(f"{'='*60}")
                        print(f"  Validation Loss: {val_loss:.4f}")
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
            if eval_dataloader:
                print(f"\n{'='*60}")
                print(f"🔍 End of Epoch {epoch + 1} Validation")
                print(f"{'='*60}")
                val_loss, val_metrics = self.evaluate(model, eval_dataloader, global_step, epoch)
                
                if self.accelerator.is_main_process:
                    print(f"\n{'='*60}")
                    print(f"📈 Epoch {epoch + 1} Validation Results")
                    print(f"{'='*60}")
                    print(f"  Validation Loss: {val_loss:.4f}")
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
        if "speakers" in batch:
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
        input_codec_embedding[:, 6, :] = speaker_embedding
        
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
        
        loss = outputs.loss + self.config.sub_talker_loss_weight * sub_talker_loss

        if self.config.use_auxiliary_losses and self.dual_loss_trainer is not None and "target_mel" in batch and "target_audio" in batch:
            # Every audio_loss_every_n_steps, generate waveforms and include waveform/voice/prosody losses
            generated_audio_tensor = None
            if global_step is not None and self.config.audio_loss_every_n_steps > 0 and global_step % self.config.audio_loss_every_n_steps == 0:
                with torch.no_grad():
                    gen_list = self._generate_audio_from_batch(model, batch)
                if gen_list is not None:
                    generated_audio_tensor = self._pad_wav_list_to_tensor(gen_list, device)

            aux = self.dual_loss_trainer.compute_auxiliary_losses(
                batch=batch,
                model=model,
                outputs=outputs,
                sample_rate=24000,
                generated_audio=generated_audio_tensor,
            )
            loss = loss + self.config.mel_reconstruction_weight * aux["mel_reconstruction_loss"]
            if self.config.reconstruction_weight and aux["waveform_reconstruction_loss"].item() != 0:
                loss = loss + self.config.reconstruction_weight * aux["waveform_reconstruction_loss"]
            if self.config.voice_consistency_weight and aux["voice_consistency_loss"].item() != 0:
                loss = loss + self.config.voice_consistency_weight * aux["voice_consistency_loss"]
            if self.config.prosody_weight and aux["prosody_loss"].item() != 0:
                loss = loss + self.config.prosody_weight * aux["prosody_loss"]

            self._last_aux = {k: v.item() for k, v in aux.items() if torch.is_tensor(v)}
        else:
            self._last_aux = None
        return loss

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
            codes_list.append(codes_i.cpu())

        valid_codes = [c for c in codes_list if c is not None and c.shape[0] > 0]
        valid_indices = [i for i in range(B) if codes_list[i] is not None and codes_list[i].shape[0] > 0]
        if not valid_codes:
            return None
        if not hasattr(model.model, "speech_tokenizer") or model.model.speech_tokenizer is None:
            return None
        try:
            wavs, sr = model.model.speech_tokenizer.decode([{"audio_codes": c} for c in valid_codes])
        except Exception:
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
        valid = [i for i, w in enumerate(wav_list) if w is not None and len(w) > 0]
        if not valid:
            return None
        max_len = max(len(wav_list[i]) for i in valid)
        B = len(wav_list)
        out = np.zeros((B, max_len), dtype=np.float32)
        for i in range(B):
            if wav_list[i] is not None and len(wav_list[i]) > 0:
                L = len(wav_list[i])
                w = np.asarray(wav_list[i], dtype=np.float32).flatten()
                out[i, :L] = w[:L]
        return torch.from_numpy(out).to(device)

    def evaluate(self, model, eval_dataloader, step, epoch):
        """Evaluate model on validation set. Reports EVALUATION_METRICS."""
        print(f"\nEvaluating at step {step}...")
        model.model.eval()

        total_loss = 0
        total_samples = 0
        speaker_embeddings = []
        sample_batch_for_quality = None  # one batch with target_audio for quality_metrics

        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                loss = self.training_step(model, batch)
                total_loss += loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)

                if "speakers" in batch:
                    ref_mels = ref_mels_for_speakers(batch["speakers"], model.model.device)
                else:
                    ref_mels = batch["ref_mels"].to(model.model.device)
                speaker_emb = model.model.speaker_encoder(ref_mels)
                speaker_embeddings.append(speaker_emb.cpu())

                if sample_batch_for_quality is None and "target_audio" in batch and "target_audio_lengths" in batch:
                    sample_batch_for_quality = batch

        avg_loss = total_loss / total_samples

        # Speaker embedding consistency (similarity within batch)
        if len(speaker_embeddings) > 1:
            all_embeddings = torch.cat(speaker_embeddings, dim=0)
            embeddings_norm = torch.nn.functional.normalize(all_embeddings, p=2, dim=-1)
            similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
            upper_tri = similarity_matrix.triu(diagonal=1)
            avg_similarity = upper_tri.mean().item()
        else:
            avg_similarity = 1.0

        # Build metrics with all five EVALUATION_METRICS
        metrics = {
            "perplexity": torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 20 else float("inf"),
            "speaker_embedding_consistency": avg_similarity,
            "pronunciation_accuracy": 0.0,
            "tonal_accuracy": 0.0,
            "prosody_accuracy": 0.0,
        }

        # When we have target audio, generate audio from the model and compute all five
        # metrics with QualityMetricsCalculator(generated_audio, target_audio).
        if sample_batch_for_quality is not None and self.accelerator.is_main_process:
            try:
                with torch.no_grad():
                    gen_wavs = self._generate_audio_from_batch(model, sample_batch_for_quality)
                if gen_wavs is not None:
                    qcalc = QualityMetricsCalculator(sample_rate=24000)
                    target_audio = sample_batch_for_quality["target_audio"]
                    target_audio_lengths = sample_batch_for_quality["target_audio_lengths"]
                    B = min(len(gen_wavs), target_audio.shape[0])
                    scores = {k: [] for k in EVALUATION_METRICS}
                    for i in range(B):
                        L = int(target_audio_lengths[i].item())
                        if L < 100 or i >= len(gen_wavs) or gen_wavs[i] is None or len(gen_wavs[i]) < 100:
                            continue
                        ref_np = target_audio[i, :L].float().cpu().numpy()
                        gen_np = gen_wavs[i] if isinstance(gen_wavs[i], np.ndarray) else np.array(gen_wavs[i], dtype=np.float32)
                        # Trim or pad gen to similar length for metric (quality_metrics pads internally)
                        m = qcalc.calculate_all_metrics(gen_np, ref_np)
                        for k in EVALUATION_METRICS:
                            scores[k].append(m[k])
                    for k in EVALUATION_METRICS:
                        if scores[k]:
                            metrics[k] = float(np.mean(scores[k]))
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
                "val/step": step,
                "val/epoch": epoch,
            }
            for k in EVALUATION_METRICS:
                log_dict[f"val/{k}"] = metrics[k]
            self.accelerator.log(log_dict, step=step)

        if self.accelerator.is_main_process:
            print(f"Validation Loss: {avg_loss:.4f}")
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
        """Save model checkpoint."""
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
        
        # Save model weights (access the underlying model directly)
        state_dict = {k: v.detach().to("cpu") for k, v in model.model.state_dict().items()}
        
        # Count the number of layer weights being saved
        layer_keys = [k for k in state_dict.keys() if 'talker.model.layers' in k]
        unique_layers = set()
        for key in layer_keys:
            parts = key.split('.')
            if 'layers' in parts:
                layer_idx = parts[parts.index('layers') + 1]
                unique_layers.add(layer_idx)
        print(f"  Saving {len(unique_layers)} layers in checkpoint")
        
        # Add speaker embedding
        if self.target_speaker_embedding is not None:
            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = self.target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
        
        # Save safetensors
        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, save_path)
        print(f"✓ Saved model.safetensors")
        
        # Save configuration
        config_dict = model.model.config.to_dict()
        config_dict["tts_model_type"] = "custom_voice"
        talker_config = config_dict.get("talker_config", {})
        talker_config["spk_id"] = {self.config.speaker_name: 3000}
        talker_config["spk_is_dialect"] = {self.config.speaker_name: False}
        config_dict["talker_config"] = talker_config
        
        # Verify the layer count in the saved configuration
        saved_num_layers = config_dict["talker_config"]["num_hidden_layers"]
        actual_num_layers = len(model.model.talker.model.layers)
        print(f"Saving checkpoint with {saved_num_layers} layers (actual: {actual_num_layers})")
        if saved_num_layers != actual_num_layers:
            print(f"⚠ Warning: Configuration layer count ({saved_num_layers}) doesn't match actual layers ({actual_num_layers})")
            config_dict["talker_config"]["num_hidden_layers"] = actual_num_layers
        
        config_file = os.path.join(output_dir, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved config.json")
        
        # Save generation_config.json if available
        if hasattr(model.model, 'generate_config') and model.model.generate_config is not None:
            generation_config_file = os.path.join(output_dir, "generation_config.json")
            with open(generation_config_file, 'w', encoding='utf-8') as f:
                json.dump(model.model.generate_config, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved generation_config.json")
        
        # Save processor config from the loaded model (includes tokenizer files)
        if hasattr(model.model, 'processor') and model.model.processor is not None:
            model.model.processor.save_pretrained(output_dir)
            print(f"✓ Saved processor and tokenizer files")
        
        # Save speech tokenizer separately
        if hasattr(model.model, 'speech_tokenizer') and model.model.speech_tokenizer is not None:
            speech_tokenizer_dir = os.path.join(output_dir, "speech_tokenizer")
            os.makedirs(speech_tokenizer_dir, exist_ok=True)
            # The speech_tokenizer is a wrapper, save its underlying model and feature_extractor
            model.model.speech_tokenizer.model.save_pretrained(speech_tokenizer_dir)
            model.model.speech_tokenizer.feature_extractor.save_pretrained(speech_tokenizer_dir)
            print(f"✓ Saved speech_tokenizer to {speech_tokenizer_dir}")
        
        # Save speaker encoder for the new speaker (optional, for reference)
        if hasattr(model.model, 'speaker_encoder') and model.model.speaker_encoder is not None:
            speaker_encoder_dir = os.path.join(output_dir, "speaker_encoder")
            os.makedirs(speaker_encoder_dir, exist_ok=True)
            # Save speaker encoder config
            speaker_encoder_config = {
                "model_type": "qwen3_tts_speaker_encoder",
                "speaker_name": self.config.speaker_name,
                "speaker_embedding_dim": self.target_speaker_embedding.shape[-1] if self.target_speaker_embedding is not None else 1024
            }
            speaker_encoder_config_file = os.path.join(speaker_encoder_dir, "speaker_config.json")
            with open(speaker_encoder_config_file, 'w', encoding='utf-8') as f:
                json.dump(speaker_encoder_config, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved speaker encoder config for speaker: {self.config.speaker_name}")
        
        # Save training state
        training_state = {
            "step": step,
            "epoch": epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_metrics": self.best_val_metrics,
            "config": asdict(self.config),
        }
        
        state_file = os.path.join(output_dir, "training_state.json")
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(training_state, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved training_state.json")
        
        # List all saved files
        print(f"\n{'='*60}")
        print(f"Checkpoint contents:")
        print(f"{'='*60}")
        saved_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, output_dir)
                saved_files.append(rel_path)
                print(f"  - {rel_path}")
        print(f"{'='*60}")
        print(f"Total files saved: {len(saved_files)}")
        print(f"{'='*60}\n")
        
        # Create a README file in the checkpoint directory
        readme_content = f"""# Fine-Tuned Qwen3-TTS Model Checkpoint

## Model Information
- **Speaker Name**: {self.config.speaker_name}
- **Base Model**: {self.config.init_model_path}
- **Number of Layers**: {config_dict.get('talker_config', {}).get('num_hidden_layers', 'N/A')}
- **Hidden Size**: {config_dict.get('talker_config', {}).get('hidden_size', 'N/A')}
- **Training Epoch**: {epoch}
- **Training Step**: {step}
- **Best Validation Loss**: {self.best_val_loss:.4f}

## Training Configuration
- **Learning Rate**: {self.config.learning_rate}
- **Batch Size**: {self.config.batch_size}
- **Gradient Accumulation Steps**: {self.config.gradient_accumulation_steps}
- **Weight Decay**: {self.config.weight_decay}
- **Warmup Steps**: {self.config.warmup_steps}
- **Speaker Encoder Frozen**: {self.config.freeze_speaker_encoder}
- **Layer Replacement**: {self.config.replace_last_n_layers} layers replaced, {self.config.add_new_layers} layers added
- **Original Layers Frozen**: {self.config.freeze_original_layers}

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
The model has been fine-tuned for speaker: **{self.config.speaker_name}**
Speaker embedding is stored at index 3000 in the codec embedding layer.
Speaker encoder weights are included in the checkpoint and have been fine-tuned.

## Notes
- This model uses the Qwen3-TTS tokenizer
- The model supports streaming generation
- For best results, use reference audio from the same speaker used during training
- The speaker encoder has been fine-tuned to better capture speaker characteristics
"""
        readme_file = os.path.join(output_dir, "README.md")
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print(f"✓ Created README.md with usage instructions")

        print(f"✓ Saved {checkpoint_type} checkpoint to {output_dir}")

        # Upload to HuggingFace if configured
        if checkpoint_type in ["best", "last"]:
            if checkpoint_type == "best":
                repo_id = self.config.hf_best_model_repo
            else:
                repo_id = self.config.hf_last_model_repo

            self._upload_checkpoint_to_hf(output_dir, repo_id, checkpoint_type)
    
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
        """Cleanup background preprocessors."""
        print("\nCleaning up background preprocessors...")
        
        if self.train_preprocessor:
            self.train_preprocessor.stop()
        
        if self.eval_preprocessor:
            self.eval_preprocessor.stop()
        
        print("✓ Cleanup complete")


def main():
    """Main training function."""
    print("="*60)
    print("Qwen3-TTS Training Pipeline")
    print("="*60)
    
    # Load configuration
    config = TrainingConfig()
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")
    
    # Step 1: Prepare data
    data_processor = DataProcessor(config)
    data_processor.prepare_data()
    
    # Step 2: Load model
    trainer = Trainer(config)
    model = trainer.load_model()
    
    # Step 3: Get dataloaders
    config_obj = AutoConfig.from_pretrained(config.init_model_path)
    train_dataloader = data_processor.get_train_dataloader(model, config_obj)
    eval_dataloader = data_processor.get_eval_dataloader(model, config_obj)

    # Load ref audio/mel for each train speaker (required when batch has "speakers")
    speakers_list = [s.strip() for s in config.train_speakers.split(",") if s.strip()]
    if speakers_list:
        load_speaker_refs(speakers_list)
    
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
