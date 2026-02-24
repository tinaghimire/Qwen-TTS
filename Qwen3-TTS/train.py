#!/usr/bin/env python3
# coding=utf-8
"""
Unified Training Script for Qwen3-TTS Fine-tuning.

Main Workflow:
1. Load/prepare dataset
2. Load models
3. Login to WandB
4. Fine-tune with parameters

Features:
- Supports multiple data modes (jsonl, direct, none)
- Uses 25Hz tokenizer
- Layer replacement and addition
- Training loss logging to training_log.jsonl
- Validation loss and metrics logging to validation_log.jsonl
- WandB tracking
- Best and last model checkpointing
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

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS", "finetuning"))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts import Qwen3TTSTokenizer
from safetensors.torch import save_file
from finetuning.dataset import TTSDataset
from finetuning.layer_utils import replace_and_add_layers, print_model_summary, get_trainable_params

# Load environment variables
load_dotenv()


@dataclass
class TrainingConfig:
    """Training configuration loaded from .env file."""
    
    
    # Data Configuration - Mode for loading data
    
    data_mode: str = os.getenv("DATA_MODE", "jsonl")  # Options: jsonl (use prepared JSONL files), direct (load from HuggingFace), none (skip preparation)
    dataset_name: str = os.getenv("DATASET_NAME", "vaghawan/hausa-tts-22k")  # HuggingFace dataset name (used for direct mode)
    train_jsonl: str = os.getenv("TRAIN_JSONL", "./data/train.jsonl")  # Path to training JSONL file (used for jsonl mode)
    validation_jsonl: str = os.getenv("VALIDATION_JSONL", "./data/validation.jsonl")  # Path to validation JSONL file (optional)
    
    
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
    max_eval_samples: Optional[int] = int(os.getenv("MAX_EVAL_SAMPLES")) if os.getenv("MAX_EVAL_SAMPLES") else None  # Max validation samples (for debugging)
    
    
    # Reference Audio - Reference audio for speaker cloning
    
    ref_audio_path: Optional[str] = os.getenv("REF_AUDIO_PATH", None)  # Path to reference audio file (optional)
    ref_text: str = os.getenv("REF_TEXT", "MTN Entertainment and Lifestyle. Entertainment and Lifestyle are at the heart of MTN's offering. We bring you music, movies, games and more through our digital platforms. With MTN musicals, you can stream your favorite")  # Reference text for ICL (optional)


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
            "learning_rate": learning_rate,
            "timestamp": datetime.datetime.now().isoformat(),
            **kwargs
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
    
    def prepare_data(self):
        """Prepare data based on mode."""
        print("="*60)
        print("Step 1: Data Preparation")
        print("="*60)
        print(f"Data mode: {self.config.data_mode}")
        
        if self.config.data_mode == "none":
            print("Skipping data preparation (DATA_MODE=none)")
            return
        
        elif self.config.data_mode == "jsonl":
            # Check if JSONL files exist
            train_exists = os.path.exists(self.config.train_jsonl)
            val_exists = self.config.validation_jsonl and os.path.exists(self.config.validation_jsonl)
            
            if train_exists and (val_exists or not self.config.validation_jsonl):
                print(f"✓ Local JSONL files already exist:")
                print(f"  Train: {self.config.train_jsonl}")
                if val_exists:
                    print(f"  Validation: {self.config.validation_jsonl}")
                print("Skipping data preparation")
                return
            
            # Prepare data from HuggingFace to JSONL
            print(f"Preparing data from HuggingFace to JSONL...")
            print(f"Dataset: {self.config.dataset_name}")
            
            from data_processing import JSONLDataPreparer
            
            preparer = JSONLDataPreparer(
                dataset_name=self.config.dataset_name,
                output_dir=os.path.dirname(self.config.train_jsonl),
                tokenizer_path=self.config.tokenizer_path,
                device=self.config.device,
            )
            
            # Prepare train split
            preparer.prepare_split("train", max_samples=self.config.max_train_samples)
            
            # Prepare validation split if needed
            if self.config.validation_jsonl:
                preparer.prepare_split("validation", max_samples=self.config.max_eval_samples)
            
            print("Data preparation complete!")
        
        elif self.config.data_mode == "direct":
            print(f"Using direct HuggingFace loading mode")
            print(f"Dataset: {self.config.dataset_name}")
            print(f"Data will be loaded on-the-fly during training")
            print("No data preparation needed")
        
        else:
            raise ValueError(f"Invalid DATA_MODE: {self.config.data_mode}")
    
    def get_train_dataloader(self, model, config) -> DataLoader:
        """Get training dataloader based on mode."""
        if self.config.data_mode == "jsonl":
            # Load from JSONL
            train_data = []
            with open(self.config.train_jsonl, 'r') as f:
                for line in f:
                    train_data.append(json.loads(line))
            
            if self.config.max_train_samples:
                train_data = train_data[:self.config.max_train_samples]
            
            dataset = TTSDataset(train_data, model.processor, config)
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        
        elif self.config.data_mode == "direct":
            # Load directly from HuggingFace
            from data_processing import HFDirectDataLoader
            
            dataset = HFDirectDataLoader(
                dataset_name=self.config.dataset_name,
                split="train",
                tokenizer_path=self.config.tokenizer_path,
                max_samples=self.config.max_train_samples,
                audio_sr_device=self.config.device,
            )
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        else:
            raise ValueError(f"Invalid DATA_MODE: {self.config.data_mode}")
    
    def get_eval_dataloader(self, model, config) -> Optional[DataLoader]:
        """Get evaluation dataloader based on mode."""
        if not self.config.validation_jsonl:
            return None
        
        if self.config.data_mode == "jsonl":
            # Load from JSONL
            val_data = []
            with open(self.config.validation_jsonl, 'r') as f:
                for line in f:
                    val_data.append(json.loads(line))
            
            if self.config.max_eval_samples:
                val_data = val_data[:self.config.max_eval_samples]
            
            dataset = TTSDataset(val_data, model.processor, config)
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        
        elif self.config.data_mode == "direct":
            # Load directly from HuggingFace
            from data_processing import HFDirectDataLoader
            
            dataset = HFDirectDataLoader(
                dataset_name=self.config.dataset_name,
                split="validation",
                tokenizer_path=self.config.tokenizer_path,
                max_samples=self.config.max_eval_samples,
                audio_sr_device=self.config.device,
            )
            return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
        else:
            return None


class Trainer:
    """Main trainer class."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = Logger(config.output_dir)
        self.target_speaker_embedding = None
        self.best_val_loss = float('inf')
        self.best_val_metrics = {}
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
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
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            print(f"✓ Model loaded with flash_attention_2")
        except ImportError as e:
            print(f"⚠ Flash attention not available, falling back to SDPA")
            print(f"   (You can install flash-attn for potentially faster training)")
            model = Qwen3TTSModel.from_pretrained(
                self.config.init_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            print(f"✓ Model loaded with SDPA (Scaled Dot Product Attention)")
        except Exception as e:
            print(f"⚠ Error loading model with flash_attention_2: {e}")
            print(f"   Trying SDPA fallback...")
            model = Qwen3TTSModel.from_pretrained(
                self.config.init_model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            print(f"✓ Model loaded with SDPA")
        
        # Freeze or unfreeze speaker encoder
        if self.config.freeze_speaker_encoder:
            print(f"\nFreezing speaker encoder...")
            for param in model.speaker_encoder.parameters():
                param.requires_grad = False
            print(f"✓ Speaker encoder frozen")
        else:
            print(f"\nUnfreezing speaker encoder for finetuning...")
            for param in model.speaker_encoder.parameters():
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
        
        for name, param in model.named_parameters():
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
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
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
        
        model.train()
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            for step, batch in enumerate(train_dataloader):
                with self.accelerator.accumulate(model):
                    loss = self.training_step(model, batch)
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                global_step += 1
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    self.logger.log_training(
                        step=global_step,
                        epoch=epoch,
                        loss=loss.item(),
                        learning_rate=current_lr
                    )
                    
                    if self.config.use_wandb:
                        self.accelerator.log({
                            "train/loss": loss.item(),
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch,
                            "train/global_step": global_step
                        }, step=global_step)
                    
                    if self.accelerator.is_main_process:
                        print(f"Epoch {epoch} | Step {global_step} | Loss: {loss.item():.4f} | LR: {current_lr:.2e}")
                
                # Evaluation
                if eval_dataloader and global_step % self.config.eval_steps == 0:
                    val_loss, val_metrics = self.evaluate(model, eval_dataloader, global_step, epoch)
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.best_val_metrics = val_metrics
                        self.save_checkpoint(model, optimizer, scheduler, global_step, epoch, "best")
                
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self.save_checkpoint(model, optimizer, scheduler, global_step, epoch, "checkpoint")
            
            # End of epoch evaluation
            if eval_dataloader:
                val_loss, val_metrics = self.evaluate(model, eval_dataloader, global_step, epoch)
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_val_metrics = val_metrics
                    self.save_checkpoint(model, optimizer, scheduler, global_step, epoch, "best")
        
        # Save last checkpoint
        self.save_checkpoint(model, optimizer, scheduler, global_step, epoch, "last")
        
        print("\n" + "="*60)
        print("Training complete!")
        print("="*60)
    
    def training_step(self, model, batch):
        """Perform one training step."""
        input_ids = batch['input_ids']
        codec_ids = batch['codec_ids']
        ref_mels = batch['ref_mels']
        text_embedding_mask = batch['text_embedding_mask']
        codec_embedding_mask = batch['codec_embedding_mask']
        attention_mask = batch['attention_mask']
        codec_0_labels = batch['codec_0_labels']
        codec_mask = batch['codec_mask']
        
        # Get speaker embedding
        speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype))
        if self.target_speaker_embedding is None:
            self.target_speaker_embedding = speaker_embedding.detach()
        
        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]
        
        input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
        input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
        input_codec_embedding[:, 6, :] = speaker_embedding
        
        input_embeddings = input_text_embedding + input_codec_embedding
        
        for i in range(1, 16):
            codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_i_embedding
        
        outputs = model.talker(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=codec_0_labels[:, 1:],
            output_hidden_states=True
        )
        
        hidden_states = outputs.hidden_states[0][-1]
        talker_hidden_states = hidden_states[codec_mask[:, 1:]]
        talker_codec_ids = codec_ids[codec_mask]
        
        sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
        
        loss = outputs.loss + self.config.sub_talker_loss_weight * sub_talker_loss
        
        return loss
    
    def evaluate(self, model, eval_dataloader, step, epoch):
        """Evaluate model on validation set."""
        print(f"\nEvaluating at step {step}...")
        model.eval()
        
        total_loss = 0
        total_samples = 0
        speaker_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                loss = self.training_step(model, batch)
                total_loss += loss.item() * batch['input_ids'].size(0)
                total_samples += batch['input_ids'].size(0)
                
                # Collect speaker embeddings for similarity calculation
                ref_mels = batch['ref_mels']
                speaker_emb = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype))
                speaker_embeddings.append(speaker_emb.cpu())
        
        avg_loss = total_loss / total_samples
        
        # Calculate speaker embedding consistency (similarity within batch)
        if len(speaker_embeddings) > 1:
            all_embeddings = torch.cat(speaker_embeddings, dim=0)
            # Calculate cosine similarity matrix
            embeddings_norm = torch.nn.functional.normalize(all_embeddings, p=2, dim=-1)
            similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
            # Get upper triangle (excluding diagonal)
            upper_tri = similarity_matrix.triu(diagonal=1)
            avg_similarity = upper_tri.mean().item()
        else:
            avg_similarity = 1.0  # Only one batch, perfect consistency
        
        # Calculate metrics
        metrics = {
            "speaker_embedding_consistency": avg_similarity,
            "perplexity": torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 20 else float('inf'),
        }
        
        # Log validation
        self.logger.log_validation(
            step=step,
            epoch=epoch,
            loss=avg_loss,
            metrics=metrics
        )
        
        if self.config.use_wandb:
            self.accelerator.log({
                "val/loss": avg_loss,
                "val/perplexity": metrics["perplexity"],
                "val/speaker_embedding_consistency": metrics["speaker_embedding_consistency"],
                "val/step": step,
                "val/epoch": epoch
            }, step=step)
        
        if self.accelerator.is_main_process:
            print(f"Validation Loss: {avg_loss:.4f}")
            print(f"Validation Perplexity: {metrics['perplexity']:.4f}")
            print(f"Speaker Embedding Consistency: {metrics['speaker_embedding_consistency']:.4f}")
        
        model.train()
        return avg_loss, metrics
    
    def save_checkpoint(self, model, optimizer, scheduler, step, epoch, checkpoint_type):
        """Save model checkpoint."""
        if not self.accelerator.is_main_process:
            return
        
        output_dir = os.path.join(self.config.output_dir, checkpoint_type)
        os.makedirs(output_dir, exist_ok=True)
        
        # Unwrap model
        unwrapped_model = self.accelerator.unwrap_model(model)
        
        # Save model weights
        state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
        
        # Count the number of layer weights being saved
        layer_keys = [k for k in state_dict.keys() if 'talker.model.layers' in k]
        unique_layers = set()
        for key in layer_keys:
            parts = key.split('.')
            if 'layers' in parts:
                layer_idx = parts[parts.index('layers') + 1]
                unique_layers.add(layer_idx)
        print(f"Saving {len(unique_layers)} layers in checkpoint")
        
        # Add speaker embedding
        if self.target_speaker_embedding is not None:
            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = self.target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
        
        # Save safetensors
        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, save_path)
        print(f"✓ Saved model.safetensors")
        
        # Save configuration
        config_dict = unwrapped_model.config.to_dict()
        config_dict["tts_model_type"] = "custom_voice"
        talker_config = config_dict.get("talker_config", {})
        talker_config["spk_id"] = {self.config.speaker_name: 3000}
        talker_config["spk_is_dialect"] = {self.config.speaker_name: False}
        config_dict["talker_config"] = talker_config
        
        # Verify the layer count in the saved configuration
        saved_num_layers = config_dict["talker_config"]["num_hidden_layers"]
        actual_num_layers = len(unwrapped_model.talker.model.layers)
        print(f"Saving checkpoint with {saved_num_layers} layers (actual: {actual_num_layers})")
        if saved_num_layers != actual_num_layers:
            print(f"⚠ Warning: Configuration layer count ({saved_num_layers}) doesn't match actual layers ({actual_num_layers})")
            config_dict["talker_config"]["num_hidden_layers"] = actual_num_layers
        
        config_file = os.path.join(output_dir, "config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved config.json")
        
        # Save generation_config.json if available
        if hasattr(unwrapped_model, 'generate_config') and unwrapped_model.generate_config is not None:
            generation_config_file = os.path.join(output_dir, "generation_config.json")
            with open(generation_config_file, 'w', encoding='utf-8') as f:
                json.dump(unwrapped_model.generate_config, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved generation_config.json")
        
        # Save processor config from the loaded model (includes tokenizer files)
        if unwrapped_model.processor is not None:
            unwrapped_model.processor.save_pretrained(output_dir)
            print(f"✓ Saved processor and tokenizer files")
        
        # Save speech tokenizer separately
        if hasattr(unwrapped_model, 'speech_tokenizer') and unwrapped_model.speech_tokenizer is not None:
            speech_tokenizer_dir = os.path.join(output_dir, "speech_tokenizer")
            os.makedirs(speech_tokenizer_dir, exist_ok=True)
            unwrapped_model.speech_tokenizer.save_pretrained(speech_tokenizer_dir)
            print(f"✓ Saved speech_tokenizer to {speech_tokenizer_dir}")
        
        # Save speaker encoder for the new speaker (optional, for reference)
        if hasattr(unwrapped_model, 'speaker_encoder') and unwrapped_model.speaker_encoder is not None:
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
    
    # Step 4: Train
    trainer.train(model, train_dataloader, eval_dataloader)
    
    # Step 5: Upload to HuggingFace
    trainer.upload_to_huggingface()
    
    # Finish WandB
    if config.use_wandb:
        trainer.accelerator.end_training()
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()
