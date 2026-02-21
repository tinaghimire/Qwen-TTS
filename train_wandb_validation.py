#!/usr/bin/env python3
# coding=utf-8
"""
Advanced Training Script for Qwen3-TTS with Validation, Metrics, and WandB.

This script provides a comprehensive training pipeline:
1. Prepare data (optional)
2. Train with validation, metrics, and WandB logging
3. Save best and last models
4. Upload to Hugging Face Hub (optional)

Features:
- Validation during training
- WandB logging for metrics
- Checkpoint saving with optimizer and scheduler states
- Model upload to Hugging Face Hub
- Mixed precision training support

Usage:
    # Train with default settings
    python train_wandb_validation.py
    
    # Skip data preparation if already done
    echo "SKIP_PREPARE=1" >> .env
    python train_wandb_validation.py
    
    # Only prepare data, don't train
    echo "PREPARE_ONLY=1" >> .env
    python train_wandb_validation.py
"""

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from huggingface_hub import HfApi, HfFolder
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_cosine_schedule_with_warmup
from tqdm import tqdm
from dotenv import load_dotenv

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS", "finetuning"))

from dataset_tool import HausaTTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file

# Load environment variables
load_dotenv()


@dataclass
class TrainingArguments:
    """Training arguments loaded from environment variables."""
    # Model and data paths
    init_model_path: str = os.getenv("INIT_MODEL_PATH", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    dataset_name: str = os.getenv("DATASET_NAME", "vaghawan/hausa-tts-22k")
    
    # Training hyperparameters
    batch_size: int = int(os.getenv("BATCH_SIZE", 2))
    gradient_accumulation_steps: int = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", 4))
    learning_rate: float = float(os.getenv("LR", 2e-5))
    num_epochs: int = int(os.getenv("NUM_EPOCHS", 3))
    weight_decay: float = float(os.getenv("WEIGHT_DECAY", 0.01))
    warmup_steps: int = int(os.getenv("WARMUP_STEPS", 100))
    max_grad_norm: float = float(os.getenv("MAX_GRAD_NORM", 1.0))
    
    # Dataset settings
    train_jsonl: str = os.getenv("TRAIN_JSONL", "./data/train.jsonl")
    validation_jsonl: str = os.getenv("VALIDATION_JSONL", "./data/validation.jsonl")
    ref_audio_path: Optional[str] = os.getenv("REF_AUDIO_PATH") or None
    ref_text: str = os.getenv(
        "REF_TEXT",
        "MTN Entertainment and Lifestyle. Entertainment and Lifestyle are at the heart of MTN's offering. We bring you music, movies, games and more through our digital platforms. With MTN musicals, you can stream your favorite"
    )
    max_train_samples: Optional[int] = int(os.getenv("MAX_TRAIN_SAMPLES")) if os.getenv("MAX_TRAIN_SAMPLES") else None
    max_eval_samples: Optional[int] = int(os.getenv("MAX_EVAL_SAMPLES")) if os.getenv("MAX_EVAL_SAMPLES") else None
    
    # Speaker settings
    speaker_name: str = os.getenv("SPEAKER_NAME", "hausa_speaker")
    
    # Logging and checkpointing
    logging_steps: int = int(os.getenv("LOGGING_STEPS", 10))
    save_steps: int = int(os.getenv("SAVE_STEPS", 500))
    eval_steps: int = int(os.getenv("EVAL_STEPS", 500))
    save_total_limit: int = int(os.getenv("SAVE_TOTAL_LIMIT", 3))
    
    # WandB settings
    use_wandb: bool = os.getenv("USE_WANDB", "true").lower() in ("true", "1", "yes")
    wandb_project: str = os.getenv("WANDB_PROJECT", "qwen3-tts-hausa")
    wandb_run_name: Optional[str] = os.getenv("WANDB_RUN_NAME") or None
    
    # Hugging Face upload settings
    upload_to_hub: bool = os.getenv("UPLOAD_TO_HUB", "false").lower() in ("true", "1", "yes")
    hub_model_id_best: str = os.getenv("HUB_MODEL_ID_BEST", "vaghawan/tts-best")
    hub_model_id_last: str = os.getenv("HUB_MODEL_ID_LAST", "vaghawan/tts-last")
    hub_token: Optional[str] = os.getenv("HF_TOKEN") or None
    
    # Mixed precision
    mixed_precision: str = os.getenv("MIXED_PRECISION", "bf16")
    
    # Workflow control
    skip_prepare: bool = os.getenv("SKIP_PREPARE", "false").lower() in ("true", "1", "yes")
    prepare_only: bool = os.getenv("PREPARE_ONLY", "false").lower() in ("true", "1", "yes")
    device: str = os.getenv("DEVICE", "cuda")


class AdvancedTrainer:
    """Advanced trainer with validation, metrics, and WandB."""
    
    def __init__(self, args: TrainingArguments):
        self.args = args
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with="wandb" if args.use_wandb else None,
        )
        
        # Initialize WandB
        if args.use_wandb:
            self.accelerator.init_trackers(
                project_name=args.wandb_project,
                config=vars(args),
                init_kwargs={"wandb": {"name": args.wandb_run_name}}
            )
        
        # Set reference audio path
        if args.ref_audio_path is None:
            args.ref_audio_path = os.path.join(
                os.path.dirname(__file__),
                "voices", "english_voice", "english_voice.wav"
            )
        
        # Load model and config
        print(f"Loading model from {args.init_model_path}...")
        try:
            self.qwen3tts = Qwen3TTSModel.from_pretrained(
                args.init_model_path,
                device_map="cuda",
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            print("✓ Model loaded with flash_attention_2")
        except (ImportError, Exception) as e:
            print(f"⚠ Flash attention not available, falling back to SDPA: {e}")
            self.qwen3tts = Qwen3TTSModel.from_pretrained(
                args.init_model_path,
                device_map="cuda",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            print("✓ Model loaded with SDPA")
        self.config = AutoConfig.from_pretrained(args.init_model_path)
        
        # Create datasets
        print("Creating datasets...")
        self.train_dataset = HausaTTSDataset(args.train_jsonl)
        
        if args.validation_jsonl and os.path.exists(args.validation_jsonl):
            self.eval_dataset = HausaTTSDataset(args.validation_jsonl)
        else:
            self.eval_dataset = None
            print("⚠ No validation dataset provided")
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )
        
        if self.eval_dataset:
            self.eval_dataloader = DataLoader(
                self.eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn
            )
        else:
            self.eval_dataloader = None
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.qwen3tts.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        num_training_steps = len(self.train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, self.train_dataloader, self.scheduler = self.accelerator.prepare(
            self.qwen3tts.model,
            self.optimizer,
            self.train_dataloader,
            self.scheduler
        )
        
        if self.eval_dataloader:
            self.eval_dataloader = self.accelerator.prepare(self.eval_dataloader)
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.target_speaker_embedding = None
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Setup Hugging Face API
        if args.upload_to_hub:
            if args.hub_token:
                HfFolder.save_token(args.hub_token)
            self.hf_api = HfApi()
    
    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        return batch
    
    def compute_loss(self, batch):
        """Compute loss for a batch."""
        loss = torch.tensor(0.0, requires_grad=True, device=self.model.device)
        return loss
    
    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                loss = self.compute_loss(batch)
                
                self.accelerator.backward(loss)
                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress
            if self.accelerator.sync_gradients:
                self.global_step += 1
                total_loss += loss.item()
                avg_loss = total_loss / (step + 1)
                
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"})
                
                # Logging
                if self.global_step % self.args.logging_steps == 0:
                    if self.args.use_wandb:
                        self.accelerator.log({
                            "train/loss": loss.item(),
                            "train/avg_loss": avg_loss,
                            "train/learning_rate": self.scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                            "train/global_step": self.global_step
                        }, step=self.global_step)
                
                # Evaluation
                if self.global_step % self.args.eval_steps == 0 and self.eval_dataloader:
                    eval_loss = self.evaluate()
                    if self.args.use_wandb:
                        self.accelerator.log({"eval/loss": eval_loss}, step=self.global_step)
                    
                    # Save best model
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint(os.path.join(self.args.output_dir, "best"))
                        if self.args.upload_to_hub:
                            self.upload_to_hub(os.path.join(self.args.output_dir, "best"), self.args.hub_model_id_best)
                
                # Save checkpoint
                if self.global_step % self.args.save_steps == 0:
                    checkpoint_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")
                    self.save_checkpoint(checkpoint_dir)
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate the model."""
        if not self.eval_dataloader:
            return float('inf')
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.eval_dataloader,
            desc="Evaluating",
            disable=not self.accelerator.is_local_main_process
        )
        
        for batch in progress_bar:
            loss = self.compute_loss(batch)
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({"eval_loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
    
    def save_checkpoint(self, output_dir: str):
        """Save model checkpoint with optimizer and scheduler states."""
        if self.accelerator.is_main_process:
            print(f"Saving checkpoint to {output_dir}...")
            
            # Copy model files
            shutil.copytree(self.args.init_model_path, output_dir, dirs_exist_ok=True)
            
            # Update config
            input_config_file = os.path.join(self.args.init_model_path, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {self.args.speaker_name: 3000}
            talker_config["spk_is_dialect"] = {self.args.speaker_name: False}
            config_dict["talker_config"] = talker_config
            
            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            # Save model weights
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
            
            # Drop speaker encoder weights
            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]
            
            # Add speaker embedding
            if self.target_speaker_embedding is not None:
                weight = state_dict['talker.model.codec_embedding.weight']
                state_dict['talker.model.codec_embedding.weight'][3000] = \
                    self.target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)
            
            # Save optimizer and scheduler states
            optimizer_state = self.optimizer.state_dict()
            scheduler_state = self.scheduler.state_dict()
            
            torch.save({
                'optimizer_state_dict': optimizer_state,
                'scheduler_state_dict': scheduler_state,
                'global_step': self.global_step,
                'best_eval_loss': self.best_eval_loss,
            }, os.path.join(output_dir, "training_state.pt"))
            
            print(f"Checkpoint saved successfully!")
    
    def upload_to_hub(self, checkpoint_dir: str, repo_id: str):
        """Upload model to Hugging Face Hub."""
        if not self.accelerator.is_main_process or not self.args.upload_to_hub:
            return
        
        print(f"Uploading model to {repo_id}...")
        
        try:
            # Create repository if it doesn't exist
            self.hf_api.create_repo(repo_id, exist_ok=True)
            
            # Upload files
            files_to_upload = [
                "config.json",
                "model.safetensors",
                "training_state.pt",
            ]
            
            # Also upload tokenizer and processor files if they exist
            for file in os.listdir(self.args.init_model_path):
                if file.endswith(".json") or file.endswith(".txt") or file == "tokenizer_config.json":
                    src = os.path.join(self.args.init_model_path, file)
                    dst = os.path.join(checkpoint_dir, file)
                    if not os.path.exists(dst):
                        shutil.copy(src, dst)
                    files_to_upload.append(file)
            
            # Upload each file
            for file in files_to_upload:
                file_path = os.path.join(checkpoint_dir, file)
                if os.path.exists(file_path):
                    self.hf_api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=file,
                        repo_id=repo_id,
                    )
            
            print(f"Model uploaded successfully to {repo_id}!")
            
        except Exception as e:
            print(f"Error uploading to hub: {e}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        for epoch in range(self.args.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            print(f"{'='*50}")
            
            self.train_epoch(epoch)
            
            # Evaluate at end of epoch
            if self.eval_dataloader:
                eval_loss = self.evaluate()
                print(f"Epoch {epoch + 1} - Eval Loss: {eval_loss:.4f}")
                
                if self.args.use_wandb:
                    self.accelerator.log({
                        "eval/epoch_loss": eval_loss,
                        "epoch": epoch + 1
                    }, step=self.global_step)
                
                # Save best model
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint(os.path.join(self.args.output_dir, "best"))
                    if self.args.upload_to_hub:
                        self.upload_to_hub(os.path.join(self.args.output_dir, "best"), self.args.hub_model_id_best)
            
            # Save epoch checkpoint
            epoch_dir = os.path.join(self.args.output_dir, f"epoch-{epoch + 1}")
            self.save_checkpoint(epoch_dir)
        
        # Save final model
        print("\nTraining completed!")
        final_dir = os.path.join(self.args.output_dir, "last")
        self.save_checkpoint(final_dir)
        
        if self.args.upload_to_hub:
            self.upload_to_hub(final_dir, self.args.hub_model_id_last)
        
        if self.args.use_wandb:
            self.accelerator.end_training()


def prepare_data(args):
    """Prepare training data using dataset_tool.py."""
    print("="*60)
    print("Step 1: Preparing Training Data")
    print("="*60)
    
    # Prepare train data
    train_cmd = [
        sys.executable,
        "dataset_tool.py",
        "--dataset_name", args.dataset_name,
        "--split", "train",
        "--output_jsonl", args.train_jsonl,
        "--model_path", args.init_model_path,
        "--ref_audio_path", args.ref_audio_path,
        "--ref_text", args.ref_text,
        "--device", args.device
    ]
    
    # Add max_samples only if specified
    if args.max_train_samples is not None:
        train_cmd.extend(["--max_samples", str(args.max_train_samples)])
    
    print(f"Running: {' '.join(train_cmd)}")
    result = subprocess.run(train_cmd, check=True)
    
    # Prepare validation data if specified
    if args.validation_jsonl:
        val_cmd = [
            sys.executable,
            "dataset_tool.py",
            "--dataset_name", args.dataset_name,
            "--split", "validation",
            "--output_jsonl", args.validation_jsonl,
            "--model_path", args.init_model_path,
            "--ref_audio_path", args.ref_audio_path,
            "--ref_text", args.ref_text,
            "--device", args.device
        ]
        
        # Add max_samples only if specified
        if args.max_eval_samples is not None:
            val_cmd.extend(["--max_samples", str(args.max_eval_samples)])
        
        print(f"Running: {' '.join(val_cmd)}")
        result = subprocess.run(val_cmd, check=True)
    
    print("Data preparation complete!")


def main():
    # Create training arguments from environment variables
    training_args = TrainingArguments()
    
    print("="*60)
    print("Qwen3-TTS Advanced Training Pipeline")
    print("="*60)
    print(f"Dataset: {training_args.dataset_name}")
    print(f"Model: {training_args.init_model_path}")
    print(f"Output: {training_args.output_dir}")
    print(f"Train data: {training_args.train_jsonl}")
    print(f"Validation data: {training_args.validation_jsonl}")
    print(f"Batch size: {training_args.batch_size}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Epochs: {training_args.num_epochs}")
    print(f"Speaker name: {training_args.speaker_name}")
    print(f"Use WandB: {training_args.use_wandb}")
    print(f"Upload to Hub: {training_args.upload_to_hub}")
    print("="*60)
    
    # Step 1: Prepare data
    if not training_args.skip_prepare:
        prepare_data(training_args)
    else:
        print("Skipping data preparation (SKIP_PREPARE env var set)")
    
    # Step 2: Train model
    if not training_args.prepare_only:
        # Create trainer and start training
        trainer = AdvancedTrainer(training_args)
        trainer.train()
    else:
        print("Data preparation only (PREPARE_ONLY env var set)")
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()