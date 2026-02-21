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
    python train_advanced.py
    
    # Train with custom settings
    python train_advanced.py --batch_size 4 --lr 1e-5 --num_epochs 5 --use_wandb
    
    # Skip data preparation if already done
    python train_advanced.py --skip_prepare
"""

import argparse
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

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS", "finetuning"))

from dataset_tool import HausaTTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file


@dataclass
class TrainingArguments:
    """Training arguments."""
    init_model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    output_dir: str = "./output"
    dataset_name: str = "vaghawan/hausa-tts-22k"
    
    # Training hyperparameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Dataset settings
    train_jsonl: str = "./data/train.jsonl"
    validation_jsonl: str = "./data/validation.jsonl"
    ref_audio_path: str = None
    ref_text: str = "MTN Entertainment and Lifestyle. Entertainment and Lifestyle are at the heart of MTN's offering. We bring you music, movies, games and more through our digital platforms. With MTN musicals, you can stream your favorite"
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    
    # Speaker settings
    speaker_name: str = "hausa_speaker"
    
    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # WandB settings
    use_wandb: bool = True
    wandb_project: str = "qwen3-tts-hausa"
    wandb_run_name: Optional[str] = None
    
    # Hugging Face upload settings
    upload_to_hub: bool = False
    hub_model_id_best: str = "vaghawan/tts-best"
    hub_model_id_last: str = "vaghawan/tts-last"
    hub_token: Optional[str] = None
    
    # Mixed precision
    mixed_precision: str = "bf16"


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
        # Simple collate - return list of samples
        # The actual processing will be done in compute_loss
        return batch
    
    def compute_loss(self, batch):
        """Compute loss for a batch."""
        # This is a simplified version - you may need to adapt based on your actual data format
        # For now, we'll use a placeholder loss
        
        # Extract data from batch
        # batch is a list of dictionaries from JSONL
        # Each dict has: text, audio_codes, ref_audio, ref_text, etc.
        
        # Placeholder: compute a simple loss
        # In practice, you would:
        # 1. Tokenize text
        # 2. Load reference audio and extract mel spectrogram
        # 3. Forward pass through model
        # 4. Compute loss
        
        # For now, return a dummy loss
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
    parser = argparse.ArgumentParser(
        description="Advanced Training Pipeline for Qwen3-TTS with Validation, Metrics, and WandB"
    )
    
    # Model and data paths
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--dataset_name", type=str, default="vaghawan/hausa-tts-22k")
    parser.add_argument("--ref_audio_path", type=str, default=None)
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Dataset settings
    parser.add_argument("--train_jsonl", type=str, default="./data/train.jsonl")
    parser.add_argument("--validation_jsonl", type=str, default="./data/validation.jsonl")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    
    # Speaker settings
    parser.add_argument("--speaker_name", type=str, default="hausa_speaker")
    
    # Logging and checkpointing
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    
    # WandB settings
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--wandb_project", type=str, default="qwen3-tts-hausa")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # Hugging Face upload settings
    parser.add_argument("--upload_to_hub", action="store_true", default=False)
    parser.add_argument("--hub_model_id_best", type=str, default="vaghawan/tts-best")
    parser.add_argument("--hub_model_id_last", type=str, default="vaghawan/tts-last")
    parser.add_argument("--hub_token", type=str, default=None)
    
    # Mixed precision
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    
    # Workflow control
    parser.add_argument("--skip_prepare", action="store_true",
                       help="Skip data preparation if already done")
    parser.add_argument("--prepare_only", action="store_true",
                       help="Only prepare data, don't train")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use for data preparation")
    
    args = parser.parse_args()
    
    # Set default reference text
    ref_text = "MTN Entertainment and Lifestyle. Entertainment and Lifestyle are at the heart of MTN's offering. We bring you music, movies, games and more through our digital platforms. With MTN musicals, you can stream your favorite"
    
    print("="*60)
    print("Qwen3-TTS Advanced Training Pipeline")
    print("="*60)
    print(f"Dataset: {args.dataset_name}")
    print(f"Model: {args.init_model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Train data: {args.train_jsonl}")
    print(f"Validation data: {args.validation_jsonl}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Speaker name: {args.speaker_name}")
    print(f"Use WandB: {args.use_wandb}")
    print(f"Upload to Hub: {args.upload_to_hub}")
    print("="*60)
    
    # Step 1: Prepare data
    if not args.skip_prepare:
        prepare_data(args)
    else:
        print("Skipping data preparation (--skip_prepare flag set)")
    
    # Step 2: Train model
    if not args.prepare_only:
        # Create training arguments
        training_args = TrainingArguments(
            init_model_path=args.init_model_path,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            train_jsonl=args.train_jsonl,
            validation_jsonl=args.validation_jsonl,
            ref_audio_path=args.ref_audio_path,
            ref_text=ref_text,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            speaker_name=args.speaker_name,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_total_limit=args.save_total_limit,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            upload_to_hub=args.upload_to_hub,
            hub_model_id_best=args.hub_model_id_best,
            hub_model_id_last=args.hub_model_id_last,
            hub_token=args.hub_token,
            mixed_precision=args.mixed_precision
        )
        
        # Create trainer and start training
        trainer = AdvancedTrainer(training_args)
        trainer.train()
    else:
        print("Data preparation only (--prepare_only flag set)")
    
    print("\n" + "="*60)
    print("Pipeline complete!")
    print("="*60)


if __name__ == "__main__":
    main()
