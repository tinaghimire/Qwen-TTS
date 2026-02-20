# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Training script for Hausa TTS fine-tuning with Qwen3-TTS.
Features:
- Trainer class with training and evaluation loops
- WandB logging
- Model upload to Hugging Face (best and last models)
- Checkpoint saving with optimizer and scheduler states
"""
import argparse
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from huggingface_hub import HfApi, HfFolder
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_cosine_schedule_with_warmup
from tqdm import tqdm

from hausa_dataset import HausaTTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file


@dataclass
class TrainingArguments:
    """Training arguments."""
    init_model_path: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    output_dir: str = "./output"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"
    
    # Training hyperparameters
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Dataset settings
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
    upload_to_hub: bool = True
    hub_model_id_best: str = "vaghawan/tts-best"
    hub_model_id_last: str = "vaghawan/tts-last"
    hub_token: Optional[str] = None
    
    # Mixed precision
    mixed_precision: str = "bf16"


class HausaTTSTrainer:
    """Trainer class for Hausa TTS fine-tuning."""
    
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
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
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
            print(f"✓ Model loaded with flash_attention_2")
        except (ImportError, Exception) as e:
            print(f"⚠ Flash attention not available, falling back to SDPA: {e}")
            self.qwen3tts = Qwen3TTSModel.from_pretrained(
                args.init_model_path,
                device_map="cuda",
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            print(f"✓ Model loaded with SDPA")
        self.config = AutoConfig.from_pretrained(args.init_model_path)
        
        # Create datasets
        print("Creating datasets...")
        self.train_dataset = HausaTTSDataset(
            split=args.train_split,
            processor=self.qwen3tts.processor,
            config=self.config,
            ref_audio_path=args.ref_audio_path,
            ref_text=args.ref_text,
            max_samples=args.max_train_samples
        )
        
        self.eval_dataset = HausaTTSDataset(
            split=args.validation_split,
            processor=self.qwen3tts.processor,
            config=self.config,
            ref_audio_path=args.ref_audio_path,
            ref_text=args.ref_text,
            max_samples=args.max_eval_samples
        )
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn
        )
        
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=self.eval_dataset.collate_fn
        )
        
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
    
    def compute_loss(self, batch):
        """Compute loss for a batch."""
        input_ids = batch['input_ids']
        codec_ids = batch['codec_ids']
        ref_mels = batch['ref_mels']
        text_embedding_mask = batch['text_embedding_mask']
        codec_embedding_mask = batch['codec_embedding_mask']
        attention_mask = batch['attention_mask']
        codec_0_labels = batch['codec_0_labels']
        codec_mask = batch['codec_mask']
        
        # Get speaker embedding
        speaker_embedding = self.model.speaker_encoder(ref_mels.to(self.model.device).to(self.model.dtype)).detach()
        if self.target_speaker_embedding is None:
            self.target_speaker_embedding = speaker_embedding
        
        # Prepare input embeddings
        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]
        
        input_text_embedding = self.model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
        input_codec_embedding = self.model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
        input_codec_embedding[:, 6, :] = speaker_embedding
        
        input_embeddings = input_text_embedding + input_codec_embedding
        
        # Add codec embeddings
        for i in range(1, 16):
            codec_i_embedding = self.model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_i_embedding
        
        # Forward pass
        outputs = self.model.talker(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=codec_0_labels[:, 1:],
            output_hidden_states=True
        )
        
        # Get sub-talker loss
        hidden_states = outputs.hidden_states[0][-1]
        talker_hidden_states = hidden_states[codec_mask[:, 1:]]
        talker_codec_ids = codec_ids[codec_mask]
        
        sub_talker_logits, sub_talker_loss = self.model.talker.forward_sub_talker_finetune(
            talker_codec_ids, talker_hidden_states
        )
        
        # Total loss
        loss = outputs.loss + 0.3 * sub_talker_loss
        
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
                if self.global_step % self.args.eval_steps == 0:
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
        
        avg_loss = total_loss / num_batches
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
            eval_loss = self.evaluate()
            print(f"Epoch {epoch + 1} - Eval Loss: {eval_loss:.4f}")
            
            if self.args.use_wandb:
                self.accelerator.log({
                    "eval/epoch_loss": eval_loss,
                    "epoch": epoch + 1
                }, step=self.global_step)
            
            # Save epoch checkpoint
            epoch_dir = os.path.join(self.args.output_dir, f"epoch-{epoch + 1}")
            self.save_checkpoint(epoch_dir)
            
            # Save best model
            if eval_loss < self.best_eval_loss:
                self.best_eval_loss = eval_loss
                self.save_checkpoint(os.path.join(self.args.output_dir, "best"))
                if self.args.upload_to_hub:
                    self.upload_to_hub(os.path.join(self.args.output_dir, "best"), self.args.hub_model_id_best)
        
        # Save final model
        print("\nTraining completed!")
        final_dir = os.path.join(self.args.output_dir, "last")
        self.save_checkpoint(final_dir)
        
        if self.args.upload_to_hub:
            self.upload_to_hub(final_dir, self.args.hub_model_id_last)
        
        if self.args.use_wandb:
            self.accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-TTS on Hausa data")
    
    # Model and data paths
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_dir", type=str, default="./output")
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
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--validation_split", type=str, default="validation")
    parser.add_argument("--test_split", type=str, default="test")
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
    parser.add_argument("--upload_to_hub", action="store_true", default=True)
    parser.add_argument("--hub_model_id_best", type=str, default="vaghawan/tts-best")
    parser.add_argument("--hub_model_id_last", type=str, default="vaghawan/tts-last")
    parser.add_argument("--hub_token", type=str, default=None)
    
    # Mixed precision
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    
    args = parser.parse_args()
    
    # Create training arguments
    training_args = TrainingArguments(**vars(args))
    
    # Create trainer and start training
    trainer = HausaTTSTrainer(training_args)
    trainer.train()


if __name__ == "__main__":
    main()
