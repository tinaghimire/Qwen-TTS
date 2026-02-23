#!/usr/bin/env python3
# coding=utf-8
"""
Training script for Qwen3 TTS that loads data directly from HuggingFace.
No need to create intermediate JSONL files.
"""
import argparse
import json
import os
import sys

# Add Qwen3-TTS directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
qwen_tts_dir = os.path.join(script_dir, "Qwen3-TTS")
if qwen_tts_dir not in sys.path:
    sys.path.insert(0, qwen_tts_dir)

import torch
from accelerate import Accelerator
from datasets import load_dataset
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoConfig

# Import necessary components from existing dataset module
sys.path.insert(0, os.path.join(qwen_tts_dir, "finetuning"))
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
import librosa
import numpy as np


class HFStreamingDataset(TorchDataset):
    """
    Dataset that loads data directly from HuggingFace.
    Handles audio code extraction on-the-fly.
    """
    def __init__(self, hf_dataset, processor, config, speech_tokenizer=None, max_samples=None):
        super().__init__()
        self.dataset = hf_dataset
        self.processor = processor
        self.config = config
        self.speech_tokenizer = speech_tokenizer
        
        # Optionally limit the number of samples
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
    
    def __len__(self):
        return len(self.dataset)
    
    def _load_audio(self, audio_data):
        """Load audio from HuggingFace format (can be dict with array or file path)"""
        if isinstance(audio_data, dict) and 'array' in audio_data:
            # HF format with numpy array
            audio = audio_data['array'].astype(np.float32)
            sr = audio_data['sampling_rate']
        elif isinstance(audio_data, str):
            # File path
            audio, sr = librosa.load(audio_data, sr=24000, mono=True)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=-1)
            audio = audio.astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio format: {type(audio_data)}")
        
        return audio, sr
    
    def _extract_audio_codes(self, audio, sr):
        """Extract audio codes using speech tokenizer if available"""
        if self.speech_tokenizer is not None:
            # Use the speech tokenizer to extract codes
            with torch.no_grad():
                codes = self.speech_tokenizer.encode(
                    torch.from_numpy(audio).float().unsqueeze(0)
                )
                return codes.cpu().numpy().tolist()
        else:
            # Fallback: generate placeholder codes or load from dataset if available
            # This is a simplified version - in production you'd need the speech tokenizer
            return [[1, 2, 3]]  # Placeholder - you'll need to adjust this
    
    def _extract_mel(self, audio, sr):
        """Extract mel spectrogram from reference audio"""
        assert sr == 24000, "Only support 24kHz audio"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000
        ).transpose(1, 2)
        return mels
    
    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    def _tokenize_texts(self, text):
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id
    
    @torch.inference_mode()
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get text
        text = item['text']
        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)
        
        # Load audio
        audio_data = item['audio']
        audio, sr = self._load_audio(audio_data)
        
        # Extract audio codes
        # Check if codes are already in the dataset (e.g., from preprocessing)
        if 'audio_codes' in item and item['audio_codes'] is not None:
            audio_codes = item['audio_codes']
        else:
            audio_codes = self._extract_audio_codes(audio, sr)
        
        audio_codes = torch.tensor(audio_codes, dtype=torch.long)
        
        # Load reference audio for speaker embedding
        ref_audio_path = item.get('ref_audio', None)
        if ref_audio_path is None:
            # Use the same audio as reference
            ref_audio = audio
            ref_sr = sr
        else:
            ref_audio, ref_sr = self._load_audio(ref_audio_path)
        
        ref_mel = self._extract_mel(ref_audio, ref_sr)
        
        return {
            "text_ids": text_ids[:, :-5],
            "audio_codes": audio_codes,
            "ref_mel": ref_mel
        }
    
    def collate_fn(self, batch):
        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
        max_length = max(item_length) + 8
        b, t = len(batch), max_length
        
        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), -100, dtype=torch.long)
        
        for i, data in enumerate(batch):
            text_ids = data['text_ids']
            audio_codec_0 = data['audio_codes'][:, 0]
            audio_codecs = data['audio_codes']
            
            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]
            
            # text channel
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8+text_ids_len-3, 0] = text_ids[0, 3:]
            input_ids[i, 8+text_ids_len-3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8+text_ids_len-2:8+text_ids_len+codec_ids_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, :8+text_ids_len+codec_ids_len] = True
            
            # codec channel
            input_ids[i, 3:8, 1] = torch.tensor([
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                0,  # for speaker embedding
                self.config.talker_config.codec_pad_id
            ])
            input_ids[i, 8:8+text_ids_len-3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8+text_ids_len-3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8+text_ids_len-2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8+text_ids_len-1+codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id
            
            codec_0_labels[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id
            
            codec_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len, :] = audio_codecs
            
            codec_embedding_mask[i, 3:8+text_ids_len+codec_ids_len] = True
            codec_embedding_mask[i, 6] = False  # for speaker embedding
            
            codec_mask[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = True
            attention_mask[i, :8+text_ids_len+codec_ids_len] = True
        
        ref_mels = [data['ref_mel'] for data in batch]
        ref_mels = torch.cat(ref_mels, dim=0)
        
        return {
            'input_ids': input_ids,
            'ref_mels': ref_mels,
            'attention_mask': attention_mask,
            'text_embedding_mask': text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask': codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels': codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask': codec_mask
        }


def train():
    parser = argparse.ArgumentParser(description="Train Qwen3 TTS directly from HuggingFace dataset")
    parser.add_argument("--hf_dataset", type=str, help="HuggingFace dataset name (e.g., vaghawan/hausa-tts-22k)")
    parser.add_argument("--train_jsonl", type=str, help="Path to JSONL file (alternative to HF dataset)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--ref_audio", type=str, required=True, help="Reference audio path for speaker embedding")
    parser.add_argument("--output_model_path", type=str, default="./output")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of training samples")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps")
    args = parser.parse_args()
    
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation, mixed_precision="bf16")
    
    MODEL_PATH = args.init_model_path
    
    # Load model
    try:
        qwen3tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print("✓ Model loaded with flash_attention_2")
    except ImportError as e:
        print(f"⚠ Flash attention not available, falling back to SDPA")
        qwen3tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print("✓ Model loaded with SDPA (Scaled Dot Product Attention)")
    except Exception as e:
        print(f"⚠ Error loading model with flash_attention_2: {e}")
        print("   Trying SDPA fallback...")
        qwen3tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print("✓ Model loaded with SDPA")
    
    config = AutoConfig.from_pretrained(MODEL_PATH)
    
    # Load dataset
    if args.hf_dataset:
        print(f"Loading dataset from HuggingFace: {args.hf_dataset}")
        hf_dataset = load_dataset(args.hf_dataset, split=args.split)
        print(f"✓ Loaded {len(hf_dataset)} samples")
        
        # Add reference audio to all samples
        hf_dataset = hf_dataset.map(lambda x: {"ref_audio": args.ref_audio})
        
        dataset = HFStreamingDataset(
            hf_dataset,
            qwen3tts.processor,
            config,
            max_samples=args.max_samples
        )
    elif args.train_jsonl:
        print(f"Loading dataset from JSONL: {args.train_jsonl}")
        train_data = []
        with open(args.train_jsonl) as f:
            for line in f:
                data = json.loads(line)
                train_data.append(data)
        
        # Import the original TTSDataset
        from dataset import TTSDataset
        dataset = TTSDataset(train_data, qwen3tts.processor, config)
        
        if args.max_samples:
            dataset.data_list = dataset.data_list[:args.max_samples]
            print(f"Limited to {args.max_samples} samples")
    else:
        raise ValueError("Must specify either --hf_dataset or --train_jsonl")
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)
    
    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )
    
    num_epochs = args.num_epochs
    model.train()
    
    target_speaker_embedding = None
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Effective batch size (with gradient accumulation): {args.batch_size * args.gradient_accumulation}")
    print(f"Learning rate: {args.lr}")
    
    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']
                
                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding
                
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
                
                loss = outputs.loss + 0.3 * sub_talker_loss
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()
            
            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
        
        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            os.makedirs(output_dir, exist_ok=True)
            
            config_dict = config.to_dict()
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            config_dict["talker_config"] = talker_config
            
            output_config_file = os.path.join(output_dir, "config.json")
            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            if qwen3tts.processor is not None:
                qwen3tts.processor.save_pretrained(output_dir)
            
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
            
            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]
            
            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)
            
            accelerator.print(f"✓ Checkpoint saved to {output_dir}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    train()