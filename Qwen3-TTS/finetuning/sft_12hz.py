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
import argparse
import json
import os
import shutil
import sys

# Add Qwen3-TTS directory to Python path to find qwen_tts module
script_dir = os.path.dirname(os.path.abspath(__file__))
qwen3_tts_dir = os.path.dirname(script_dir)
if qwen3_tts_dir not in sys.path:
    sys.path.insert(0, qwen3_tts_dir)

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig
from layer_utils import replace_and_add_layers, print_model_summary

target_speaker_embedding = None
def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--replace_last_n_layers", type=int, default=2,
                        help="Number of last layers to replace with newly initialized layers")
    parser.add_argument("--add_new_layers", type=int, default=4,
                        help="Number of additional layers to add after replacement")
    parser.add_argument("--freeze_original_layers", type=bool, default=True,
                        help="Whether to freeze the original (non-replaced) layers")
    parser.add_argument("--freeze_speaker_encoder", type=bool, default=False,
                        help="Whether to freeze the speaker encoder (default: False for finetuning)")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=8, mixed_precision="bf16")

    MODEL_PATH = args.init_model_path

    # Try to load model with different attention implementations
    try:
        qwen3tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print(f"✓ Model loaded with flash_attention_2")
    except ImportError as e:
        print(f"⚠ Flash attention not available, falling back to SDPA")
        print(f"   (You can install flash-attn for potentially faster training)")
        qwen3tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print(f"✓ Model loaded with SDPA (Scaled Dot Product Attention)")
    except Exception as e:
        print(f"⚠ Error loading model with flash_attention_2: {e}")
        print(f"   Trying SDPA fallback...")
        qwen3tts = Qwen3TTSModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print(f"✓ Model loaded with SDPA")

    config = AutoConfig.from_pretrained(MODEL_PATH)

    # Freeze or unfreeze speaker encoder
    if args.freeze_speaker_encoder:
        print(f"\nFreezing speaker encoder...")
        for param in qwen3tts.model.speaker_encoder.parameters():
            param.requires_grad = False
        print(f"✓ Speaker encoder frozen")
    else:
        print(f"\nUnfreezing speaker encoder for finetuning...")
        for param in qwen3tts.model.speaker_encoder.parameters():
            param.requires_grad = True
        print(f"✓ Speaker encoder unfrozen (gradients will flow)")

    # Replace and add layers if requested
    if args.replace_last_n_layers > 0 or args.add_new_layers > 0:
        print(f"\nApplying layer replacement and addition...")
        qwen3tts.model = replace_and_add_layers(
            qwen3tts.model,
            replace_last_n=args.replace_last_n_layers,
            add_new_layers=args.add_new_layers,
            freeze_original_layers=args.freeze_original_layers,
            verbose=True
        )
        print_model_summary(qwen3tts.model)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

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

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype))
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding.detach()

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

            # Use the config object we loaded earlier instead of reading from MODEL_PATH
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

            # Verify the layer count in the saved configuration
            saved_num_layers = config_dict["talker_config"]["num_hidden_layers"]
            actual_num_layers = len(model.talker.model.layers)
            accelerator.print(f"Saving checkpoint with {saved_num_layers} layers (actual: {actual_num_layers})")
            if saved_num_layers != actual_num_layers:
                accelerator.print(f"⚠ Warning: Configuration layer count ({saved_num_layers}) doesn't match actual layers ({actual_num_layers})")
                # Fix the configuration
                config_dict["talker_config"]["num_hidden_layers"] = actual_num_layers

            output_config_file = os.path.join(output_dir, "config.json")
            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            # Save generation_config.json if available
            if hasattr(qwen3tts.model, 'generate_config') and qwen3tts.model.generate_config is not None:
                generation_config_file = os.path.join(output_dir, "generation_config.json")
                with open(generation_config_file, 'w', encoding='utf-8') as f:
                    json.dump(qwen3tts.model.generate_config, f, indent=2, ensure_ascii=False)
                accelerator.print(f"✓ Saved generation_config.json")

            # Save processor config from the loaded model (includes tokenizer files)
            if qwen3tts.processor is not None:
                qwen3tts.processor.save_pretrained(output_dir)
                accelerator.print(f"✓ Saved processor and tokenizer files")

            # Save speech tokenizer separately
            if hasattr(qwen3tts.model, 'speech_tokenizer') and qwen3tts.model.speech_tokenizer is not None:
                speech_tokenizer_dir = os.path.join(output_dir, "speech_tokenizer")
                os.makedirs(speech_tokenizer_dir, exist_ok=True)
                qwen3tts.model.speech_tokenizer.save_pretrained(speech_tokenizer_dir)
                accelerator.print(f"✓ Saved speech_tokenizer to {speech_tokenizer_dir}")

            # Save speaker encoder for the new speaker (optional, for reference)
            if hasattr(qwen3tts.model, 'speaker_encoder') and qwen3tts.model.speaker_encoder is not None:
                speaker_encoder_dir = os.path.join(output_dir, "speaker_encoder")
                os.makedirs(speaker_encoder_dir, exist_ok=True)
                # Save speaker encoder config
                speaker_encoder_config = {
                    "model_type": "qwen3_tts_speaker_encoder",
                    "speaker_name": args.speaker_name,
                    "speaker_embedding_dim": target_speaker_embedding.shape[-1] if target_speaker_embedding is not None else 1024
                }
                speaker_encoder_config_file = os.path.join(speaker_encoder_dir, "speaker_config.json")
                with open(speaker_encoder_config_file, 'w', encoding='utf-8') as f:
                    json.dump(speaker_encoder_config, f, indent=2, ensure_ascii=False)
                accelerator.print(f"✓ Saved speaker encoder config for speaker: {args.speaker_name}")

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            # Count the number of layer weights being saved
            layer_keys = [k for k in state_dict.keys() if 'talker.model.layers' in k]
            unique_layers = set()
            for key in layer_keys:
                # Extract layer index from key (e.g., "talker.model.layers.0.self_attn.q_proj.weight" -> 0)
                parts = key.split('.')
                if 'layers' in parts:
                    layer_idx = parts[parts.index('layers') + 1]
                    unique_layers.add(layer_idx)
            accelerator.print(f"Saving {len(unique_layers)} layers in checkpoint")

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)
            accelerator.print(f"✓ Saved model.safetensors")

            # List all saved files
            accelerator.print(f"\n{'='*60}")
            accelerator.print(f"Checkpoint contents:")
            accelerator.print(f"{'='*60}")
            saved_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, output_dir)
                    saved_files.append(rel_path)
                    accelerator.print(f"  - {rel_path}")
            accelerator.print(f"{'='*60}")
            accelerator.print(f"Total files saved: {len(saved_files)}")
            accelerator.print(f"{'='*60}\n")

            # Create a README file in the checkpoint directory
            readme_content = f"""# Fine-Tuned Qwen3-TTS Model Checkpoint

## Model Information
- **Speaker Name**: {args.speaker_name}
- **Base Model**: {MODEL_PATH}
- **Number of Layers**: {config_dict.get('talker_config', {}).get('num_hidden_layers', 'N/A')}
- **Hidden Size**: {config_dict.get('talker_config', {}).get('hidden_size', 'N/A')}
- **Training Epoch**: {epoch}

## Files Included
- `config.json` - Model configuration
- `generation_config.json` - Generation parameters
- `model.safetensors` - Model weights
- `tokenizer_config.json` - Tokenizer configuration
- `vocab.json` - Vocabulary file
- `merges.txt` - BPE merges file
- `preprocessor_config.json` - Preprocessor configuration
- `speech_tokenizer/` - Speech tokenizer model and config
- `speaker_encoder/speaker_config.json` - Speaker encoder configuration

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
The model has been fine-tuned for speaker: **{args.speaker_name}**
Speaker embedding is stored at index 3000 in the codec embedding layer.

## Notes
- This model uses the Qwen3-TTS-12Hz tokenizer
- The model supports streaming generation
- For best results, use reference audio from the same speaker used during training
"""
            readme_file = os.path.join(output_dir, "README.md")
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            accelerator.print(f"✓ Created README.md with usage instructions")

            accelerator.print(f"✓ Checkpoint saved to {output_dir}")

if __name__ == "__main__":
    train()
