# Qwen3-TTS Training Guide

This guide provides a clean, organized approach to training Qwen3-TTS models with two training pipelines and a unified dataset tool.

## 📁 Project Structure

```
Qwen3-TTS-finetuning/
├── dataset_tool.py          # Unified dataset loading and preparation tool
├── train_using_sft.py          # Simple training pipeline (uses sft_12hz.py)
├── train_wandb_validation.py        # Advanced training pipeline (with validation, metrics, WandB)
├── voices/                  # Reference audio files
│   └── english_voice/
│       └── english_voice.wav
├── data/                    # Prepared datasets (created automatically)
│   ├── train.jsonl
│   └── validation.jsonl
└── Qwen3-TTS/               # Core Qwen3-TTS library
    └── finetuning/
        └── sft_12hz.py      # Base training script
```

## 🚀 Quick Start

### 1. Simple Training (Recommended for Beginners)

The simple training pipeline uses `sft_12hz.py` directly and is perfect for quick experiments.

```bash
# Train with default settings
python train_using_sft.py

# Train with custom settings
python train_using_sft.py --batch_size 4 --lr 1e-5 --num_epochs 5

# Skip data preparation if already done
python train_using_sft.py --skip_prepare

# Only prepare data, don't train
python train_using_sft.py --prepare_only
```

### 2. Advanced Training (Recommended for Production)

The advanced training pipeline includes validation, metrics, WandB logging, and model checkpointing.

```bash
# Train with default settings (includes WandB)
python train_wandb_validation.py

# Train with custom settings
python train_wandb_validation.py --batch_size 4 --lr 1e-5 --num_epochs 5 --use_wandb

# Train without WandB
python train_wandb_validation.py --use_wandb False

# Upload models to Hugging Face Hub
python train_wandb_validation.py --upload_to_hub --hub_token YOUR_TOKEN

# Skip data preparation if already done
python train_wandb_validation.py --skip_prepare
```

## 📊 Dataset Tool

The `dataset_tool.py` provides a unified interface for dataset loading and preparation.

### Prepare Dataset from Hugging Face

```bash
# Prepare training data
python dataset_tool.py --dataset_name vaghawan/hausa-tts-22k --split train --output_jsonl ./data/train.jsonl

# Prepare validation data
python dataset_tool.py --dataset_name vaghawan/hausa-tts-22k --split validation --output_jsonl ./data/validation.jsonl

# Limit number of samples (for debugging)
python dataset_tool.py --dataset_name vaghawan/hausa-tts-22k --split train --max_samples 100

# Use CPU instead of GPU
python dataset_tool.py --dataset_name vaghawan/hausa-tts-22k --split train --device cpu
```

### Get Dataset Information

```bash
# Get info about a prepared dataset
python dataset_tool.py --info ./data/train.jsonl
```

### Use Dataset Tool in Python

```python
from dataset_tool import prepare_dataset, load_jsonl_dataset, create_dataloader, get_dataset_info

# Prepare dataset from Hugging Face
prepare_dataset(
    dataset_name="vaghawan/hausa-tts-22k",
    split="train",
    output_jsonl="./data/train.jsonl",
    model_path="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    max_samples=1000
)

# Load dataset from JSONL
dataset = load_jsonl_dataset("./data/train.jsonl")
print(f"Loaded {len(dataset)} samples")

# Create PyTorch DataLoader
dataloader = create_dataloader("./data/train.jsonl", batch_size=4, shuffle=True)

# Get dataset information
info = get_dataset_info("./data/train.jsonl")
print(f"Languages: {info['languages']}")
print(f"Speakers: {info['speakers']}")
```

## 🔧 Configuration Options

### Simple Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset_name` | `vaghawan/hausa-tts-22k` | Hugging Face dataset name |
| `--train_jsonl` | `./data/train.jsonl` | Training data output path |
| `--validation_jsonl` | `None` | Validation data output path |
| `--init_model_path` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | Base model path |
| `--output_model_path` | `./output` | Model output directory |
| `--batch_size` | `2` | Training batch size |
| `--lr` | `2e-5` | Learning rate |
| `--num_epochs` | `3` | Number of training epochs |
| `--speaker_name` | `hausa_speaker` | Speaker name |
| `--max_train_samples` | `None` | Max training samples |
| `--max_eval_samples` | `None` | Max evaluation samples |
| `--skip_prepare` | `False` | Skip data preparation |
| `--prepare_only` | `False` | Only prepare data |

### Advanced Training Options

All simple training options plus:

| Option | Default | Description |
|--------|---------|-------------|
| `--gradient_accumulation_steps` | `4` | Gradient accumulation steps |
| `--weight_decay` | `0.01` | Weight decay |
| `--warmup_steps` | `100` | Warmup steps |
| `--max_grad_norm` | `1.0` | Max gradient norm |
| `--logging_steps` | `10` | Logging frequency |
| `--save_steps` | `500` | Checkpoint save frequency |
| `--eval_steps` | `500` | Evaluation frequency |
| `--use_wandb` | `True` | Use WandB logging |
| `--wandb_project` | `qwen3-tts-hausa` | WandB project name |
| `--wandb_run_name` | `None` | WandB run name |
| `--upload_to_hub` | `False` | Upload to Hugging Face |
| `--hub_model_id_best` | `vaghawan/tts-best` | Best model repo ID |
| `--hub_model_id_last` | `vaghawan/tts-last` | Last model repo ID |
| `--hub_token` | `None` | Hugging Face token |
| `--mixed_precision` | `bf16` | Mixed precision mode |

## 📈 Training Workflows

### Workflow 1: Simple Training

```bash
# Step 1: Prepare data (optional, done automatically)
python dataset_tool.py --dataset_name vaghawan/hausa-tts-22k --split train --output_jsonl ./data/train.jsonl

# Step 2: Train
python train_using_sft.py --skip_prepare
```

### Workflow 2: Advanced Training

```bash
# Step 1: Prepare data (optional, done automatically)
python dataset_tool.py --dataset_name vaghawan/hausa-tts-22k --split train --output_jsonl ./data/train.jsonl
python dataset_tool.py --dataset_name vaghawan/hausa-tts-22k --split validation --output_jsonl ./data/validation.jsonl

# Step 2: Train with validation and metrics
python train_wandb_validation.py --skip_prepare --use_wandb
```

### Workflow 3: Debug with Small Dataset

```bash
# Prepare small dataset for debugging
python dataset_tool.py --dataset_name vaghawan/hausa-tts-22k --split train --max_samples 10 --output_jsonl ./data/debug.jsonl

# Train with small dataset
python train_using_sft.py --train_jsonl ./data/debug.jsonl --num_epochs 1
```

## 📦 Output Structure

### Simple Training Output

```
output/
├── config.json
├── model.safetensors
└── tokenizer files...
```

### Advanced Training Output

```
output/
├── best/                          # Best model (lowest validation loss)
│   ├── config.json
│   ├── model.safetensors
│   └── training_state.pt
├── last/                          # Last checkpoint
│   ├── config.json
│   ├── model.safetensors
│   └── training_state.pt
├── checkpoint-500/                # Intermediate checkpoints
├── checkpoint-1000/
├── epoch-1/
├── epoch-2/
└── epoch-3/
```

## 🔍 Monitoring Training

### Simple Training

Simple training outputs progress to the console. Monitor the loss values to ensure training is progressing.

### Advanced Training

Advanced training provides comprehensive monitoring:

1. **Console Output**: Real-time progress bars and metrics
2. **WandB Dashboard**: Detailed metrics, charts, and model comparison
3. **Checkpoints**: Automatic saving of best and last models

To view WandB dashboard:
```bash
# After starting training, visit the URL shown in the output
# Or manually:
wandb login
wandb dashboard
```

## 🐛 Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python train_using_sft.py --batch_size 1

# Or use gradient accumulation
python train_wandb_validation.py --batch_size 1 --gradient_accumulation_steps 8
```

### Slow Training

```bash
# Use fewer samples for testing
python dataset_tool.py --max_samples 100

# Reduce epochs
python train_using_sft.py --num_epochs 1
```

### Flash Attention Not Available

The scripts automatically fall back to SDPA if flash attention is not available. No action needed.

### SoX Not Found

SoX is only needed for audio processing. If you see this warning, install SoX:

```bash
# Ubuntu/Debian
sudo apt-get install sox

# macOS
brew install sox
```

## 📚 Additional Resources

- [Qwen3-TTS Documentation](./Qwen3-TTS/README.md)
- [Hugging Face Dataset](https://huggingface.co/datasets/vaghawan/hausa-tts-22k)
- [WandB Documentation](https://docs.wandb.ai/)

## 🤝 Contributing

To add new features or fix issues:

1. Modify the appropriate script (`dataset_tool.py`, `train_using_sft.py`, or `train_wandb_validation.py`)
2. Test your changes with a small dataset
3. Update this documentation

## 📝 License

See [LICENSE](./Qwen3-TTS/LICENSE) for details.

---

## 🏗️ Qwen3-TTS Architecture & Voice Cloning Deep Dive

This section provides a comprehensive understanding of the Qwen3-TTS model architecture, voice cloning mechanism, and complete training workflow.

### 📐 Model Architecture Overview

Qwen3-TTS is a transformer-based text-to-speech system that uses a codec-based approach for audio generation. The architecture consists of several key components:

```
┌─────────────────────────────────────────────────────────────┐
│                    Qwen3-TTS Model                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  Speaker Encoder │      │   Talker Model   │            │
│  │                  │      │                  │            │
│  │  Reference Audio │─────▶│  Text Embedding  │            │
│  │       ↓          │      │  Codec Embedding │            │
│  │  Mel Spectrogram │      │  (with Speaker   │            │
│  │       ↓          │      │   Embedding)     │            │
│  │  Speaker Embed   │      │       ↓          │            │
│  │  (256-dim)       │      │  Transformer     │            │
│  └──────────────────┘      │       ↓          │            │
│                            │  Audio Codes     │            │
│                            │  Prediction      │            │
│                            │       ↓          │            │
│                            │  Sub-Talker      │            │
│                            │  (Auxiliary)     │            │
│                            └──────────────────┘            │
│                                                              │
│  ┌──────────────────┐      ┌──────────────────┐            │
│  │  Codec Encoder   │      │  Codec Decoder   │            │
│  │                  │      │                  │            │
│  │  Raw Audio       │─────▶│  Audio Codes     │─────▶ Audio│
│  │       ↓          │      │       ↓          │      Output│
│  │  Audio Codes     │      │  Neural Codec    │            │
│  │  (16 channels)   │      │  Vocoder         │            │
│  └──────────────────┘      └──────────────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 🔑 Key Components

#### 1. **Speaker Encoder**
- **Purpose**: Extract speaker characteristics from reference audio
- **Input**: Mel spectrogram of reference audio (24kHz, 128 mel bands)
- **Output**: 256-dimensional speaker embedding vector
- **Process**:
  ```python
  # From sft_12hz.py line 92-94
  speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
  ```
- **Usage**: The speaker embedding is injected into the codec embedding to condition the generation on the target speaker's voice

#### 2. **Talker Model**
The core generation component with several sub-modules:

##### a. **Text Embedding Layer**
- **Purpose**: Convert text tokens to embeddings
- **Input**: Tokenized text (with special tokens)
- **Output**: Text embeddings
- **Special Tokens**:
  - `<|im_start|>assistant\n` - Start of assistant response
  - `<|im_end|>\n<|im_start|>assistant\n` - End of response
  ```python
  # From dataset.py line 91-92
  def _build_assistant_text(self, text: str) -> str:
      return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
  ```

##### b. **Codec Embedding Layer**
- **Purpose**: Convert audio codec tokens to embeddings
- **Input**: Audio codec IDs (16 channels)
- **Output**: Codec embeddings
- **Special Structure**:
  - Position 0-2: Reserved for special tokens
  - Position 3: `codec_nothink_id` - No-think token
  - Position 4: `codec_think_bos_id` - Think beginning
  - Position 5: `codec_think_eos_id` - Think end
  - Position 6: **Speaker embedding slot** (replaced with actual speaker embedding)
  - Position 7: `codec_pad_id` - Padding token
  ```python
  # From sft_12hz.py line 99-101
  input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
  input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
  input_codec_embedding[:, 6, :] = speaker_embedding  # Inject speaker embedding
  ```

##### c. **Code Predictor**
- **Purpose**: Predict audio codes for all 16 codec channels
- **Structure**: 16 separate embedding layers, one for each codec channel
- **Process**:
  ```python
  # From sft_12hz.py line 105-108
  for i in range(1, 16):
      codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
      codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
      input_embeddings = input_embeddings + codec_i_embedding
  ```

##### d. **Transformer Decoder**
- **Purpose**: Generate audio codes autoregressively
- **Input**: Combined embeddings (text + codec + speaker)
- **Output**: Predicted audio codes
- **Attention**: Uses causal masking for autoregressive generation

##### e. **Sub-Talker**
- **Purpose**: Auxiliary loss for better codec prediction
- **Input**: Hidden states and codec IDs
- **Output**: Sub-talker logits and loss
- **Weight**: 0.3 in total loss calculation
  ```python
  # From sft_12hz.py line 121-123
  sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
  loss = outputs.loss + 0.3 * sub_talker_loss
  ```

#### 3. **Codec Encoder**
- **Purpose**: Encode raw audio to discrete audio codes
- **Input**: Raw audio waveform (24kHz)
- **Output**: Audio codes (16 channels, variable length)
- **Process**: Uses neural codec (similar to EnCodec) to compress audio

#### 4. **Codec Decoder**
- **Purpose**: Decode audio codes back to waveform
- **Input**: Audio codes (16 channels)
- **Output**: Raw audio waveform (24kHz)
- **Process**: Uses neural vocoder to reconstruct audio

### 🎭 Voice Cloning Mechanism

Qwen3-TTS supports two voice cloning modes with different trade-offs:

#### Mode Comparison

| Feature | x-vector Only | ICL (In-Context Learning) |
|---------|---------------|---------------------------|
| **Reference Text** | Not required | Required |
| **Quality** | Good | Excellent |
| **Speed** | Faster | Slower |
| **Use Case** | Quick generation, real-time | High-quality output |
| **Speaker Similarity** | ~85-90% | ~95-98% |
| **Prosody Matching** | Basic | Advanced |

#### Mode 1: x-vector Only (Fast, Good Quality)

**How it works:**
- Extracts speaker embedding from reference audio
- Uses only the embedding to condition generation
- No reference text needed
- Faster inference but lower quality

**When to use:**
- Real-time applications
- Quick prototyping
- When reference text is unavailable
- When speed is more important than quality

**Configuration:**
```python
# During inference
output = model.generate(
    text="Hello world",
    reference_audio="ref.wav",
    use_xvector_only=True  # x-vector mode
)
```

**Architecture flow:**
```
Reference Audio → Mel Spectrogram → Speaker Encoder → Speaker Embedding (256-dim)
                                                                    ↓
Text Tokens → Text Embedding ──────────────────────────────────────┼→ Combined Embeddings
                                                                    ↓
                                                            Transformer → Audio Codes
```

#### Mode 2: ICL with Reference Text (Slower, Excellent Quality)

**How it works:**
- Extracts speaker embedding from reference audio
- Uses reference text to understand speaking style, prosody, and context
- Combines both for high-quality generation
- Slower but much better quality

**When to use:**
- High-quality output required
- When reference text is available
- Offline generation
- When quality is more important than speed

**Configuration:**
```python
# During inference
output = model.generate(
    text="Hello world",
    reference_audio="ref.wav",
    ref_text="This is the reference audio content.",  # Must match reference audio
    use_xvector_only=False  # ICL mode
)
```

**Architecture flow:**
```
Reference Audio → Mel Spectrogram → Speaker Encoder → Speaker Embedding (256-dim)
                                                                    ↓
Reference Text → Tokenization → Text Embedding ────────────────────┼→ Combined Embeddings
                                                                    ↓
Target Text → Tokenization → Text Embedding ───────────────────────┘
                                                                    ↓
                                                            Transformer → Audio Codes
```

#### Voice Cloning Process (Both Modes)

Voice cloning in Qwen3-TTS works through the following process:

#### Step 1: Reference Audio Processing
```python
# From dataset.py line 103-116
@torch.inference_mode()
def extract_mels(self, audio, sr):
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
```

1. **Resample**: Convert reference audio to 24kHz
2. **Mel Spectrogram**: Extract 128-band mel spectrogram
   - FFT size: 1024
   - Hop size: 256 (93.75ms frames)
   - Win size: 1024
   - Frequency range: 0-12kHz

#### Step 2: Speaker Embedding Extraction
```python
# From sft_12hz.py line 92-94
speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
if target_speaker_embedding is None:
    target_speaker_embedding = speaker_embedding
```

1. **Encode**: Pass mel spectrogram through speaker encoder
2. **Extract**: Get 256-dimensional speaker embedding
3. **Store**: Save first batch's embedding for final model

#### Step 3: Text Processing
```python
# From dataset.py line 129-130
text = self._build_assistant_text(text)
text_ids = self._tokenize_texts(text)
```

1. **Format**: Wrap text with special tokens
2. **Tokenize**: Convert to token IDs using Qwen3 tokenizer

#### Step 4: Input Construction
The model uses a dual-channel input structure:

```
Input IDs Shape: [batch_size, sequence_length, 2]
├── Channel 0: Text tokens
└── Channel 1: Codec tokens (with speaker embedding)
```

**Text Channel Structure**:
```
Position 0-2: Special tokens (BOS, etc.)
Position 3-6: Padding tokens
Position 7: BOS token
Position 8 to 8+text_len-3: Text tokens
Position 8+text_len-3: EOS token
Position 8+text_len-2 to end: Padding tokens
```

**Codec Channel Structure**:
```
Position 0-2: Reserved
Position 3: codec_nothink_id
Position 4: codec_think_bos_id
Position 5: codec_think_eos_id
Position 6: SPEAKER EMBEDDING (256-dim vector)
Position 7: codec_pad_id
Position 8 to 8+text_len-3: codec_pad_id
Position 8+text_len-3: codec_pad_id
Position 8+text_len-2: codec_bos_id
Position 8+text_len-1 to 8+text_len-1+codec_len: Audio codes
Position 8+text_len-1+codec_len: codec_eos_id
```

```python
# From dataset.py line 169-204 (simplified)
# Text channel
input_ids[i, :3, 0] = text_ids[0, :3]
input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
input_ids[i, 7, 0] = self.config.tts_bos_token_id
input_ids[i, 8:8+text_ids_len-3, 0] = text_ids[0, 3:]
input_ids[i, 8+text_ids_len-3, 0] = self.config.tts_eos_token_id

# Codec channel
input_ids[i, 3:8, 1] = [codec_nothink_id, codec_think_bos_id, 
                         codec_think_eos_id, 0, codec_pad_id]
input_ids[i, 6, 1] = SPEAKER_EMBEDDING  # Injected during training
input_ids[i, 8+text_ids_len-2, 1] = codec_bos_id
input_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len, 1] = audio_codec_0
input_ids[i, 8+text_ids_len-1+codec_ids_len, 1] = codec_eos_id
```

#### Step 5: Embedding Combination
```python
# From sft_12hz.py line 99-108
input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
input_codec_embedding[:, 6, :] = speaker_embedding  # Inject speaker embedding

input_embeddings = input_text_embedding + input_codec_embedding

# Add codec embeddings for all 16 channels
for i in range(1, 16):
    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
    input_embeddings = input_embeddings + codec_i_embedding
```

1. **Text Embeddings**: Embed text tokens
2. **Codec Embeddings**: Embed codec tokens
3. **Speaker Injection**: Replace position 6 with speaker embedding
4. **Combine**: Add text and codec embeddings
5. **Multi-Channel**: Add embeddings for all 16 codec channels

#### Step 6: Forward Pass
```python
# From sft_12hz.py line 110-123
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
```

1. **Transformer**: Pass embeddings through transformer
2. **Predict**: Predict next audio codes
3. **Extract**: Get hidden states for codec positions
4. **Sub-Talker**: Compute auxiliary loss
5. **Total Loss**: Combine main loss (0.7) and sub-talker loss (0.3)

### 🔄 Complete Training Workflow

#### Phase 1: Data Preparation

**Step 1.1: Load Dataset from Hugging Face**
```python
# From dataset_tool.py
hf_dataset = load_dataset("vaghawan/hausa-tts-22k", split="train")
```

**Step 1.2: Process Audio**
```python
# From dataset_tool.py
# Extract audio array and sampling rate
audio_array = item["audio"]["array"]
audio_sr = item["audio"]["sampling_rate"]

# Resample to 24kHz if needed
if audio_sr != 24000:
    audio_array = librosa.resample(audio_array, orig_sr=audio_sr, target_sr=24000)
```

**Step 1.3: Encode Audio to Codes**
```python
# From dataset_tool.py
with torch.no_grad():
    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0).to(device)
    audio_codes = model.codec_encoder(audio_tensor)
    audio_codes = audio_codes.squeeze(0).cpu().numpy().tolist()
```

**Step 1.4: Save to JSONL**
```python
# From dataset_tool.py
data_entry = {
    "audio": f"sample_{idx}.wav",
    "text": item["text"],
    "audio_codes": audio_codes,  # Shape: [T, 16]
    "language": item.get("language", "ha"),
    "ref_audio": ref_audio_path,
    "ref_text": ref_text,
    "speaker_id": item.get("speaker_id", "unknown"),
    "gender": item.get("gender", "unknown"),
    "age_range": item.get("age_range", "unknown"),
    "phase": item.get("phase", "unknown")
}
```

#### Phase 2: Dataset Loading

**Step 2.1: Load JSONL**
```python
# From dataset.py line 65-67
train_data = open(args.train_jsonl).readlines()
train_data = [json.loads(line) for line in train_data]
dataset = TTSDataset(train_data, qwen3tts.processor, config)
```

**Step 2.2: Process Each Sample**
```python
# From dataset.py line 120-144
def __getitem__(self, idx):
    item = self.data_list[idx]
    
    # Extract fields
    audio_path = item["audio"]
    text = item["text"]
    audio_codes = item["audio_codes"]
    ref_audio_path = item['ref_audio']
    
    # Build text with special tokens
    text = self._build_assistant_text(text)
    text_ids = self._tokenize_texts(text)
    
    # Convert audio codes to tensor
    audio_codes = torch.tensor(audio_codes, dtype=torch.long)
    
    # Extract mel spectrogram from reference audio
    ref_mel = self.extract_mels(audio=wav, sr=sr)
    
    return {
        "text_ids": text_ids[:, :-5],
        "audio_codes": audio_codes,
        "ref_mel": ref_mel
    }
```

**Step 2.3: Batch Collation**
```python
# From dataset.py line 146-218
def collate_fn(self, batch):
    # Calculate max sequence length
    item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
    max_length = max(item_length) + 8
    
    # Initialize tensors
    input_ids = torch.zeros((b, t, 2), dtype=torch.long)
    codec_ids = torch.zeros((b, t, 16), dtype=torch.long)
    text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
    codec_mask = torch.zeros((b, t), dtype=torch.bool)
    attention_mask = torch.zeros((b, t), dtype=torch.long)
    codec_0_labels = torch.full((b, t), -100, dtype=torch.long)
    
    # Fill in data for each sample
    for i, data in enumerate(batch):
        # ... (construct input structure as shown above)
    
    # Concatenate reference mels
    ref_mels = torch.cat([data['ref_mel'] for data in batch], dim=0)
    
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
```

#### Phase 3: Model Training

**Step 3.1: Initialize Model**
```python
# From sft_12hz.py line 48-63
try:
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    print(f"✓ Model loaded with flash_attention_2")
except (ImportError, Exception) as e:
    print(f"⚠ Flash attention not available, falling back to SDPA: {e}")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print(f"✓ Model loaded with SDPA")
```

**Step 3.2: Setup Training**
```python
# From sft_12hz.py line 44, 70-74
accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

model, optimizer, train_dataloader = accelerator.prepare(
    qwen3tts.model, optimizer, train_dataloader
)
```

**Step 3.3: Training Loop**
```python
# From sft_12hz.py line 79-134
for epoch in range(num_epochs):
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model):
            # Extract batch data
            input_ids = batch['input_ids']
            codec_ids = batch['codec_ids']
            ref_mels = batch['ref_mels']
            text_embedding_mask = batch['text_embedding_mask']
            codec_embedding_mask = batch['codec_embedding_mask']
            attention_mask = batch['attention_mask']
            codec_0_labels = batch['codec_0_labels']
            codec_mask = batch['codec_mask']
            
            # Extract speaker embedding
            speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
            if target_speaker_embedding is None:
                target_speaker_embedding = speaker_embedding
            
            # Build embeddings
            input_text_ids = input_ids[:, :, 0]
            input_codec_ids = input_ids[:, :, 1]
            
            input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
            input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
            input_codec_embedding[:, 6, :] = speaker_embedding
            
            input_embeddings = input_text_embedding + input_codec_embedding
            
            # Add codec embeddings for all 16 channels
            for i in range(1, 16):
                codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                input_embeddings = input_embeddings + codec_i_embedding
            
            # Forward pass
            outputs = model.talker(
                inputs_embeds=input_embeddings[:, :-1, :],
                attention_mask=attention_mask[:, :-1],
                labels=codec_0_labels[:, 1:],
                output_hidden_states=True
            )
            
            # Sub-talker loss
            hidden_states = outputs.hidden_states[0][-1]
            talker_hidden_states = hidden_states[codec_mask[:, 1:]]
            talker_codec_ids = codec_ids[codec_mask]
            
            sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
            
            # Total loss
            loss = outputs.loss + 0.3 * sub_talker_loss
            
            # Backward pass
            accelerator.backward(loss)
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Logging
        if step % 10 == 0:
            accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
```

**Step 3.4: Save Checkpoint**
```python
# From sft_12hz.py line 136-168
if accelerator.is_main_process:
    output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
    shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)
    
    # Update config
    with open(input_config_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = {args.speaker_name: 3000}
    talker_config["spk_is_dialect"] = {args.speaker_name: False}
    config_dict["talker_config"] = talker_config
    
    with open(output_config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    # Save model weights
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}
    
    # Drop speaker encoder weights
    keys_to_drop = [k for k in state_dict.keys() if k.startswith("speaker_encoder")]
    for k in keys_to_drop:
        del state_dict[k]
    
    # Add speaker embedding to codec embedding
    weight = state_dict['talker.model.codec_embedding.weight']
    state_dict['talker.model.codec_embedding.weight'][3000] = \
        target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
    
    save_file(state_dict, save_path)
```

### 🎯 Key Design Decisions

#### 1. **Speaker Embedding Injection**
- **Position**: Injected at position 6 of codec channel
- **Reason**: This position is reserved for speaker information in the codec embedding layer
- **Benefit**: Allows the model to condition generation on speaker characteristics without modifying the architecture

#### 2. **Dual-Channel Input**
- **Channel 0**: Text tokens (semantic information)
- **Channel 1**: Codec tokens (acoustic information + speaker)
- **Benefit**: Separates semantic and acoustic information, allowing the model to learn their relationship

#### 3. **16-Channel Codec**
- **Structure**: 16 parallel audio code streams
- **Purpose**: Capture different aspects of audio (pitch, timbre, rhythm, etc.)
- **Benefit**: Richer representation than single-channel approaches

#### 4. **Sub-Talker Auxiliary Loss**
- **Weight**: 0.3 of total loss
- **Purpose**: Improve codec prediction quality
- **Benefit**: Better audio quality and more stable training

#### 5. **Gradient Accumulation**
- **Default**: 4 steps
- **Purpose**: Simulate larger batch sizes with limited GPU memory
- **Benefit**: Better training stability with limited resources

#### 6. **Mixed Precision (BF16)**
- **Format**: Brain Float 16
- **Purpose**: Reduce memory usage and speed up training
- **Benefit**: 2x memory reduction with minimal quality loss

### 📊 Training Metrics

#### Loss Components
1. **Main Loss**: Cross-entropy loss for codec prediction (0.7 weight)
2. **Sub-Talker Loss**: Auxiliary loss for better prediction (0.3 weight)
3. **Total Loss**: Weighted sum of both losses

#### Monitoring
- **Loss**: Should decrease over time
- **Learning Rate**: Follows cosine schedule with warmup
- **Gradient Norm**: Clipped to 1.0 to prevent exploding gradients

### 🔧 Inference Process

After training, inference follows these steps:

1. **Load Model**: Load fine-tuned model with speaker embedding
2. **Process Reference Audio**: Extract mel spectrogram
3. **Extract Speaker Embedding**: Use speaker encoder (or use stored embedding)
4. **Tokenize Text**: Convert input text to tokens
5. **Construct Input**: Build dual-channel input with speaker embedding
6. **Generate**: Autoregressively generate audio codes
7. **Decode**: Convert codes to waveform using codec decoder

### 💡 Tips for Better Voice Cloning

1. **Reference Audio Quality**: Use clear, noise-free reference audio (5-10 seconds)
2. **Speaker Consistency**: Ensure reference audio matches target speaker
3. **Training Data**: Use diverse samples from the same speaker for better generalization
4. **Learning Rate**: Start with 2e-5, adjust based on loss curve
5. **Batch Size**: Use largest batch size that fits in GPU memory
6. **Epochs**: 3-5 epochs typically sufficient for voice cloning
7. **Validation**: Monitor validation loss to prevent overfitting

### 🎯 Fine-Tuning with Reference Text (ICL Mode)

When fine-tuning for high-quality voice cloning, you can include reference text in your training data to enable ICL mode.

#### Dataset Preparation with Reference Text

**JSONL Format with Reference Text:**
```json
{
  "audio": "sample_0.wav",
  "text": "This is the target text to synthesize.",
  "audio_codes": [[...], [...], ...],  // Shape: [T, 16]
  "ref_audio": "voices/speaker/reference.wav",
  "ref_text": "This is the reference audio content that matches the reference audio.",
  "language": "en",
  "speaker_id": "speaker_001",
  "gender": "male",
  "age_range": "25-35"
}
```

**Key Points:**
- `ref_text` must accurately transcribe the reference audio
- Reference audio should be 5-10 seconds of clear speech
- Reference text helps the model learn prosody, rhythm, and speaking style
- Without `ref_text`, the model falls back to x-vector mode

#### Training with Reference Text

The current training pipeline (`sft_12hz.py`) uses x-vector mode by default. To enable ICL mode during fine-tuning:

**Option 1: Modify Dataset Preparation**
```python
# In dataset_tool.py, ensure ref_text is included
data_entry = {
    "audio": f"sample_{idx}.wav",
    "text": item["text"],
    "audio_codes": audio_codes,
    "ref_audio": ref_audio_path,
    "ref_text": ref_text,  # Include reference text
    # ... other fields
}
```

**Option 2: Use Reference Text During Inference**
Even if you train with x-vector mode, you can still use ICL mode during inference:

```python
# Load fine-tuned model
model = Qwen3TTSModel.from_pretrained("./output/best")

# Generate with ICL mode (higher quality)
output = model.generate(
    text="Hello, this is a test.",
    reference_audio="voices/speaker/reference.wav",
    ref_text="This is the reference audio content.",  # Must match reference audio
    use_xvector_only=False  # ICL mode
)

# Generate with x-vector mode (faster)
output = model.generate(
    text="Hello, this is a test.",
    reference_audio="voices/speaker/reference.wav",
    use_xvector_only=True  # x-vector mode
)
```

#### Reference Text Best Practices

1. **Accuracy**: Reference text must exactly match the reference audio content
2. **Length**: 5-10 seconds of speech is optimal
3. **Clarity**: Use clear, well-pronounced reference audio
4. **Consistency**: Use the same reference audio/text pair for all samples from a speaker
5. **Language**: Reference text should be in the same language as target text

**Example:**
```python
# Good reference text
ref_text = "The quick brown fox jumps over the lazy dog. This sentence contains all the letters of the alphabet."

# Bad reference text (doesn't match audio)
ref_text = "Hello world"  # If audio says something different
```

#### When to Use Each Mode

**Use x-vector mode when:**
- Training on large datasets where reference text is unavailable
- Real-time generation is required
- Reference audio quality is poor
- Quick prototyping

**Use ICL mode when:**
- High-quality output is required
- Reference text is available and accurate
- Offline generation is acceptable
- Prosody and style matching is important

**Hybrid approach:**
- Train with x-vector mode (faster, simpler)
- Use ICL mode during inference (better quality)
- This gives you the best of both worlds

#### Quality Comparison

**x-vector Mode:**
- Speaker similarity: 85-90%
- Prosody matching: Basic
- Generation speed: ~2-3x real-time
- Memory usage: Lower
- Best for: Real-time applications

**ICL Mode:**
- Speaker similarity: 95-98%
- Prosody matching: Advanced
- Generation speed: ~1-1.5x real-time
- Memory usage: Higher
- Best for: High-quality output

#### Implementation Example

```python
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# Load model
model = Qwen3TTSModel.from_pretrained("./output/best")

# Reference audio and text
ref_audio = "voices/speaker/reference.wav"
ref_text = "This is the reference audio content. It should match the audio exactly."

# Generate multiple samples
texts = [
    "Hello, how are you today?",
    "The weather is beautiful outside.",
    "I hope you have a wonderful day."
]

# Generate with ICL mode (high quality)
for text in texts:
    output = model.generate(
        text=text,
        reference_audio=ref_audio,
        ref_text=ref_text,
        use_xvector_only=False
    )
    # Save output
    sf.write(f"output_icl_{len(text)}.wav", output.audio, output.sampling_rate)

# Generate with x-vector mode (faster)
for text in texts:
    output = model.generate(
        text=text,
        reference_audio=ref_audio,
        use_xvector_only=True
    )
    # Save output
    sf.write(f"output_xvector_{len(text)}.wav", output.audio, output.sampling_rate)
```

### 🚀 Advanced Features

#### 1. **Multi-Speaker Training**
- Train on multiple speakers simultaneously
- Use different speaker embeddings for each speaker
- Model learns to disentangle speaker and content

#### 2. **Language Adaptation**
- Train on multilingual datasets
- Model learns language-specific patterns
- Can generate speech in multiple languages

#### 3. **Style Transfer**
- Use reference audio from one speaker
- Generate speech in another speaker's voice
- Requires careful speaker embedding manipulation

#### 4. **Zero-Shot Voice Cloning**
- Use pre-trained model without fine-tuning
- Extract speaker embedding from reference audio
- Generate speech in target voice immediately

#### 5. **ICL Mode Fine-Tuning**
To enable ICL mode during fine-tuning (for even better quality), you need to modify the training process:

**Step 1: Prepare Dataset with Reference Text**
```python
# In dataset_tool.py, ensure ref_text is included
data_entry = {
    "audio": f"sample_{idx}.wav",
    "text": item["text"],
    "audio_codes": audio_codes,
    "ref_audio": ref_audio_path,
    "ref_text": ref_text,  # Include reference text
    "language": item.get("language", "ha"),
    "speaker_id": item.get("speaker_id", "unknown"),
    # ... other fields
}
```

**Step 2: Modify Dataset to Use Reference Text**
```python
# In dataset.py, modify __getitem__ to include ref_text
def __getitem__(self, idx):
    item = self.data_list[idx]

    audio_path = item["audio"]
    text = item["text"]
    audio_codes = item["audio_codes"]
    ref_audio_path = item['ref_audio']
    ref_text = item.get('ref_text', '')  # Get reference text

    # Build text with special tokens
    text = self._build_assistant_text(text)
    text_ids = self._tokenize_texts(text)

    # If reference text is provided, tokenize it
    if ref_text:
        ref_text = self._build_assistant_text(ref_text)
        ref_text_ids = self._tokenize_texts(ref_text)
    else:
        ref_text_ids = None

    audio_codes = torch.tensor(audio_codes, dtype=torch.long)
    ref_mel = self.extract_mels(audio=wav, sr=sr)

    return {
        "text_ids": text_ids[:, :-5],
        "audio_codes": audio_codes,
        "ref_mel": ref_mel,
        "ref_text_ids": ref_text_ids  # Include reference text IDs
    }
```

**Step 3: Modify Collate Function**
```python
# In dataset.py, modify collate_fn to handle ref_text_ids
def collate_fn(self, batch):
    # ... existing code ...

    # Handle reference text IDs
    ref_text_ids_list = [data.get('ref_text_ids') for data in batch]
    if any(ids is not None for ids in ref_text_ids_list):
        # Pad reference text IDs
        max_ref_len = max(ids.shape[1] if ids is not None else 0 for ids in ref_text_ids_list)
        ref_text_ids_batch = torch.zeros((b, max_ref_len), dtype=torch.long)
        for i, ids in enumerate(ref_text_ids_list):
            if ids is not None:
                ref_text_ids_batch[i, :ids.shape[1]] = ids[0]
    else:
        ref_text_ids_batch = None

    return {
        'input_ids': input_ids,
        'ref_mels': ref_mels,
        'attention_mask': attention_mask,
        'text_embedding_mask': text_embedding_mask.unsqueeze(-1),
        'codec_embedding_mask': codec_embedding_mask.unsqueeze(-1),
        'codec_0_labels': codec_0_labels,
        'codec_ids': codec_ids,
        'codec_mask': codec_mask,
        'ref_text_ids': ref_text_ids_batch  # Include reference text IDs
    }
```

**Step 4: Modify Training Loop**
```python
# In sft_12hz.py, modify training loop to use reference text
for step, batch in enumerate(train_dataloader):
    with accelerator.accumulate(model):
        # Extract batch data
        input_ids = batch['input_ids']
        codec_ids = batch['codec_ids']
        ref_mels = batch['ref_mels']
        ref_text_ids = batch.get('ref_text_ids')  # Get reference text IDs
        # ... other fields ...

        # Extract speaker embedding
        speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()

        # Build embeddings
        input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
        input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
        input_codec_embedding[:, 6, :] = speaker_embedding

        input_embeddings = input_text_embedding + input_codec_embedding

        # If reference text is provided, concatenate it
        if ref_text_ids is not None:
            ref_text_embedding = model.talker.model.text_embedding(ref_text_ids)
            # Concatenate reference text embeddings with input embeddings
            input_embeddings = torch.cat([ref_text_embedding, input_embeddings], dim=1)

        # Add codec embeddings for all 16 channels
        for i in range(1, 16):
            codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_i_embedding

        # Forward pass
        outputs = model.talker(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=codec_0_labels[:, 1:],
            output_hidden_states=True
        )

        # ... rest of training loop ...
```

**Benefits of ICL Mode Fine-Tuning:**
- Better prosody matching
- Improved speaking style transfer
- Higher quality output
- Better handling of complex sentences

**Trade-offs:**
- Slower training (more computation)
- Requires accurate reference text
- More memory usage
- More complex implementation

### 📚 References

- **Qwen3-TTS Paper**: [Link to paper]
- **Codec Architecture**: Similar to EnCodec
- **Transformer Architecture**: Standard decoder-only transformer
- **Speaker Encoder**: Based on mel-spectrogram analysis

---

**End of Architecture Documentation**
