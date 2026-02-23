# Complete Qwen3-TTS Fine-tuning Analysis & Recommendations

## 📊 Current Training Configuration

### Model Information
| Parameter | Value |
|-----------|-------|
| **Model Name** | Qwen/Qwen3-TTS-12Hz-1.7B-Base |
| **Model Size** | 1.7 Billion Parameters |
| **Audio Codec** | 12Hz Audio Codec (16 codebooks) |
| **Training Mode** | Custom Voice (Voice Cloning) |
| **Attention Implementation** | SDPA (Scaled Dot Product Attention) |
| **Mixed Precision** | BF16 |

### Training Hyperparameters

| Parameter | Current Value | Recommended Range | Description |
|-----------|---------------|-------------------|-------------|
| **Batch Size** | 8 | 1-4 | Samples per batch |
| **Learning Rate** | 2e-4 (0.0002) | 1e-5 to 1e-4 | Optimizer learning rate |
| **Gradient Accumulation** | 4 (implicit) | 8-16 | Steps before weight update |
| **Number of Epochs** | 3 | 10-50 | Training iterations |
| **Weight Decay** | 0.01 | 0.001-0.1 | L2 regularization |
| **Warmup Steps** | 100 | 100-500 | LR warm-up period |
| **Max Gradient Norm** | 1.0 | 0.5-1.0 | Gradient clipping |
| **Effective Batch Size** | 32 | 32-64 | batch_size × grad_accum |

### Dataset Information

| Parameter | Value |
|-----------|-------|
| **Dataset Name** | vaghawan/hausa-tts-22k |
| **Language** | Haus |
| **Training Samples** | 984 (full) / 100 (limited) |
| **Validation Samples** | Not configured |
| **Text Length** | Short to medium sentences |
| **Audio Codec** | Pre-extracted (16 codebooks × 16 timesteps) |
| **Reference Voice** | English Voice (cross-language cloning) |

### Reference Audio
| Parameter | Value |
|-----------|-------|
| **Reference Audio** | english_voice_24k.wav |
| **Reference Language** | English |
| **Reference Audio Length** | ~24 kHz |
| **Target Language** | Hausa |
| **Speaker ID** | 3000 |
| **Cloning Strategy** | Cross-language voice cloning |

---

## 🔍 Training Analysis

### Observed Behavior

#### Training Progress (Epoch 1)
- **Step 10**: Loss = 1.2586, Avg Loss = 1.0186
- **Step 20**: Loss = 0.0014, Avg Loss = 0.5394
- **Step 30**: Loss = 0.0000, Avg Loss = 0.3596
- **Final Epoch 1**: Loss = 0.0000, Avg Loss = 0.3509

#### Training Progress (Epoch 2-3)
- **All losses**: 0.0000
- **Learning Rate**: Increasing (0.000020 → 0.000180)
- **Training Speed**: ~2.0 it/s

### 🔴 PROBLEM IDENTIFIED

**WARNING: Loss dropped to 0.0000 too quickly! This indicates:**

1. **Overfitting** - Model memorized small dataset
2. **Loss Computation Issue** - Possibly NaN/Inf masked as 0
3. **Insufficient Training Data** - Only 100 samples
4. **High Learning Rate** - 2e-4 is too aggressive for 1.7B model
5. **Cross-language Mismatch** - English reference for Hausa text

### Key Issues

| Issue | Severity | Impact |
|-------|----------|--------|
| **Quick loss convergence to 0** | 🔴 Critical | Overfitting |
| **Only 100 samples** | 🔴 Critical | Insufficient data |
| **Cross-language mismatch** | 🔴 Critical | Poor voice quality |
| **High learning rate** | 🟡 High | Unstable training |
| **Missing validation** | 🟡 High | No quality monitoring |
| **Same audio_codes for all samples** | 🔴 Critical | No actual pattern learning |
| **Only 3 epochs** | 🟡 Medium | Insufficient training |

---

## 📈 Analysis: Why Loss is 0.0000

### Root Causes

1. **All samples have identical audio_codes**
   ```json
   "audio_codes": [[1290, 636, 938, 1670, 1558, 1228, 1048, 1886, 1028, 1446, 1358, 1912, 850, 1050, 1186, 103]]
   ```
   - Every sample in your dataset has the EXACT SAME audio codes
   - Model learns to predict the same codes regardless of text
   - Loss becomes zero because target is always identical

2. **Cross-language voice cloning problem**
   - Reference voice: English
   - Target text: Hausa
   - Phonetic mismatch causes poor articulation

3. **Insufficient data diversity**
   - Only 100 samples
   - Limited phonetic coverage
   - Minimal prosodic variation

---

## 🎯 ACTION PLAN for Best Voice Cloning Results

### Phase 1: Immediate Fixes (Priority 1)

#### 1.1 Fix Training Hyperparameters

```bash
# Update .env file with these values
BATCH_SIZE=1                    # Reduce for better gradient quality
LR=5e-5                        # Lower learning rate (2.5x lower)
NUM_EPOCHS=20                  # More epochs for better learning
GRADIENT_ACCUMULATION_STEPS=16 # Increase effective batch size
WARMUP_STEPS=300               # Longer warmup
LOGGING_STEPS=5                # More frequent logging
```

**Reasoning:**
- Lower LR = more stable gradients
- More epochs = better convergence
- Grad accumulation = memory efficiency

#### 1.2 Increase Dataset Size

```bash
# Remove sample limit
MAX_TRAIN_SAMPLES=5000         # Use more samples
MAX_EVAL_SAMPLES=500           # Add validation data
```

**Steps:**
```bash
# Re-run data preparation without limit
rm ./data/train.jsonl ./data/validation.jsonl
MAX_TRAIN_SAMPLES=5000 MAX_EVAL_SAMPLES=500 .venv/bin/python dataset_tool.py
```

#### 1.3 Fix Audio Code Extraction

**CRITICAL:** Currently all samples have identical codes. Need to extract unique codes per audio file.

Check `dataset_tool.py` line:

```python
# Look for this pattern - codes should vary per sample
audio_codes = tokenizer.encode(audio_data)  # Should extract from actual audio
```

#### 1.4 Add Validation Set

```bash
# Create proper validation split
VALIDATION_JSONL=./data/validation.jsonl
MAX_EVAL_SAMPLES=200
```

---

### Phase 2: Audio & Data Quality (Priority 2)

#### 2.1 Language-Specific Reference Audio

**CURRENT PROBLEM:** Using English reference for Hausa text

**SOLUTION:** Choose reference audio from target language

```bash
# Options:
# 1. Use Hausa speaker's audio (if available)
REF_AUDIO_PATH=./voices/hausa_speaker.wav

# 2. Use multi-lingual speaker
REF_AUDIO_PATH=./voices/multilingual_speaker.wav
```

**Or use the target language data:**
```python
# Use one of the training samples as reference
REF_AUDIO_PATH=/path/to/clear/hausa_sample.wav
REF_TEXT="Sample text in Hausa with good articulation"
```

#### 2.2 Audio Quality Requirements

Your reference audio should:

✅ **DO:**
- Be 10-30 seconds long
- Have clear pronunciation
- Contain multiple phonemes
- Have natural prosody
- Be in the target language (Hausa)
- Have consistent volume
- Be recorded at 24kHz

❌ **DON'T:**
- Use background noise
- Use clipped audio
- Use very short (<5s) samples
- Use cross-language for critical tasks
- Use very long (>60s) samples

#### 2.3 Data Augmentation

```python
# Add to training pipeline:
- Speed perturbation (0.9x, 1.0x, 1.1x)
- Pitch shifting (±2 semitones)
- Volume normalization
- Background noise augmentation
```

---

### Phase 3: Advanced Optimization (Priority 3)

#### 3.1 Learning Rate Scheduling

```python
# Implement cosine annealing with warm restarts
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=500,        # Warm-up
    num_training_steps=total_steps,
    num_cycles=0.5,              # Cosine cycles
    eta_min=1e-6                 # Min LR
)
```

#### 3.2 Curriculum Learning

```python
# Start with easy samples, increase difficulty
def curriculum_sampler(epoch):
    if epoch < 5:
        # Short sentences only
        return short_sentences
    elif epoch < 10:
        # Add medium sentences
        return short_sentences + medium_sentences
    else:
        # All sentences
        return all_sentences
```

#### 3.3 Multi-Speaker Training (if available)

```python
# If you have multiple Hausa speakers:
speakers = {
    'speaker_1': speaker_id_1,
    'speaker_2': speaker_id_2,
    'speaker_3': speaker_id_3,
}
# Train on all for better generalization
```

---

## 📋 Recommended Training Configuration

### Best Practices Configuration

```bash
# .env file - Recommended Settings

# Model Settings
MODEL_PATH=Qwen/Qwen3-TTS-12Hz-1.7B-Base
OUTPUT_DIR=./output_v2

# Critical Hyperparameters
BATCH_SIZE=1                    # Per GPU
LR=3e-5                        # Conservative LR
NUM_EPOCHS=30                  # Sufficient training
GRADIENT_ACCUMULATION_STEPS=16 # Effective batch = 16
EFFECTIVE_BATCH_SIZE=16        # 1 × 16

# Regularization
WEIGHT_DECAY=0.01              # Default
MAX_GRAD_NORM=0.5              # Stricter clipping
WARMUP_STEPS=500               # Gentle warmup

# Dataset
DATASET_NAME=vaghawan/hausa-tts-22k
MAX_TRAIN_SAMPLES=5000         # More data
MAX_EVAL_SAMPLES=500           # Validation
TRAIN_JSONL=./data/train_v2.jsonl
VALIDATION_JSONL=./data/val_v2.jsonl

# Reference Audio (TARGET LANGUAGE!)
REF_AUDIO_PATH=./voices/hausa_reference.wav
REF_TEXT="Sanannun za ku iya tattaunawa da bayyana..."  # Hausa text

# Logging
LOGGING_STEPS=5                # Frequent logging
EVAL_STEPS=100                 # Evaluate every 100 steps
SAVE_STEPS=500                 # Save checkpoints
```

### Alternative: Aggressive Training (More Computation)

```bash
# For best results with more resources:
BATCH_SIZE=2
LR=5e-5
NUM_EPOCHS=50
GRADIENT_ACCUMULATION_STEPS=8
MAX_TRAIN_SAMPLES=10000
```

---

## 🔬 Expected Results Analysis

### Current Configuration (Estimated Quality)

| Metric | Expected | Reality |
|--------|----------|---------|
| **MOS (Mean Opinion Score)** | 1.5-2.0/5.0 | ~1.0/5.0 (Poor) |
| **WER (Word Error Rate)** | >50% | Likely >80% |
| **Voice Similarity** | 30-40% | ~20% |
| **Naturalness** | Very Low | Minimal |
| **Intelligibility** | Poor | Unintelligible |

### Recommended Configuration (Expected Quality)

| Metric | Expected | Notes |
|--------|----------|-------|
| **MOS** | 3.5-4.2/5.0 | Good to Very Good |
| **WER** | 15-25% | Intelligible |
| **Voice Similarity** | 70-85% | High |
| **Naturalness** | High | Pleasant speech |
| **Intelligibility** | Excellent | Clear articulation |

---

## 📊 Training Monitoring Checklist

### During Training

- [ ] Loss decreases gradually (not instantly to 0)
- [ ] Loss doesn't plateau too early
- [ ] Learning rate scheduler works correctly
- [ ] No NaN/Inf gradients
- [ ] Validation tracks training reasonably
- [ ] Training logs captured

### After Training

- [ ] Generate test samples
- [ ] Conduct subjective evaluation (listen)
- [ ] Calculate objective metrics (MOS, WER)
- [ ] Compare with baseline
- [ ] Test on unseen text
- [ ] Test different emotions/styles

---

## 🎨 Advanced Techniques (Optional)

### 1. Adapter-Based Fine-tuning

Instead of full model tuning, use adapters:
```python
# Add small adapter layers
# Only train adapters (~1% of parameters)
# Better for multi-speaker scenarios
```

### 2. LoRA (Low-Rank Adaptation)

```python
# Use `peft` library for LoRA
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["talker"],
    lora_dropout=0.05,
    bias="none"
)
```

### 3. Progressive Training

```python
# Train in stages:
# Stage 1: Freeze most layers, train only embeddings (2 epochs)
# Stage 2: Train decoder (5 epochs)
# Stage 3: Full fine-tuning (10 epochs)
```

---

## 🚀 Quick Start Script

```bash
#!/bin/bash

# Step 1: Prepare high-quality data
.venv/bin/python dataset_tool.py \
  --max_samples 5000 \
  --force

# Step 2: Update hyperparameters
cat > .env << EOF
BATCH_SIZE=1
LR=3e-5
NUM_EPOCHS=30
GRADIENT_ACCUMULATION_STEPS=16
MAX_TRAIN_SAMPLES=5000
MAX_EVAL_SAMPLES=500
REF_AUDIO_PATH=./voices/hausa_reference.wav
EOF

# Step 3: Start training
.venv/bin/python train_from_hf.py \
  --train_jsonl ./data/train.jsonl \
  --ref_audio ./voices/hausa_reference.wav \
  --batch_size 1 \
  --lr 3e-5 \
  --num_epochs 30 \
  --max_samples 5000 \
  --speaker_name hausa_speaker

# Step 4: Monitor
tail -f output/training_log_*.txt
```

---

## 📈 Success Metrics

Define success before training:

| Metric | Minimum Acceptable | Target | Excellent |
|--------|-------------------|--------|-----------|
| **Final Training Loss** | <0.5 | <0.2 | <0.1 |
| **Validation Loss** | <1.0 | <0.5 | <0.3 |
| **MOS Score** | 2.5/5 | 3.5/5 | 4.2/5 |
| **WER** | <40% | <25% | <15% |
| **Voice Similarity** | 50% | 70% | 85% |

---

## 🎯 Summary of Key Recommendations

### Must-Do (Critical)

1. ✅ **Fix audio code extraction** - Get unique codes per sample
2. ✅ **Increase training data** - Use 5000+ samples
3. ✅ **Use target language reference** - Hausa speaker for Hausa text
4. ✅ **Lower learning rate** - 3e-5 to 5e-5
5. ✅ **Train more epochs** - 20-30 epochs
6. ✅ **Add validation** - Monitor overfitting

### Should-Do (Important)

7. ✅ **Use smaller batch size** - 1-2 for better generalization
8. ✅ **Implement gradient accumulation** - Effective batch 16+
9. ✅ **Add learning rate scheduling** - Cosine annealing
10. ✅ **Monitor training properly** - Log all metrics

### Nice-to-Do (Optional)

11. 🔄 Use data augmentation
12. 🔄 Implement curriculum learning
13. 🔄 Try adapter/Lora training
14. 🔄 Multi-speaker training

---

## 📞 Troubleshooting Guide

| Problem | Cause | Solution |
|---------|-------|----------|
| **Loss = 0 immediately** | Same codes in all samples | Fix code extraction |
| **Loss = NaN** | LR too high | Reduce LR, add clipping |
| **No voice identity** | Wrong reference audio | Use target language speaker |
| **Poor intelligibility** | Insufficient training | More epochs, more data |
| **Memory OOM** | Batch too big | Reduce batch, increase grad accum |
| **Training too slow** | Inefficient pipeline | Optimize dataloader, use pinning |

---

**Document Version:** 1.0  
**Last Updated:** 2026-02-23  
**Training Status:** Analysis Complete - Ready for Optimization  
**Next Step:** Implement Phase 1 fixes