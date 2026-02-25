# Qwen3-TTS fine-tuning (multi-speaker)

Fine-tune Qwen3-TTS with **two speakers** in one pipeline: data is **uploaded once** to a single Hugging Face dataset repo, then training uses a **CUDA-friendly dataloader**, **dual-loss training** (including auxiliary losses from `dual_loss_trainer`), and **extended evaluation metrics**. With enough GPU memory, you can increase batch size for faster training.

## Ready for training?

Before you start, ensure:

1. **Dependencies** – From `Qwen3-TTS`: `uv sync`
2. **Environment** – Copy `Qwen3-TTS/.env.training.example` to `Qwen3-TTS/.env` and set at least:
   - `HF_DATASET_REPO` (your dataset repo, e.g. `username/qwen3-tts-multi-speaker`)
   - `HF_TOKEN` (if you upload or use private data)
   - `TRAIN_SPEAKERS` (e.g. `hausa_speaker,english_speaker`)
3. **Reference audio** – Place one WAV per speaker under `Qwen3-TTS/voices/`:
   - `voices/hausa_speaker.wav`
   - `voices/english_speaker.wav`
4. **Data uploaded** – Run Step 1 once so the dataset repo has train/validation for both speakers.

Then run Step 3 (fine-tuning) from `Qwen3-TTS`: `uv run python finetuning/finetune.py`.

## Summary

- **Dataloader**: Two-speaker (e.g. Hausa + English) from one Hugging Face dataset; batches are CUDA-ready.
- **Loss**: Main loss plus auxiliary losses from `finetuning/dual_loss_trainer.py` (mel reconstruction, voice consistency, prosody).
- **Evaluation**: Validation uses five metrics: perplexity, speaker_embedding_consistency, pronunciation_accuracy, tonal_accuracy, prosody_accuracy (generated vs reference audio).
- **CUDA batching**: All data loading and training support GPU batching; larger batch size when GPU memory allows for faster training.
- **Data upload once**: One dataset repo holds both speakers (per-speaker configs); no need to re-upload for each run.

---

## Prerequisites

- **Python**: 3.10–3.12  
- **uv**: [Install uv](https://docs.astral.sh/uv/getting-started/installation/)  
- **CUDA**: For GPU training  
- **Hugging Face**: Account and token for dataset upload/download  

---

## Setup

From the **`Qwen3-TTS`** directory (where `pyproject.toml` and `.env` live):

```bash
cd Qwen3-TTS

# Install dependencies with uv
uv sync

# Copy and edit env (repo id, HF token, speakers, etc.)
cp .env.training.example .env
# Edit .env: HF_DATASET_REPO, HF_TOKEN, TRAIN_SPEAKERS, OUTPUT_DIR, etc.
```

---

## Step 1: Data preparation (once per dataset)

Prepare train/validation data for **both** speakers and upload **once** to a single Hugging Face dataset repo. Each speaker is a separate config (e.g. `hausa_speaker`, `english_speaker`) in the same repo.

From **`Qwen3-TTS`**:

```bash
uv run python finetuning/data_preparation.py --speaker both --upload --repo_id "YOUR_USER/qwen3-tts-multi-speaker"
```

- `--speaker both`: prepare Hausa and English (or use `hausa` / `english`).
- `--upload`: push to Hugging Face after preparation.
- `--repo_id`: target dataset repo (e.g. `vaghawan/qwen3-tts-multi-speaker`).

Optional:

- `--max_samples N`: limit samples per split.
- `--output_dir DIR`: where to write JSONL before upload (default: `./data`).

Reference audio for each speaker is read at **load time** from `voices/<speaker>.wav` (e.g. `voices/hausa_speaker.wav`, `voices/english_speaker.wav`). Ensure those files exist under the project (e.g. under `Qwen3-TTS` or the path your code uses).

---

## Step 2: Data processing (used during training)

There is no separate “processing” step to run by hand. **Processing is done inside training**:

- `finetuning/data_processing.py` provides `get_multispeaker_finetune_dataloader()`.
- Training (Step 3) loads from the Hugging Face repo by split and speaker config, builds batches (with ref audio from `voices/` and tokenizer-derived audio codes), and moves them to CUDA. So **data processing = dataloader usage during finetuning**.

---

## Step 3: Fine-tuning

From **`Qwen3-TTS`**:

**Single GPU**

```bash
uv run python finetuning/finetune.py
```

**Multi-GPU (e.g. 4 GPUs)**

```bash
uv run accelerate launch --num_processes=4 finetuning/finetune.py
```

Important `.env` options:

- `HF_DATASET_REPO`: same repo you used in Step 1 (e.g. `YOUR_USER/qwen3-tts-multi-speaker`).
- `TRAIN_SPEAKERS`: comma-separated list, e.g. `hausa_speaker,english_speaker`.
- `BATCH_SIZE`: increase for faster training if GPU memory allows.
- `OUTPUT_DIR`: where checkpoints and logs are saved.

Training uses:

- **Loss**: main talker loss + optional auxiliary losses from `dual_loss_trainer` (mel reconstruction, voice consistency, prosody) when `USE_AUXILIARY_LOSSES=true` and batch includes `target_audio` / `target_mel`.
- **Evaluation**: validation runs the five metrics (perplexity, speaker_embedding_consistency, pronunciation_accuracy, tonal_accuracy, prosody_accuracy) by generating audio from the model and comparing to reference (target) audio.

---

## Pipeline overview

| Step              | Script / component              | Command / usage |
|------------------|----------------------------------|-----------------|
| 1. Data prep     | `finetuning/data_preparation.py` | `uv run python finetuning/data_preparation.py --speaker both --upload --repo_id "..."` |
| 2. Processing    | `finetuning/data_processing.py`  | Used inside finetuning (dataloader) |
| 3. Fine-tuning   | `finetuning/finetune.py`         | `uv run python finetuning/finetune.py` or `uv run accelerate launch --num_processes=N finetuning/finetune.py` |

All commands are intended to be run with **`uv run`** from the **`Qwen3-TTS`** directory.

---

## Project layout (relevant parts)

```
Qwen3-TTS-finetuning/
├── README.md
└── Qwen3-TTS/
    ├── .env
    ├── .env.training.example
    ├── pyproject.toml
    ├── voices/
    │   ├── hausa_speaker.wav
    │   └── english_speaker.wav
    └── finetuning/
        ├── finetune.py           # Training entrypoint
        ├── data_preparation.py   # Step 1: prepare + upload dataset
        ├── data_processing.py    # Dataloader (Step 2, used in Step 3)
        ├── dual_loss_trainer.py  # Auxiliary losses
        └── quality_metrics.py    # Evaluation metrics
```

---

## Summary checklist

- **Dataloader with two speakers**: Yes — one HF repo, multiple speaker configs; `get_multispeaker_finetune_dataloader` builds train/val loaders.
- **Loss including dual_loss_trainer**: Yes — optional mel reconstruction, voice consistency, prosody (when auxiliary losses enabled and batch has target audio/mel).
- **Evaluation extended with other metrics**: Yes — perplexity, speaker_embedding_consistency, pronunciation_accuracy, tonal_accuracy, prosody_accuracy.
- **CUDA batching**: Yes — batches are moved to GPU; you can raise `BATCH_SIZE` when GPU memory allows.
- **Faster training with enough GPU/memory**: Yes — increase `BATCH_SIZE` and/or use multi-GPU with `accelerate launch`.
- **Data upload once for two speakers**: Yes — Step 1 uploads both speakers to one repo; Step 3 uses that repo for all runs.

All steps use **`uv`** and **`uv run`** as shown above.
