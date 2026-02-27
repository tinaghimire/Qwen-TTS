# Fine-tuning scripts

For **multi-speaker fine-tuning** (two speakers, one dataset upload, CUDA batching, dual-loss training, extended evaluation metrics), see the root README:

**→ [Qwen3-TTS-finetuning/README.md](../../README.md)**

Quick reference:

1. **Data preparation (once):**  
   `uv run python finetuning/data_preparation.py --speaker both --upload --repo_id "vaghawan/qwen3-tts-multi-speaker"`

2. **Fine-tuning:**  
   `uv run python finetuning/finetune.py`  
   (or `uv run accelerate launch --num_processes=N finetuning/finetune.py` for multi-GPU)

All commands from the `Qwen3-TTS` directory, using `uv run`.

---

Legacy single-speaker workflow (optional): `prepare_data.py` (add audio_codes to JSONL) and `sft_12hz.py` (SFT). The supported pipeline uses `data_preparation.py` + `data_processing.py` + `finetune.py` as described in the root README.
