#!/usr/bin/env python3
# coding=utf-8
"""
Load Qwen3-TTS CustomVoice model and apply weights from a checkpoint that may be missing
config files (e.g. preprocessor_config.json, generation_config.json, speech_tokenizer).

Strategy:
1. Load the full base model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice (provides configs, processor, speech_tokenizer, generation_config).
2. Load only the model weights (model.safetensors) from the checkpoint repo.
3. Apply checkpoint weights with load_state_dict(..., strict=False) so missing/extra keys are ignored.

Usage:
  From Qwen3-TTS-finetuning/Qwen3-TTS (so uv can find dependencies):
    uv run python ../load_custom_checkpoint.py
    uv run python ../load_custom_checkpoint.py --output test_custom.wav

  Or from Qwen3-TTS-finetuning if torch/transformers/huggingface_hub/safetensors are installed:
    python load_custom_checkpoint.py
"""

import argparse
import sys
from pathlib import Path

# Add Qwen3-TTS package root so "qwen_tts" is importable when run from Qwen3-TTS-finetuning
_script_dir = Path(__file__).resolve().parent
_qwen_root = _script_dir / "Qwen3-TTS"
if str(_qwen_root) not in sys.path:
    sys.path.insert(0, str(_qwen_root))

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
CHECKPOINT_REPO = "vaghawan/tts-600k-last"


def load_checkpoint_weights(repo_id: str, filename: str = "model.safetensors") -> dict:
    """Download and load state dict from a HuggingFace repo."""
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return load_file(path)


def apply_checkpoint_weights(model, repo_id: str, verbose: bool = True) -> tuple:
    """
    Load model.safetensors (or shards) from a HuggingFace repo and apply to a Qwen3 TTS model.
    Use this before finetuning to start from a checkpoint that may be missing config files.

    Args:
        model: Qwen3TTSModel (or any object with .model that has load_state_dict).
        repo_id: HuggingFace repo id (e.g. "vaghawan/tts-600k-last").
        verbose: If True, print progress and key counts.

    Returns:
        (missing_keys, unexpected_keys) from load_state_dict(..., strict=False).
    """
    from huggingface_hub import list_repo_files

    if verbose:
        print(f"   Loading checkpoint weights from: {repo_id}")
    files = list_repo_files(repo_id=repo_id)
    weight_files = [f for f in files if f.endswith(".safetensors")]
    if not weight_files:
        raise FileNotFoundError(f"No .safetensors file found in repo {repo_id}")

    if "model.safetensors" in weight_files:
        weight_file = "model.safetensors"
    else:
        weight_file = sorted(weight_files)[0]
    if verbose:
        print(f"   Using weight file: {weight_file}")

    ckpt = load_checkpoint_weights(repo_id, weight_file)
    if len(weight_files) > 1 and weight_file != "model.safetensors":
        all_keys = set(ckpt.keys())
        for wf in sorted(weight_files):
            if wf == weight_file:
                continue
            part = load_checkpoint_weights(repo_id, wf)
            for k, v in part.items():
                if k not in all_keys:
                    ckpt[k] = v
                    all_keys.add(k)

    inner = model.model if hasattr(model, "model") else model
    missing, unexpected = inner.load_state_dict(ckpt, strict=False)
    if verbose:
        print(f"   Loaded {len(ckpt)} weight tensors from checkpoint.")
        if missing:
            print(f"   Note: {len(missing)} keys in model not in checkpoint (using base weights).")
        if unexpected:
            print(f"   Note: {len(unexpected)} keys in checkpoint not in model (ignored).")
        print("   ✓ Checkpoint weights applied.")
    return missing, unexpected


def main():
    parser = argparse.ArgumentParser(description="Load CustomVoice model and apply checkpoint weights")
    parser.add_argument("--device", default="cuda:0", help="Device for inference")
    parser.add_argument("--output", default=None, help="Optional: save a test WAV to this path")
    parser.add_argument("--base-model", default=BASE_MODEL, help="Base CustomVoice model (default: %(default)s)")
    parser.add_argument("--checkpoint", default=CHECKPOINT_REPO, help="Checkpoint repo (default: %(default)s)")
    args = parser.parse_args()

    device = args.device
    print(f"1. Loading base CustomVoice model: {args.base_model}")
    print("   (This provides configs, processor, speech_tokenizer, generation_config.)")

    try:
        tts = Qwen3TTSModel.from_pretrained(
            args.base_model,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        print("   ✓ Loaded with flash_attention_2")
    except (ImportError, Exception) as e:
        print(f"   ⚠ Flash attention not available, falling back to SDPA: {e}")
        tts = Qwen3TTSModel.from_pretrained(
            args.base_model,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        print("   ✓ Loaded with SDPA")

    print(f"\n2. Loading checkpoint weights from: {args.checkpoint}")
    apply_checkpoint_weights(tts, args.checkpoint, verbose=True)

    if args.output:
        print(f"\n3. Generating test WAV: {args.output}")
        wavs, sr = tts.generate_custom_voice(
            text="This is a test of the custom checkpoint.",
            language="English",
            speaker="Vivian",
            instruct="",
        )
        import soundfile as sf
        sf.write(args.output, wavs[0], sr)
        print(f"   ✓ Saved to {args.output}")

    print("\nDone. Use the returned `tts` object for inference, or run with --output to save a test WAV.")
    return tts


if __name__ == "__main__":
    main()
