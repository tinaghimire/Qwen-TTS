#!/usr/bin/env python3
# coding=utf-8
"""
Data preprocessing and dataloader for Qwen3-TTS training.

Used by finetune.py only. Loads from HuggingFace by subset/split, ref_audio from
voices/ (hausa_speaker.wav, english_speaker.wav), tokenizer produces audio_codes on
the fly. Use get_multispeaker_finetune_dataloader() for train/val.

For data preparation and upload (CLI), run:
    python data_preparation.py --speaker both --upload --repo_id "vaghawan/qwen3-tts-multi-speaker"
"""

import base64
import io
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import DataLoader as TorchDataLoader
from dotenv import load_dotenv

# .env and project paths
_env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(_env_file):
    load_dotenv(_env_file)
else:
    load_dotenv()

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "finetuning"))

from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

try:
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
except ImportError:
    Qwen3TTSConfig = None

# Config from .env
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "Qwen/Qwen3-TTS-Tokenizer-12Hz")
DEVICE = os.getenv("DEVICE", "cuda")
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "150000")) if os.getenv("MAX_TRAIN_SAMPLES") else 150000
MAX_VAL_SAMPLES = int(os.getenv("MAX_VAL_SAMPLES", "15000")) if os.getenv("MAX_VAL_SAMPLES") else 15000


def get_device_for_current_process() -> str:
    """Device for current process (multi-GPU aware)."""
    try:
        from accelerate import Accelerator
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            return f"cuda:{accelerator.process_index}"
    except Exception:
        pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_num_workers_for_dataloader(num_gpus: Optional[int] = None) -> int:
    """Recommended DataLoader num_workers."""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    import multiprocessing
    total_cpus = multiprocessing.cpu_count()
    recommended = min(2 * num_gpus, 8)
    if total_cpus < 8:
        recommended = min(recommended, total_cpus // 2)
    return max(0, min(recommended, total_cpus))


def base64_to_audio(audio_b64: str) -> Tuple[np.ndarray, int]:
    """Decode base64 audio string to (array, sampling_rate)."""
    wav_bytes = base64.b64decode(audio_b64)
    with io.BytesIO(wav_bytes) as buffer:
        audio_array, sr = sf.read(buffer, dtype='float32')
    return audio_array, sr


class MultiSpeakerTTSDataLoader(TorchDataset):
    """
    Load from HuggingFace multi-speaker repo by subset/split; decode base64 audio,
    encode with 12Hz tokenizer, set ref_audio from voices/{subset}.wav.
    """

    def __init__(
        self,
        repo_id: str,
        subset: str,
        split: str = "train",
        tokenizer_path: str = TOKENIZER_PATH,
        max_samples: Optional[int] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.repo_id = repo_id
        self.subset = subset
        self.split = split
        self.tokenizer_path = tokenizer_path
        self.max_samples = max_samples
        self.device = device or get_device_for_current_process()

        print(f"Loading {subset} ({split}) from {repo_id}...")
        self.hf_dataset = load_dataset(repo_id, name=subset, split=split, cache_dir=cache_dir)
        if max_samples is not None:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))
        print(f"  Loaded {len(self.hf_dataset)} samples")

        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(tokenizer_path, device_map=self.device)
        print(f"  Tokenizer on {self.device}")

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.hf_dataset[idx]
        audio_array, audio_sr = base64_to_audio(item["audio"])
        if audio_sr != 24000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=audio_sr, target_sr=24000)
            audio_sr = 24000
        with torch.no_grad():
            enc_result = self.tokenizer.encode([audio_array], sr=audio_sr)
            audio_codes = enc_result.audio_codes[0].cpu().numpy().tolist()
        ref_audio_path = os.path.join(project_root, "voices", f"{self.subset}.wav")
        return {
            "audio": audio_array,
            "text": item["text"],
            "audio_codes": audio_codes,
            "sr": audio_sr,
            "speaker": self.subset,
        }


class MultiSpeakerStreamingTTSDataset(TorchDataset):
    """
    Wraps MultiSpeakerTTSDataLoader (or ConcatDataset) and produces trainer format:
    text_ids, audio_codes, ref_mel (one per speaker from cache), target_audio, target_mel.
    """

    def __init__(self, multispeaker_dataset: TorchDataset, processor: Any, config: Any, lag_num: int = -1):
        self.multispeaker_dataset = multispeaker_dataset
        self.processor = processor
        self.config = config
        self.lag_num = lag_num

    def __len__(self) -> int:
        return len(self.multispeaker_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.multispeaker_dataset[idx]
        text, audio_codes = item["text"], item["audio_codes"]
        speaker = item["speaker"]

        assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self.processor(text=assistant_text, return_tensors="pt", padding=True)["input_ids"]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        text_ids = input_ids[:, :-5]

        # Target audio (ground-truth for this text) and its mel for reconstruction/prosody losses
        target_audio = item["audio"].astype(np.float32)
        if len(target_audio.shape) > 1:
            target_audio = np.mean(target_audio, axis=-1)
        if item.get("sr", 24000) != 24000:
            import librosa
            target_audio = librosa.resample(target_audio, orig_sr=item["sr"], target_sr=24000)
        with torch.inference_mode():
            target_mel = mel_spectrogram(
                torch.from_numpy(target_audio).unsqueeze(0),
                n_fft=1024, num_mels=128, sampling_rate=24000,
                hop_size=256, win_size=1024, fmin=0, fmax=12000,
            ).transpose(1, 2)

        return {
            "text_ids": text_ids,
            "audio_codes": torch.tensor(audio_codes, dtype=torch.long),
            "speaker": speaker,
            "target_audio": target_audio,
            "target_mel": target_mel,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if Qwen3TTSConfig is None or not hasattr(self.config, "tts_pad_token_id"):
            raise RuntimeError("Qwen3TTSConfig required for collate_fn")
        assert self.lag_num == -1
        item_length = [b["text_ids"].shape[1] + b["audio_codes"].shape[0] for b in batch]
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
            text_ids = data["text_ids"]
            audio_codec_0 = data["audio_codes"][:, 0]
            audio_codecs = data["audio_codes"]
            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8 : 8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2 : 8 + text_ids_len + codec_ids_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, : 8 + text_ids_len + codec_ids_len] = True

            input_ids[i, 3:8, 1] = torch.tensor([
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                0,
                self.config.talker_config.codec_pad_id,
            ])
            input_ids[i, 8 : 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = self.config.talker_config.codec_eos_token_id
            codec_ids[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len, :] = audio_codecs
            codec_embedding_mask[i, 3 : 8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False
            codec_mask[i, 8 + text_ids_len - 1 : 8 + text_ids_len - 1 + codec_ids_len] = True
            attention_mask[i, : 8 + text_ids_len + codec_ids_len] = True

        speakers = [d["speaker"] for d in batch]
        max_mel_len = max(d["target_mel"].shape[1] for d in batch)
        target_mel_list = []
        for d in batch:
            tm = d["target_mel"][0]
            if tm.shape[0] < max_mel_len:
                tm = torch.nn.functional.pad(tm, (0, 0, 0, max_mel_len - tm.shape[0]))
            target_mel_list.append(tm)
        target_mels = torch.stack(target_mel_list, dim=0)
        max_audio_len = max(len(d["target_audio"]) for d in batch)
        target_audios = torch.zeros(b, max_audio_len, dtype=torch.float32)
        target_audio_lengths = torch.zeros(b, dtype=torch.long)
        for i, d in enumerate(batch):
            ta = d["target_audio"]
            if isinstance(ta, np.ndarray):
                ta = torch.from_numpy(ta)
            L = len(ta)
            target_audios[i, :L] = ta
            target_audio_lengths[i] = L

        return {
            "input_ids": input_ids,
            "speakers": speakers,
            "target_mel": target_mels,
            "target_audio": target_audios,
            "target_audio_lengths": target_audio_lengths,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
        }


def get_multispeaker_finetune_dataloader(
    repo_id: str = "vaghawan/qwen3-tts-multi-speaker",
    speakers: Optional[List[str]] = None,
    split: str = "train",
    processor: Any = None,
    config: Any = None,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    tokenizer_path: str = TOKENIZER_PATH,
    device: Optional[str] = None,
    num_workers: Optional[int] = None,
    shuffle: bool = True,
    cache_dir: Optional[str] = None,
) -> TorchDataLoader:
    """
    Combined train/val DataLoader for finetune.py. Requires processor and config.

    CUDA: Uses pin_memory=True when CUDA is available for faster CPU->GPU transfer.
    Batches are moved to the model device in training_step() in finetune.py.
    With Accelerator (multi-GPU), batch_size is per GPU.

    Batch size: Any batch_size (e.g. 64) is supported; limit is GPU memory.
    If OOM, reduce BATCH_SIZE in .env or use GRADIENT_ACCUMULATION_STEPS for larger effective batch.
    """
    if processor is None or config is None:
        raise ValueError("get_multispeaker_finetune_dataloader requires processor and config")
    if speakers is None:
        speakers = ["hausa_speaker", "english_speaker"]
    if max_samples is None:
        max_samples = MAX_VAL_SAMPLES if split == "validation" else MAX_TRAIN_SAMPLES

    from torch.utils.data import ConcatDataset

    datasets_list = [
        MultiSpeakerTTSDataLoader(
            repo_id=repo_id,
            subset=sp,
            split=split,
            tokenizer_path=tokenizer_path,
            max_samples=max_samples,
            device=device,
            cache_dir=cache_dir,
        )
        for sp in speakers
    ]
    combined = ConcatDataset(datasets_list)
    wrapper = MultiSpeakerStreamingTTSDataset(combined, processor, config)

    if num_workers is None:
        num_workers = get_num_workers_for_dataloader(torch.cuda.device_count()) if torch.cuda.is_available() else 0

    dl = TorchDataLoader(
        wrapper,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=wrapper.collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    print(f"✓ Multi-speaker DataLoader: {len(combined)} samples, speakers={speakers}, split={split}")
    return dl


if __name__ == "__main__":
    print("Data preprocessing and dataloader module (used by finetune.py).")
    print("For prepare & upload, run: python data_preparation.py --speaker both --upload --repo_id <repo_id>")
