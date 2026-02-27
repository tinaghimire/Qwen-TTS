#!/usr/bin/env python3
# coding=utf-8
"""
Data preprocessing and dataloader for Qwen3-TTS training.

Used by finetune.py only.

Data shape (train and validation):
  - Both speakers (e.g. hausa_speaker, english_speaker) are combined into one dataset;
    each row has: target_audio, text, speaker, audio_codes.
  - For hausa_speaker, additional local data is included in the *train* split only when present: CSV at
    data/hausa-mtn-test.csv (transcript, audio file id) and WAVs in data/provided-test-speech/ (override with HAUSA_EXTRA_CSV / HAUSA_EXTRA_AUDIO_DIR).
  - target_audio: ground-truth waveform for this utterance → used for reconstruction
    loss, pauses, pronunciation, tone/pitch (dual_loss_trainer). Also converted to
    target_mel for mel losses.
  - text: transcript.
  - speaker: subset name (e.g. hausa_speaker) for ref lookup.
  - audio_codes: from 12Hz tokenizer on target_audio (training targets).
    Used for inference/generation; base model may not support "hausa" (use LANGUAGE_FOR_HAUSA in .env).
  - Ref audio: not stored per row; finetune.py loads from voices/{speaker}.wav once per speaker
    (REF_AUDIO_CACHE / REF_MEL_CACHE) and passes ref_mels in the batch for speaker
    embedding and matching.

Use get_multispeaker_finetune_dataloader() for train/val.

For data preparation and upload (CLI), run:
    python data_preparation.py --speaker both --upload --repo_id "vaghawan/qwen3-tts-multi-speaker"
"""

import base64
import csv
import io
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from torch.utils.data import DataLoader as TorchDataLoader
from dotenv import load_dotenv

# .env and project paths
_env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(_env_file):
    load_dotenv(_env_file, override=True)
else:
    load_dotenv(override=True)

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

# Speaker → language for conditioning (hausa_speaker → Hausa, english_speaker → English).
# Base Qwen3-TTS may not list "hausa"; use LANGUAGE_FOR_HAUSA in .env for inference fallback.
# To accept language="hausa" in generation: extend talker config (codec_language_id + token id / embeddings). For now, hausa_speaker + LANGUAGE_FOR_HAUSA=english is supported. During fine-tuning the model learns language conditioning via the codec prefill (think path + language_id) below.

# Optional extra Hausa data: CSV (transcript, audio_filename) + directory of WAVs. Used in addition to HF hausa_speaker.
def _hausa_extra_paths() -> Tuple[Optional[str], Optional[str]]:
    """Return (csv_path, audio_dir) for local Hausa MTN test data; (None, None) if not present."""
    _data_root = os.path.join(project_root, "data")
    csv_path = os.getenv("HAUSA_EXTRA_CSV") or os.path.join(_data_root, "hausa-mtn-test.csv")
    audio_dir = os.getenv("HAUSA_EXTRA_AUDIO_DIR") or os.path.join(_data_root, "provided-test-speech")
    if os.path.isfile(csv_path) and os.path.isdir(audio_dir):
        return csv_path, audio_dir
    return None, None


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
    """Recommended DataLoader num_workers to keep GPU fed (avoids 0% GPU / 90% CPU bottleneck)."""
    if num_gpus is None:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    import multiprocessing
    total_cpus = multiprocessing.cpu_count()
    # Per-GPU workers: more when we have many CPUs so GPU isn't starved
    per_gpu = min(4 * max(1, num_gpus), 12)
    if total_cpus >= 64:
        # Very high-CPU server (e.g. 128+ cores): allow more workers for full CPU use
        per_gpu = min(64, total_cpus // 4)
    elif total_cpus >= 32:
        # High-CPU server (e.g. 1 A100 + 128 cores)
        per_gpu = min(48, total_cpus // 4)
    elif total_cpus >= 16:
        per_gpu = min(16, total_cpus // 2)
    elif total_cpus < 8:
        per_gpu = min(per_gpu, max(1, total_cpus // 2))
    return max(0, min(per_gpu, total_cpus - 1))


def base64_to_audio(audio_b64: str) -> Tuple[np.ndarray, int]:
    """Decode base64 audio string to (array, sampling_rate)."""
    wav_bytes = base64.b64decode(audio_b64)
    with io.BytesIO(wav_bytes) as buffer:
        audio_array, sr = sf.read(buffer, dtype='float32')
    return audio_array, sr


def dataset_audio_to_array_sr(audio_value: Any) -> Tuple[np.ndarray, int]:
    """
    Convert HuggingFace dataset 'audio' column to (array, sampling_rate).
    Supports: dict with 'array'/'sampling_rate', dict with 'bytes', legacy base64 string,
    datasets AudioDecoder (get_all_samples), .array/.sampling_rate, .decode(), and raw ndarray.
    """
    if isinstance(audio_value, dict):
        if "array" in audio_value:
            arr = audio_value["array"]
            sr = audio_value.get("sampling_rate", 24000)
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr, dtype=np.float32)
            return arr.astype(np.float32), int(sr)
        if "bytes" in audio_value:
            with io.BytesIO(audio_value["bytes"]) as buf:
                arr, sr = sf.read(buf, dtype="float32")
            return arr, int(sr)
        if "path" in audio_value:
            arr, sr = sf.read(audio_value["path"], dtype="float32")
            return arr.astype(np.float32), int(sr)
        raise ValueError(f"Unknown audio dict keys: {list(audio_value.keys())}")
    if isinstance(audio_value, str):
        return base64_to_audio(audio_value)
    # datasets.features._torchcodec.AudioDecoder (get_all_samples -> .data tensor, .sample_rate)
    if type(audio_value).__name__ == "AudioDecoder" and hasattr(audio_value, "get_all_samples"):
        samples = audio_value.get_all_samples()
        data = samples.data
        if hasattr(data, "numpy"):
            arr = data.numpy().squeeze().astype(np.float32)
        else:
            arr = np.array(data).squeeze().astype(np.float32)
        sr = int(getattr(samples, "sample_rate", None) or getattr(getattr(audio_value, "metadata", None), "sample_rate", 24000))
        return arr, sr
    # Lazy/decodeable: .array / .sampling_rate
    if hasattr(audio_value, "array") and hasattr(audio_value, "sampling_rate"):
        arr = audio_value.array
        sr = audio_value.sampling_rate
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=np.float32)
        return arr.astype(np.float32), int(sr)
    # .decode() (e.g. some HuggingFace audio column types)
    if hasattr(audio_value, "decode"):
        decoded = audio_value.decode()
        if isinstance(decoded, dict) and "array" in decoded:
            arr = decoded["array"]
            sr = decoded.get("sampling_rate", 24000)
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr, dtype=np.float32)
            return arr.astype(np.float32), int(sr)
        raise ValueError(f"Decoded audio unexpected type: {type(decoded)}")
    if isinstance(audio_value, np.ndarray):
        return audio_value.astype(np.float32), 24000
    raise ValueError(f"Unsupported audio type: {type(audio_value).__name__}")


class MultiSpeakerTTSDataLoader(TorchDataset):
    """
    Load from HuggingFace multi-speaker repo by subset/split. Each row: target_audio, text,
    speaker, audio_codes. Ref audio is loaded by finetune.py from voices/{speaker}.wav
    (global path per speaker, not per row).
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
        # Tokenizer/audio codes: use CPU in workers (safe with fork) or CUDA (requires spawn).
        self.device = (device or "cpu").lower()
        if self.device not in ("cpu", "cuda"):
            self.device = "cpu"

        print(f"Loading {subset} ({split}) from {repo_id}...")
        self.hf_dataset = load_dataset(repo_id, name=subset, split=split, cache_dir=cache_dir)
        if max_samples is not None:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))
        print(f"  Loaded {len(self.hf_dataset)} samples")

        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(tokenizer_path, device_map=self.device)
        print(f"  Tokenizer on {self.device} (dataset workers)")
        # Ref audio: finetune.py loads from voices/{speaker}.wav (global per speaker)

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.hf_dataset[idx]
        target_audio, audio_sr = dataset_audio_to_array_sr(item["audio"])
        if audio_sr != 24000:
            import librosa
            target_audio = librosa.resample(target_audio, orig_sr=audio_sr, target_sr=24000)
            audio_sr = 24000
        with torch.no_grad():
            enc_result = self.tokenizer.encode([target_audio], sr=audio_sr)
            ac = enc_result.audio_codes[0]
            # Keep on same device as tokenizer; avoid .tolist() by returning numpy for faster collate
            if hasattr(ac, "cpu"):
                audio_codes = ac.cpu().numpy()
            else:
                audio_codes = np.array(ac, dtype=np.int64)
        return {
            "target_audio": target_audio,
            "text": item["text"],
            "audio_codes": audio_codes,
            "sr": audio_sr,
            "speaker": self.subset,
            "language": SPEAKER_LANGUAGE.get(self.subset, "english"),
        }


class LocalHausaCSVDataset(TorchDataset):
    """
    Load hausa_speaker samples from a local CSV (transcript, audio filename) and audio directory.
    Same output shape as MultiSpeakerTTSDataLoader for use with MultiSpeakerStreamingTTSDataset collate.
    CSV format: header row, then rows with transcript and audio file id (e.g. "1" for 1.wav).
    """

    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        speaker: str = "hausa_speaker",
        tokenizer_path: str = TOKENIZER_PATH,
        max_samples: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.csv_path = csv_path
        self.audio_dir = audio_dir
        self.speaker = speaker
        self.tokenizer_path = tokenizer_path
        self.max_samples = max_samples
        self.device = (device or "cpu").lower()
        if self.device not in ("cpu", "cuda"):
            self.device = "cpu"

        self.rows: List[Tuple[str, str]] = []  # (text, audio_basename)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0].strip() and row[1].strip():
                    text = row[0].strip()
                    audio_key = row[1].strip()
                    self.rows.append((text, audio_key))
        if max_samples is not None:
            self.rows = self.rows[: max_samples]
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(tokenizer_path, device_map=self.device)
        print(f"  Local Hausa CSV: {len(self.rows)} samples from {csv_path} + {audio_dir}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_audio_path(self, audio_basename: str) -> Optional[str]:
        """Resolve audio file path; try with and without .wav extension."""
        for name in (f"{audio_basename}.wav", audio_basename, f"{audio_basename}.mp3"):
            p = os.path.join(self.audio_dir, name)
            if os.path.isfile(p):
                return p
        return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text, audio_key = self.rows[idx]
        audio_path = self._resolve_audio_path(audio_key)
        if audio_path is None:
            raise FileNotFoundError(f"Audio not found for key '{audio_key}' in {self.audio_dir}")
        target_audio, audio_sr = sf.read(audio_path, dtype="float32")
        if target_audio.ndim > 1:
            target_audio = np.mean(target_audio, axis=-1)
        target_audio = target_audio.astype(np.float32)
        if audio_sr != 24000:
            import librosa
            target_audio = librosa.resample(target_audio, orig_sr=audio_sr, target_sr=24000)
            audio_sr = 24000
        with torch.no_grad():
            enc_result = self.tokenizer.encode([target_audio], sr=audio_sr)
            ac = enc_result.audio_codes[0]
            if hasattr(ac, "cpu"):
                audio_codes = ac.cpu().numpy()
            else:
                audio_codes = np.array(ac, dtype=np.int64)
        return {
            "target_audio": target_audio,
            "text": text,
            "audio_codes": audio_codes,
            "sr": audio_sr,
            "speaker": self.speaker,
            "language": SPEAKER_LANGUAGE.get(self.speaker, "english"),
        }


class MultiSpeakerStreamingTTSDataset(TorchDataset):
    """
    Wraps combined per-speaker datasets and produces trainer batches: text_ids, audio_codes,
    target_audio, target_mel, speakers. ref_mel is looked up by speaker in finetune.py
    (REF_MEL_CACHE from voices/).

    - target_audio / target_mel: ground-truth for this utterance → reconstruction loss,
      pauses, pronunciation, tone/pitch (dual_loss_trainer).
    - ref_audio (voices/{speaker}.wav): reference for speaker characteristics and
      encoding; loaded once per speaker in finetune.py.
    """

    def __init__(self, multispeaker_dataset: TorchDataset, processor: Any, config: Any, lag_num: int = -1, speaker_list: Optional[List[str]] = None):
        self.multispeaker_dataset = multispeaker_dataset
        self.processor = processor
        self.config = config
        self.lag_num = lag_num
        self.speaker_list = speaker_list or []

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

        # target_audio: ground-truth for reconstruction, pauses, pronunciation, tone/pitch
        target_audio = item["target_audio"].astype(np.float32)
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
            "language": item.get("language", SPEAKER_LANGUAGE.get(speaker, "english")),
            "target_audio": target_audio,
            "target_mel": target_mel,
        }

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if Qwen3TTSConfig is None or not hasattr(self.config, "tts_pad_token_id"):
            raise RuntimeError("Qwen3TTSConfig required for collate_fn")
        assert self.lag_num == -1
        b = len(batch)
        max_length = 0
        max_mel_len = 0
        max_audio_len = 0
        speakers = []
        languages = []
        for d in batch:
            tl = d["text_ids"].shape[1] + d["audio_codes"].shape[0]
            if tl + 8 > max_length:
                max_length = tl + 8
            mel_len = d["target_mel"].shape[1]
            if mel_len > max_mel_len:
                max_mel_len = mel_len
            al = len(d["target_audio"])
            if al > max_audio_len:
                max_audio_len = al
            speakers.append(d["speaker"])
            languages.append(d.get("language", SPEAKER_LANGUAGE.get(d["speaker"], "english")))
        t = max_length
        # speaker_ids: tensor so Accelerator can concatenate batches (e.g. gradient accumulation)
        speaker_to_id = {s: i for i, s in enumerate(self.speaker_list)} if self.speaker_list else {}
        if speaker_to_id:
            speaker_ids = torch.tensor([speaker_to_id.get(d["speaker"], 0) for d in batch], dtype=torch.long)
        else:
            speaker_ids = torch.zeros(b, dtype=torch.long)

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

            # Language conditioning: use think path + language_id when supported so the model learns language
            lang = data.get("language", SPEAKER_LANGUAGE.get(data["speaker"], "english"))
            tc = self.config.talker_config
            if language_id is not None:
                # [codec_think_id, codec_think_bos_id, language_id, 0 (speaker slot), codec_think_eos_id]
                input_ids[i, 3:8, 1] = torch.tensor([
                    tc.codec_think_id,
                    tc.codec_think_bos_id,
                    language_id,
                    0,
                    tc.codec_think_eos_id,
                ])
            else:
                input_ids[i, 3:8, 1] = torch.tensor([
                    tc.codec_nothink_id,
                    tc.codec_think_bos_id,
                    tc.codec_think_eos_id,
                    0,
                    tc.codec_pad_id,
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

        target_mel_list = []
        for d in batch:
            tm = d["target_mel"][0]
            if tm.shape[0] < max_mel_len:
                tm = torch.nn.functional.pad(tm, (0, 0, 0, max_mel_len - tm.shape[0]))
            target_mel_list.append(tm)
        target_mels = torch.stack(target_mel_list, dim=0)
        target_audios = torch.zeros(b, max_audio_len, dtype=torch.float32)
        target_audio_lengths = torch.zeros(b, dtype=torch.long)
        for i, d in enumerate(batch):
            ta = d["target_audio"]
            if isinstance(ta, np.ndarray):
                ta = torch.from_numpy(ta)
            L = len(ta)
            target_audios[i, :L] = ta
            target_audio_lengths[i] = L

        # ref_mels added in finetune.py from REF_MEL_CACHE by speaker (for speaker encoding / matching)
        # speaker_ids (tensor) so Accelerator can concatenate batches; trainer resolves to names via config.train_speakers
        return {
            "input_ids": input_ids,
            "speaker_ids": speaker_ids,
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


class MultiSpeakerStreamingTTSIterableDataset(TorchIterableDataset):
    """
    Streaming (IterableDataset) version: does not load full split into RAM.
    Uses load_dataset(..., streaming=True) and interleave_datasets; yields same
    sample format as MultiSpeakerStreamingTTSDataset for batch-compatible collate.
    Worker-shards the stream so each DataLoader worker sees a disjoint subset.
    """

    def __init__(
        self,
        repo_id: str,
        speakers: List[str],
        split: str,
        processor: Any,
        config: Any,
        tokenizer_path: str = TOKENIZER_PATH,
        tokenizer_device: str = "cpu",
        max_samples: Optional[int] = None,
        shuffle_buffer_size: int = 1000,
        cache_dir: Optional[str] = None,
    ):
        self.repo_id = repo_id
        self.speakers = speakers
        self.split = split
        self.processor = processor
        self.config = config
        self.tokenizer_path = tokenizer_path
        self.tokenizer_device = (tokenizer_device or "cpu").lower()
        if self.tokenizer_device not in ("cpu", "cuda"):
            self.tokenizer_device = "cpu"
        self.max_samples = max_samples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.cache_dir = cache_dir
        self._tokenizer = None

    def __iter__(self):
        from torch.utils.data import get_worker_info
        try:
            from datasets import load_dataset
        except ImportError:
            raise RuntimeError("Streaming dataset requires 'datasets' (pip install datasets)")

        worker_info = get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        # One streaming dataset per speaker; we round-robin and tag speaker so we don't load full split into RAM
        stream_iters = []
        for sp in self.speakers:
            ds = load_dataset(
                self.repo_id,
                name=sp,
                split=self.split,
                streaming=True,
                cache_dir=self.cache_dir,
            )
            stream_iters.append((sp, iter(ds)))

        if self._tokenizer is None:
            self._tokenizer = Qwen3TTSTokenizer.from_pretrained(
                self.tokenizer_path, device_map=self.tokenizer_device
            )
        tokenizer = self._tokenizer
        processor = self.processor
        config = self.config
        count = 0
        buffer: List[Dict[str, Any]] = []
        import random

        def _generate():
            while stream_iters:
                for i in range(len(stream_iters)):
                    sp, it = stream_iters[i]
                    try:
                        item = next(it)
                        item["speaker"] = sp
                        yield item
                    except StopIteration:
                        stream_iters[i] = None
                stream_iters[:] = [x for x in stream_iters if x is not None]

        for item in _generate():
            if self.max_samples is not None and count >= self.max_samples:
                break
            if num_workers > 0 and (count % num_workers) != worker_id:
                count += 1
                continue
            count += 1

            target_audio, audio_sr = dataset_audio_to_array_sr(item["audio"])
            if audio_sr != 24000:
                import librosa
                target_audio = librosa.resample(target_audio, orig_sr=audio_sr, target_sr=24000)
                audio_sr = 24000
            with torch.no_grad():
                enc_result = tokenizer.encode([target_audio], sr=audio_sr)
                ac = enc_result.audio_codes[0]
                if hasattr(ac, "cpu"):
                    audio_codes = ac.cpu().numpy()
                else:
                    audio_codes = np.array(ac, dtype=np.int64)

            speaker = item["speaker"]

            assistant_text = f"<|im_start|>assistant\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
            input_ids = processor(text=assistant_text, return_tensors="pt", padding=True)["input_ids"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            text_ids = input_ids[:, :-5]

            target_audio = target_audio.astype(np.float32)
            if len(target_audio.shape) > 1:
                target_audio = np.mean(target_audio, axis=-1)
            with torch.inference_mode():
                target_mel = mel_spectrogram(
                    torch.from_numpy(target_audio).unsqueeze(0),
                    n_fft=1024, num_mels=128, sampling_rate=24000,
                    hop_size=256, win_size=1024, fmin=0, fmax=12000,
                ).transpose(1, 2)

            sample = {
                "text_ids": text_ids,
                "audio_codes": torch.tensor(audio_codes, dtype=torch.long),
                "speaker": speaker,
                "language": SPEAKER_LANGUAGE.get(speaker, "english"),
                "target_audio": target_audio,
                "target_mel": target_mel,
            }
            buffer.append(sample)
            if self.shuffle_buffer_size > 0 and len(buffer) >= self.shuffle_buffer_size:
                random.shuffle(buffer)
                for s in buffer:
                    yield s
                buffer = []
        for s in buffer:
            yield s


def get_multispeaker_finetune_dataloader(
    repo_id: str = "vaghawan/qwen3-tts-multi-speaker",
    speakers: Optional[List[str]] = None,
    split: str = "train",
    processor: Any = None,
    config: Any = None,
    batch_size: int = 4,
    max_samples: Optional[int] = None,
    tokenizer_path: str = TOKENIZER_PATH,
    tokenizer_device: str = "cpu",
    num_workers: Optional[int] = None,
    prefetch_factor: Optional[int] = 4,
    persistent_workers: bool = True,
    shuffle: bool = True,
    cache_dir: Optional[str] = None,
    use_streaming: bool = False,
    shuffle_buffer_size: int = 1000,
) -> TorchDataLoader:
    """
    Combined train or validation DataLoader: both speakers (e.g. hausa_speaker, english_speaker)
    are concatenated; each batch row has target_audio, text, speaker, audio_codes.
    target_audio is used for reconstruction, pauses, pronunciation, tone/pitch; ref audio
    is loaded by finetune.py from voices/{speaker}.wav (global per speaker) for speaker
    characteristics.

    tokenizer_device: "cpu" (default) for tokenizer/audio codes in workers (safe with fork),
    or "cuda" (requires multiprocessing start method "spawn" set by finetune.py).

    num_workers: None = auto (4*num_gpus, capped); set via DATALOADER_NUM_WORKERS for faster loading.
    prefetch_factor: Batches to prefetch per worker (only when num_workers > 0). Keeps GPU fed.
    persistent_workers: Keep workers alive between epochs to avoid restart overhead.

    CUDA: Uses pin_memory=True when CUDA is available for faster CPU->GPU transfer.
    Batches are moved to the model device in training_step() in finetune.py.
    With Accelerator (multi-GPU), batch_size is per GPU.

    Batch size: Any batch_size (e.g. 64) is supported; limit is GPU memory.
    If OOM, reduce BATCH_SIZE in .env or use GRADIENT_ACCUMULATION_STEPS for larger effective batch.

    Streaming: When use_streaming=True, num_workers is capped (train 16, validation 4) because
    HuggingFace streaming datasets split the data into a fixed number of shards and limit
    workers to num_shards; extra workers would be idle.
    """
    if processor is None or config is None:
        raise ValueError("get_multispeaker_finetune_dataloader requires processor and config")
    if speakers is None:
        speakers = ["hausa_speaker", "english_speaker"]
    if max_samples is None:
        max_samples = MAX_VAL_SAMPLES if split == "validation" else MAX_TRAIN_SAMPLES

    if use_streaming:
        iterable_ds = MultiSpeakerStreamingTTSIterableDataset(
            repo_id=repo_id,
            speakers=speakers,
            split=split,
            processor=processor,
            config=config,
            tokenizer_path=tokenizer_path,
            tokenizer_device=tokenizer_device,
            max_samples=max_samples,
            shuffle_buffer_size=shuffle_buffer_size,
            cache_dir=cache_dir,
        )
        wrapper = MultiSpeakerStreamingTTSDataset(
            multispeaker_dataset=None,
            processor=processor,
            config=config,
            speaker_list=speakers,
        )
        dataset_for_dl = iterable_ds
        collate_fn = wrapper.collate_fn
    else:
        from torch.utils.data import ConcatDataset

        hausa_extra_csv, hausa_extra_audio_dir = _hausa_extra_paths()
        datasets_list: List[TorchDataset] = []
        for sp in speakers:
            ds = MultiSpeakerTTSDataLoader(
                repo_id=repo_id,
                subset=sp,
                split=split,
                tokenizer_path=tokenizer_path,
                max_samples=max_samples,
                device=tokenizer_device,
                cache_dir=cache_dir,
            )
            if sp == "hausa_speaker" and split == "train" and hausa_extra_csv and hausa_extra_audio_dir:
                extra_ds = LocalHausaCSVDataset(
                    csv_path=hausa_extra_csv,
                    audio_dir=hausa_extra_audio_dir,
                    speaker="hausa_speaker",
                    tokenizer_path=tokenizer_path,
                    max_samples=None,
                    device=tokenizer_device,
                )
                ds = ConcatDataset([ds, extra_ds])
                print(f"  Added {len(extra_ds)} local Hausa samples (hausa-mtn-test.csv + provided-test-speech)")
            datasets_list.append(ds)
        combined = ConcatDataset(datasets_list)
        wrapper = MultiSpeakerStreamingTTSDataset(combined, processor, config, speaker_list=speakers)
        dataset_for_dl = wrapper
        collate_fn = wrapper.collate_fn

    if num_workers is None:
        num_workers = get_num_workers_for_dataloader(torch.cuda.device_count()) if torch.cuda.is_available() else 0

    # Streaming: HuggingFace limits workers to dataset num_shards (one shard per worker).
    # Capping at 4 avoids "Too many dataloader workers" and avoids starting then stopping workers.
    if use_streaming and num_workers > 0:
        _max_streaming = 4  # typical HF streaming shards; more workers would be idle
        if num_workers > _max_streaming:
            print(f"  (streaming {split}: capping num_workers {num_workers} -> {_max_streaming}; dataset shards limit effective workers)")
            num_workers = _max_streaming

    use_prefetch = num_workers > 0 and prefetch_factor is not None and prefetch_factor > 0
    use_persistent = num_workers > 0 and persistent_workers

    dl_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle if not use_streaming else False,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }
    if use_prefetch:
        dl_kwargs["prefetch_factor"] = prefetch_factor
    if use_persistent:
        dl_kwargs["persistent_workers"] = True

    dl = TorchDataLoader(dataset_for_dl, **dl_kwargs)
    _total_samples = len(combined) if not use_streaming else "?"
    sample_info = f"streaming, speakers={speakers}" if use_streaming else f"{_total_samples} samples, speakers={speakers}"
    print(f"✓ Multi-speaker DataLoader: {sample_info}, split={split}, "
          f"num_workers={num_workers}, prefetch_factor={prefetch_factor if use_prefetch else 'N/A'}, "
          f"persistent_workers={use_persistent}")
    return dl


if __name__ == "__main__":
    print("Data preprocessing and dataloader module (used by finetune.py).")
    print("For prepare & upload, run: python data_preparation.py --speaker both --upload --repo_id <repo_id>")
