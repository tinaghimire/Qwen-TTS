#!/usr/bin/env python3
# coding=utf-8
"""
Data preparation and upload for Qwen3-TTS multi-speaker training.

Loads from HuggingFace source datasets, builds in-memory datasets (audio as WAV bytes + text),
and uploads to a single HF dataset repo with two configs: hausa_speaker and english_speaker,
each with splits train and validation. No JSONL files.

Usage:
    python data_preparation.py --speaker both --upload --repo_id "vaghawan/qwen3-tts-multi-speaker"

Options:
    --speaker    hausa | english | both
    --upload     Upload to HuggingFace
    --repo_id    HuggingFace dataset repository ID
    --max_samples  Max samples per split (optional; else from .env MAX_TRAIN_SAMPLES / MAX_VAL_SAMPLES)

Hausa limits: from .env MAX_TRAIN_SAMPLES, MAX_VAL_SAMPLES.
English: train = train + test (concatenated), validation = validation split only. Output: audio + text only.
"""

import io
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, Features, Value, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from dotenv import load_dotenv

# Load .env from project root
_env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(_env_file):
    load_dotenv(_env_file, override=True)
else:
    load_dotenv(override=True)

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "150000")) if os.getenv("MAX_TRAIN_SAMPLES") else 150000
MAX_VAL_SAMPLES = int(os.getenv("MAX_VAL_SAMPLES", "15000")) if os.getenv("MAX_VAL_SAMPLES") else 15000

AUDIO_FEATURES = Features({"audio": Audio(sampling_rate=24000), "text": Value("string")})


def audio_to_wav_bytes(audio_array: np.ndarray, sr: int) -> bytes:
    """Convert audio array to WAV bytes for HuggingFace Audio column."""
    with io.BytesIO() as buffer:
        sf.write(buffer, audio_array, sr, format="WAV", subtype="PCM_16")
        return buffer.getvalue()


class HausaSpeakerDatasetPreparer:
    """Prepare Hausa data from vaghawan/hausa-tts-22k, upload as config hausa_speaker (train + validation)."""

    DEFAULT_MAX_SAMPLES = {"train": MAX_TRAIN_SAMPLES, "validation": MAX_VAL_SAMPLES}

    def __init__(self, max_samples: Optional[int] = None):
        self.max_samples = max_samples
        self.dataset_name = "vaghawan/hausa-tts-22k"
        self._splits: Dict[str, Dataset] = {}
        print(f"Hausa Dataset Preparer: {self.dataset_name}")

    def _limit_for_split(self, split: str, max_samples_per_split: Optional[int] = None) -> Optional[int]:
        if max_samples_per_split is not None:
            return max_samples_per_split
        if self.max_samples is not None:
            return self.max_samples
        return self.DEFAULT_MAX_SAMPLES.get(split)

    def prepare_split(self, split: str, max_samples: Optional[int] = None) -> Dataset:
        limit = max_samples if max_samples is not None else self._limit_for_split(split)
        print(f"\nPreparing Hausa {split}...")
        split_spec = f"{split}[:{limit}]" if limit is not None else split
        hf_dataset = load_dataset(self.dataset_name, split=split_spec, keep_in_memory=False)

        rows: List[Dict[str, Any]] = []
        for item in tqdm(hf_dataset, desc=f"Hausa {split}"):
            try:
                audio_array, audio_sr = self._extract_audio(item["audio"])
                if audio_sr != 24000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=audio_sr, target_sr=24000)
                    audio_sr = 24000
                wav_bytes = audio_to_wav_bytes(audio_array, audio_sr)
                rows.append({"audio": {"bytes": wav_bytes, "sampling_rate": 24000}, "text": item["text"]})
            except Exception as e:
                if any(err in str(e) for err in ["End of file", "Failed to open", "Invalid data", "Could not receive frame"]):
                    continue
                raise

        ds = Dataset.from_list(rows, features=AUDIO_FEATURES)
        self._splits[split] = ds
        print(f"  Prepared {len(rows)} samples for {split}")
        return ds

    def prepare_all_splits(
        self,
        splits: Optional[List[str]] = None,
        max_samples_per_split: Optional[int] = None,
    ) -> Dict[str, Dataset]:
        splits = splits or ["train", "validation"]
        for split in splits:
            ms = self._limit_for_split(split, max_samples_per_split)
            self.prepare_split(split, ms)
        print(f"Hausa preparation complete: {list(self._splits.keys())}")
        return self._splits

    def upload_to_huggingface(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
    ) -> str:
        token = token or HF_TOKEN
        api = HfApi(token=token)
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        except Exception as e:
            print(f"Repo check: {e}")

        if not self._splits:
            raise ValueError("No Hausa splits prepared. Run prepare_all_splits() first.")
        dataset_dict = DatasetDict(self._splits)
        dataset_dict.push_to_hub(repo_id=repo_id, config_name="hausa_speaker", token=token, private=private)
        print(f"✓ Hausa uploaded to {repo_id} (config: hausa_speaker, splits: train, validation)")
        return f"https://huggingface.co/datasets/{repo_id}"

    def _extract_audio(self, audio_data: Any) -> Tuple[np.ndarray, int]:
        # Dict (decoded): array+sampling_rate, or bytes, or path
        if isinstance(audio_data, dict):
            if "array" in audio_data:
                audio_array = audio_data["array"]
                audio_sr = audio_data.get("sampling_rate", 22000)
            elif "bytes" in audio_data:
                with io.BytesIO(audio_data["bytes"]) as f:
                    audio_array, audio_sr = sf.read(f, dtype="float32")
            elif "path" in audio_data:
                audio_array, audio_sr = sf.read(audio_data["path"], dtype="float32")
            else:
                raise ValueError(f"Unknown audio dict keys: {list(audio_data.keys())}")
        # datasets.features._torchcodec.AudioDecoder (get_all_samples -> .data tensor, .sample_rate)
        elif type(audio_data).__name__ == "AudioDecoder" and hasattr(audio_data, "get_all_samples"):
            samples = audio_data.get_all_samples()
            data = samples.data
            if hasattr(data, "numpy"):
                audio_array = data.numpy().squeeze().astype(np.float32)
            else:
                audio_array = np.array(data).squeeze().astype(np.float32)
            audio_sr = int(getattr(samples, "sample_rate", None) or getattr(audio_data.metadata, "sample_rate", 22000))
        # Other lazy: .array / .sampling_rate or .decode()
        elif hasattr(audio_data, "array") and hasattr(audio_data, "sampling_rate"):
            audio_array = audio_data.array
            audio_sr = audio_data.sampling_rate
        elif hasattr(audio_data, "decode"):
            decoded = audio_data.decode()
            if isinstance(decoded, dict) and "array" in decoded:
                audio_array = decoded["array"]
                audio_sr = decoded.get("sampling_rate", 22000)
            else:
                raise ValueError(f"Decoded audio unexpected type: {type(decoded)}")
        elif isinstance(audio_data, np.ndarray):
            audio_array, audio_sr = audio_data, 22000
        else:
            raise ValueError(f"Cannot extract audio from type {type(audio_data).__name__}")

        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=-1)
        return audio_array, audio_sr


class EnglishSpeakerDatasetPreparer:
    """Prepare English data from nigerian_common_voice_dataset (english), upload as config english_speaker (train + validation)."""

    DEFAULT_MAX_SAMPLES = {"train": MAX_TRAIN_SAMPLES, "validation": MAX_VAL_SAMPLES}

    def __init__(self, max_samples: Optional[int] = None):
        self.max_samples = max_samples
        self.dataset_name = "benjaminogbonna/nigerian_common_voice_dataset"
        self.subset = "english"
        self._splits: Dict[str, Dataset] = {}
        print(f"English Dataset Preparer: {self.dataset_name} ({self.subset})")

    def _limit_for_split(self, split: str, max_samples_per_split: Optional[int] = None) -> Optional[int]:
        if max_samples_per_split is not None:
            return max_samples_per_split
        if self.max_samples is not None:
            return self.max_samples
        return self.DEFAULT_MAX_SAMPLES.get(split)

    def prepare_split(self, split: str, max_samples: Optional[int] = None) -> Dataset:
        limit = max_samples if max_samples is not None else self._limit_for_split(split)
        print(f"\nPreparing English {split}...")
        # Train = train + test (both); validation = validation split only. Only audio + text in output.
        if split == "train":
            train_ds = load_dataset(self.dataset_name, name=self.subset, split="train", keep_in_memory=False)
            test_ds = load_dataset(self.dataset_name, name=self.subset, split="test", keep_in_memory=False)
            hf_dataset = concatenate_datasets([train_ds, test_ds])
            if limit is not None:
                hf_dataset = hf_dataset.select(range(min(limit, len(hf_dataset))))
        else:
            split_spec = f"validation[:{limit}]" if limit is not None else "validation"
            hf_dataset = load_dataset(self.dataset_name, name=self.subset, split=split_spec, keep_in_memory=False)

        rows: List[Dict[str, Any]] = []
        for item in tqdm(hf_dataset, desc=f"English {split}"):
            try:
                audio_array, audio_sr = self._extract_audio(item["audio"])
                if audio_sr != 24000:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=audio_sr, target_sr=24000)
                    audio_sr = 24000
                wav_bytes = audio_to_wav_bytes(audio_array, audio_sr)
                text = item.get("sentence", item.get("text", ""))
                rows.append({"audio": {"bytes": wav_bytes, "sampling_rate": 24000}, "text": text})
            except Exception as e:
                if any(err in str(e) for err in ["End of file", "Failed to open", "Invalid data", "Could not receive frame"]):
                    continue
                raise

        ds = Dataset.from_list(rows, features=AUDIO_FEATURES)
        self._splits[split] = ds
        print(f"  Prepared {len(rows)} samples for {split}")
        return ds

    def prepare_all_splits(
        self,
        splits: Optional[List[str]] = None,
        max_samples_per_split: Optional[int] = None,
    ) -> Dict[str, Dataset]:
        splits = splits or ["train", "validation"]
        for split in splits:
            ms = self._limit_for_split(split, max_samples_per_split)
            self.prepare_split(split, ms)
        print(f"English preparation complete: {list(self._splits.keys())}")
        return self._splits

    def upload_to_huggingface(
        self,
        repo_id: str,
        private: bool = False,
        token: Optional[str] = None,
    ) -> str:
        token = token or HF_TOKEN
        api = HfApi(token=token)
        try:
            api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        except Exception as e:
            print(f"Repo check: {e}")

        if not self._splits:
            raise ValueError("No English splits prepared. Run prepare_all_splits() first.")
        dataset_dict = DatasetDict(self._splits)
        dataset_dict.push_to_hub(repo_id=repo_id, config_name="english_speaker", token=token, private=private)
        print(f"✓ English uploaded to {repo_id} (config: english_speaker, splits: train, validation)")
        return f"https://huggingface.co/datasets/{repo_id}"

    def _extract_audio(self, audio_data: Any) -> Tuple[np.ndarray, int]:
        # Dict (decoded): array+sampling_rate, or bytes, or path
        if isinstance(audio_data, dict):
            if "array" in audio_data:
                audio_array = audio_data["array"]
                audio_sr = audio_data.get("sampling_rate", 22000)
            elif "bytes" in audio_data:
                with io.BytesIO(audio_data["bytes"]) as f:
                    audio_array, audio_sr = sf.read(f, dtype="float32")
            elif "path" in audio_data:
                audio_array, audio_sr = sf.read(audio_data["path"], dtype="float32")
            else:
                raise ValueError(f"Unknown audio dict keys: {list(audio_data.keys())}")
        # datasets.features._torchcodec.AudioDecoder (get_all_samples -> .data tensor, .sample_rate)
        elif type(audio_data).__name__ == "AudioDecoder" and hasattr(audio_data, "get_all_samples"):
            samples = audio_data.get_all_samples()
            data = samples.data
            if hasattr(data, "numpy"):
                audio_array = data.numpy().squeeze().astype(np.float32)
            else:
                audio_array = np.array(data).squeeze().astype(np.float32)
            audio_sr = int(getattr(samples, "sample_rate", None) or getattr(audio_data.metadata, "sample_rate", 22000))
        # Other lazy: .array / .sampling_rate or .decode()
        elif hasattr(audio_data, "array") and hasattr(audio_data, "sampling_rate"):
            audio_array = audio_data.array
            audio_sr = audio_data.sampling_rate
        elif hasattr(audio_data, "decode"):
            decoded = audio_data.decode()
            if isinstance(decoded, dict) and "array" in decoded:
                audio_array = decoded["array"]
                audio_sr = decoded.get("sampling_rate", 22000)
            else:
                raise ValueError(f"Decoded audio unexpected type: {type(decoded)}")
        elif isinstance(audio_data, np.ndarray):
            audio_array, audio_sr = audio_data, 22000
        else:
            raise ValueError(f"Cannot extract audio from type {type(audio_data).__name__}")

        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=-1)
        return audio_array, audio_sr


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-TTS: Prepare and upload multi-speaker data to HuggingFace")
    parser.add_argument("--speaker", choices=["hausa", "english", "both"], default="both")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo_id", type=str, default=HF_DATASET_REPO or "", help="HuggingFace dataset repo ID")
    args = parser.parse_args()

    if not args.upload or not args.repo_id:
        print("⚠ Upload skipped (use --upload and --repo_id to push to HuggingFace).")

    if args.speaker in ["hausa", "both"]:
        print("\n" + "=" * 60 + "\nPreparing Hausa Dataset\n" + "=" * 60)
        preparer = HausaSpeakerDatasetPreparer(max_samples=args.max_samples)
        preparer.prepare_all_splits()
        if args.upload and args.repo_id:
            preparer.upload_to_huggingface(args.repo_id)

    if args.speaker in ["english", "both"]:
        print("\n" + "=" * 60 + "\nPreparing English Dataset\n" + "=" * 60)
        preparer = EnglishSpeakerDatasetPreparer(max_samples=args.max_samples)
        preparer.prepare_all_splits()
        if args.upload and args.repo_id:
            preparer.upload_to_huggingface(args.repo_id)

    print("\nDone!")


if __name__ == "__main__":
    main()
