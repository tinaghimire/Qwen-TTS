#!/usr/bin/env python3
# coding=utf-8
"""
Data preparation and upload for Qwen3-TTS multi-speaker training.

CLI only. Prepare datasets from HuggingFace to JSONL and upload to a single
HF dataset repo (audio + text only; ref_audio is added at load time from voices/).

Usage (terminal):
    python data_preparation.py --speaker both --upload --repo_id "vaghawan/qwen3-tts-multi-speaker"

Options:
    --speaker    hausa | english | both
    --upload     Upload to HuggingFace after preparation
    --repo_id    HuggingFace dataset repository ID
    --max_samples  Max samples per split (optional)
    --output_dir   Output directory for JSONL files (default: ./data)
"""

import base64
import io
import json
import os
from typing import Any, List, Optional, Tuple

import numpy as np
import soundfile as sf
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
from dotenv import load_dotenv

# Load .env from project root
_env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
if os.path.exists(_env_file):
    load_dotenv(_env_file)
else:
    load_dotenv()

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./data")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", None)
HF_TOKEN = os.getenv("HF_TOKEN", None)
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "150000")) if os.getenv("MAX_TRAIN_SAMPLES") else 150000
MAX_VAL_SAMPLES = int(os.getenv("MAX_VAL_SAMPLES", "15000")) if os.getenv("MAX_VAL_SAMPLES") else 15000


def audio_to_base64(audio_array: np.ndarray, sr: int) -> str:
    """Convert audio array to base64 string for JSONL storage."""
    with io.BytesIO() as buffer:
        sf.write(buffer, audio_array, sr, format='WAV', subtype='PCM_16')
        wav_bytes = buffer.getvalue()
    return base64.b64encode(wav_bytes).decode('utf-8')


class HausaSpeakerDatasetPreparer:
    """
    Prepare Hausa speaker dataset from HuggingFace to JSONL.
    Dataset: vaghawan/hausa-tts-22k. Output: audio (base64), text.
    """

    DEFAULT_MAX_SAMPLES = {"train": MAX_TRAIN_SAMPLES, "validation": MAX_VAL_SAMPLES}

    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        ref_audio_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.dataset_name = "vaghawan/hausa-tts-22k"
        self.ref_audio_path = ref_audio_path or os.path.join(os.path.dirname(__file__), "voices", "hausa_speaker.wav")
        print(f"Hausa Dataset Preparer: {self.dataset_name} -> {output_dir}")

    def prepare_split(self, split: str, max_samples: Optional[int] = None) -> str:
        max_samples = max_samples or self.max_samples
        print(f"\nPreparing Hausa {split}...")
        hf_dataset = load_dataset(self.dataset_name, split=split, keep_in_memory=False)
        if max_samples is not None:
            hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

        os.makedirs(self.output_dir, exist_ok=True)
        output_jsonl = os.path.join(self.output_dir, f"hausa_{split}.jsonl")
        processed_count = 0
        skipped_count = 0

        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(tqdm(hf_dataset, desc=f"Hausa {split}")):
                try:
                    audio_array, audio_sr = self._extract_audio(item["audio"])
                    if audio_sr != 24000:
                        import librosa
                        audio_array = librosa.resample(audio_array, orig_sr=audio_sr, target_sr=24000)
                        audio_sr = 24000
                    audio_b64 = audio_to_base64(audio_array, audio_sr)
                    data_entry = {"audio": audio_b64, "text": item["text"]}
                    f.write(json.dumps(data_entry, ensure_ascii=False) + '\n')
                    processed_count += 1
                except Exception as e:
                    if any(err in str(e) for err in ["End of file", "Failed to open", "Invalid data", "Could not receive frame"]):
                        skipped_count += 1
                        continue
                    raise

        print(f"  Output: {output_jsonl}, processed: {processed_count}, skipped: {skipped_count}")
        return output_jsonl

    def prepare_all_splits(
        self,
        splits: Optional[List[str]] = None,
        max_samples_per_split: Optional[int] = None,
    ) -> List[str]:
        splits = splits or ['train', 'validation']
        output_files = []
        for split in splits:
            ms = max_samples_per_split or self.max_samples or self.DEFAULT_MAX_SAMPLES.get(split)
            output_files.append(self.prepare_split(split, ms))
        print(f"Hausa preparation complete: {output_files}")
        return output_files

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

        splits = {}
        for split_name in ['train', 'validation']:
            jsonl_path = os.path.join(self.output_dir, f"hausa_{split_name}.jsonl")
            if os.path.exists(jsonl_path):
                data = [json.loads(line) for line in open(jsonl_path, 'r', encoding='utf-8') if line.strip()]
                splits[split_name] = Dataset.from_list(data)
                print(f"  Loaded {len(data)} samples for hausa_speaker/{split_name}")

        if not splits:
            raise ValueError("No Hausa JSONL files found. Run prepare_all_splits() first.")
        dataset_dict = DatasetDict(splits)
        dataset_dict.push_to_hub(repo_id=repo_id, config_name="hausa_speaker", token=token, private=private)
        print(f"✓ Hausa dataset uploaded to {repo_id} (config: hausa_speaker)")
        return f"https://huggingface.co/datasets/{repo_id}"

    def _extract_audio(self, audio_data: Any) -> Tuple[np.ndarray, int]:
        if isinstance(audio_data, dict) and "array" in audio_data:
            audio_array, audio_sr = audio_data["array"], audio_data["sampling_rate"]
        elif isinstance(audio_data, dict) and "bytes" in audio_data:
            with io.BytesIO(audio_data["bytes"]) as f:
                audio_array, audio_sr = sf.read(f, dtype="float32")
        else:
            audio_array, audio_sr = audio_data, 22000
        if isinstance(audio_array, (list, tuple)):
            audio_array = np.array(audio_array, dtype=np.float32)
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=-1)
        return audio_array, audio_sr


class EnglishSpeakerDatasetPreparer:
    """
    Prepare English speaker dataset from HuggingFace to JSONL.
    Dataset: benjaminogbonna/nigerian_common_voice_dataset (english). Output: audio (base64), text.
    """

    DEFAULT_MAX_SAMPLES = {"train": MAX_TRAIN_SAMPLES, "validation": MAX_VAL_SAMPLES}

    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        ref_audio_path: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.dataset_name = "benjaminogbonna/nigerian_common_voice_dataset"
        self.subset = "english"
        self.ref_audio_path = ref_audio_path or os.path.join(os.path.dirname(__file__), "voices", "english_speaker.wav")
        print(f"English Dataset Preparer: {self.dataset_name} ({self.subset}) -> {output_dir}")

    def prepare_split(self, split: str, max_samples: Optional[int] = None) -> str:
        max_samples = max_samples or self.max_samples
        print(f"\nPreparing English {split}...")
        if split == "train":
            train_ds = load_dataset(self.dataset_name, name=self.subset, split="train", keep_in_memory=False)
            test_ds = load_dataset(self.dataset_name, name=self.subset, split="test", keep_in_memory=False)
            hf_dataset = train_ds.concatenate(test_ds)
        else:
            hf_dataset = load_dataset(self.dataset_name, name=self.subset, split="validation", keep_in_memory=False)
        if max_samples is not None:
            hf_dataset = hf_dataset.select(range(min(max_samples, len(hf_dataset))))

        os.makedirs(self.output_dir, exist_ok=True)
        output_jsonl = os.path.join(self.output_dir, f"english_{split}.jsonl")
        processed_count = 0
        skipped_count = 0

        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(tqdm(hf_dataset, desc=f"English {split}")):
                try:
                    audio_array, audio_sr = self._extract_audio(item["audio"])
                    if audio_sr != 24000:
                        import librosa
                        audio_array = librosa.resample(audio_array, orig_sr=audio_sr, target_sr=24000)
                        audio_sr = 24000
                    audio_b64 = audio_to_base64(audio_array, audio_sr)
                    text = item.get("sentence", item.get("text", ""))
                    data_entry = {"audio": audio_b64, "text": text}
                    f.write(json.dumps(data_entry, ensure_ascii=False) + '\n')
                    processed_count += 1
                except Exception as e:
                    if any(err in str(e) for err in ["End of file", "Failed to open", "Invalid data", "Could not receive frame"]):
                        skipped_count += 1
                        continue
                    raise

        print(f"  Output: {output_jsonl}, processed: {processed_count}, skipped: {skipped_count}")
        return output_jsonl

    def prepare_all_splits(
        self,
        splits: Optional[List[str]] = None,
        max_samples_per_split: Optional[int] = None,
    ) -> List[str]:
        splits = splits or ['train', 'validation']
        output_files = []
        for split in splits:
            ms = max_samples_per_split or self.max_samples or self.DEFAULT_MAX_SAMPLES.get(split)
            output_files.append(self.prepare_split(split, ms))
        print(f"English preparation complete: {output_files}")
        return output_files

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

        splits = {}
        for split_name in ['train', 'validation']:
            jsonl_path = os.path.join(self.output_dir, f"english_{split_name}.jsonl")
            if os.path.exists(jsonl_path):
                data = [json.loads(line) for line in open(jsonl_path, 'r', encoding='utf-8') if line.strip()]
                splits[split_name] = Dataset.from_list(data)
                print(f"  Loaded {len(data)} samples for english_speaker/{split_name}")

        if not splits:
            raise ValueError("No English JSONL files found. Run prepare_all_splits() first.")
        dataset_dict = DatasetDict(splits)
        dataset_dict.push_to_hub(repo_id=repo_id, config_name="english_speaker", token=token, private=private)
        print(f"✓ English dataset uploaded to {repo_id} (config: english_speaker)")
        return f"https://huggingface.co/datasets/{repo_id}"

    def _extract_audio(self, audio_data: Any) -> Tuple[np.ndarray, int]:
        if isinstance(audio_data, dict) and "array" in audio_data:
            audio_array, audio_sr = audio_data["array"], audio_data["sampling_rate"]
        elif isinstance(audio_data, dict) and "bytes" in audio_data:
            with io.BytesIO(audio_data["bytes"]) as f:
                audio_array, audio_sr = sf.read(f, dtype="float32")
        else:
            audio_array, audio_sr = audio_data, 22000
        if isinstance(audio_array, (list, tuple)):
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
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--repo_id", type=str, default=HF_DATASET_REPO or "", help="HuggingFace dataset repo ID")
    args = parser.parse_args()

    if not args.upload or not args.repo_id:
        print("⚠ Upload not requested or repo_id missing. Preparing local files only.")

    if args.speaker in ["hausa", "both"]:
        print("\n" + "="*60 + "\nPreparing Hausa Dataset\n" + "="*60)
        preparer = HausaSpeakerDatasetPreparer(output_dir=args.output_dir, max_samples=args.max_samples)
        preparer.prepare_all_splits()
        if args.upload and args.repo_id:
            preparer.upload_to_huggingface(args.repo_id)

    if args.speaker in ["english", "both"]:
        print("\n" + "="*60 + "\nPreparing English Dataset\n" + "="*60)
        preparer = EnglishSpeakerDatasetPreparer(output_dir=args.output_dir, max_samples=args.max_samples)
        preparer.prepare_all_splits()
        if args.upload and args.repo_id:
            preparer.upload_to_huggingface(args.repo_id)

    print("\nDone!")


if __name__ == "__main__":
    main()
