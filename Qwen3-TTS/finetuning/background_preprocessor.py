#!/usr/bin/env python3
# coding=utf-8
"""
Background Data Preprocessor for Qwen3-TTS Training.

Pre-processes data in parallel workers during training to eliminate bottlenecks.
Uses a producer-consumer pattern where workers continuously fill a cache
that the training dataloader reads from.

Usage:
    # Start before training
    preprocessor = BackgroundPreprocessor(...)
    preprocessor.start()

    # Create dataloader that reads from cache
    dataloader = CachedDataLoader(preprocessor.cache_file, ...)

    # Train
    preprocessor.wait_for_data(min_items)  # Wait for initial batch
    for batch in dataloader:
        train(batch)
        # Workers keep running in background, filling cache
"""

import json
import os
import shelve
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty as QueueEmpty
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm


class BackgroundPreprocessor:
    """
    Continuously processes data in the background during training.

    Workers process raw audio from HuggingFace and save encoded codes
    to a shared cache file that the training dataloader reads from.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        tokenizer_path: str,
        cache_file: str,
        max_samples: Optional[int] = None,
        num_workers: int = 4,
        batch_size: int = 8,
        device: str = "cuda",
        prefetch_factor: int = 4,  # Process this many batches ahead
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer_path = tokenizer_path
        self.cache_file = cache_file
        self.max_samples = max_samples
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.device = device
        self.prefetch_factor = prefetch_factor

        self.processing = False
        self.processed_count = 0
        self.error = None

        # Create cache directory
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    def start(self):
        """Start background workers."""
        print(f"\n{'='*60}")
        print("🚀 Starting Background Data Preprocessing")
        print(f"{'='*60}")
        print(f"  Dataset: {self.dataset_name}")
        print(f"  Split: {self.split}")
        print(f"  Workers: {self.num_workers}")
        print(f"  Cache: {self.cache_file}")
        print(f"  Max samples: {self.max_samples or 'All'}")
        print(f"{'='*60}")

        # Start processing thread
        self.processing = True
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

        print("✓ Background preprocessing started")
        print("  Data will be processed in parallel while training")
        print(f"  Maintaining buffer of {self.prefetch_factor} batches ahead\n")

    def _process_loop(self):
        """Main processing loop running in background thread."""
        try:
            # Load dataset
            print(f"[Preprocessor] Loading dataset from HuggingFace...")
            hf_dataset = load_dataset(self.dataset_name, split=self.split)
            if self.max_samples:
                hf_dataset = hf_dataset.select(range(min(self.max_samples, len(hf_dataset))))
            total_samples = len(hf_dataset)

            # Initialize tokenizer (load once per worker)
            from qwen_tts import Qwen3TTSTokenizer
            print(f"[Preprocessor] Loading tokenizer on {self.device}...")
            tokenizer = Qwen3TTSTokenizer.from_pretrained(
                self.tokenizer_path,
                device_map=self.device,
            )

            # Process in batches
            with torch.no_grad():
                batch_audio = []
                batch_texts = []
                batch_indices = []

                for idx, item in enumerate(tqdm(hf_dataset, desc="[Preprocessor]", unit="samples")):
                    if not self.processing:
                        print(f"[Preprocessor] Stopping processing (processed={self.processed_count})")
                        break

                    try:
                        # Extract audio
                        audio_data = item.get("audio")
                        audio_array, audio_sr = self._extract_audio(audio_data)

                        # Resample to 24kHz
                        if audio_sr != 24000:
                            audio_array = self._resample_audio(audio_array, audio_sr, 24000)

                        batch_audio.append(audio_array)
                        batch_texts.append(item.get("text", ""))
                        batch_indices.append(idx)

                        # Process batch when full
                        if len(batch_audio) >= self.batch_size or idx == total_samples - 1:
                            # Encode to codes
                            enc_result = tokenizer.encode(batch_audio, sr=24000)

                            # Write to cache
                            for i, (codes, text, orig_idx) in enumerate(
                                zip(enc_result.audio_codes, batch_texts, batch_indices)
                            ):
                                data_entry = {
                                    "text": text,
                                    "audio_codes": codes.cpu().numpy().tolist(),
                                    "sample_idx": orig_idx,
                                }
                                self._write_to_cache(orig_idx, data_entry)
                                self.processed_count += 1

                                # Print progress every batch
                                if i == 0:
                                    print(f"[Preprocessor] Processed {self.processed_count}/{total_samples} samples")

                            # Clear batch
                            batch_audio = []
                            batch_texts = []
                            batch_indices = []

                    except Exception as e:
                        print(f"[Preprocessor] Error processing sample {idx}: {e}")
                        continue

            print(f"[Preprocessor] Finished processing {self.processed_count} samples")

        except Exception as e:
            print(f"[Preprocessor] Fatal error: {e}")
            import traceback
            traceback.print_exc()
            self.error = e
        finally:
            self.processing = False

    def _write_to_cache(self, idx: int, data: dict):
        """Write data to cache file."""
        with shelve.open(self.cache_file, writeback=False) as cache:
            cache[str(idx)] = data
            cache["processed_count"] = self.processed_count
            cache["last_update"] = time.time()

    def _extract_audio(self, audio_data) -> Tuple[np.ndarray, int]:
        """Extract audio array and sampling rate."""
        if isinstance(audio_data, dict):
            audio_array = audio_data["array"]
            audio_sr = audio_data["sampling_rate"]
        elif hasattr(audio_data, 'get_all_samples'):
            decoded = audio_data.get_all_samples()
            if hasattr(decoded, 'data'):
                audio_array = decoded.data.numpy()
            else:
                audio_array = torch.tensor(decoded).numpy()
            audio_sr = audio_data.metadata.sample_rate
        elif hasattr(audio_data, 'array') and hasattr(audio_data, 'sampling_rate'):
            audio_array = audio_data.array
            audio_sr = audio_data.sampling_rate
        else:
            audio_array = audio_data
            audio_sr = 22000

        if isinstance(audio_array, (list, tuple)):
            audio_array = np.array(audio_array, dtype=float)
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()
        elif isinstance(audio_array, np.ndarray):
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
        else:
            audio_array = np.array([audio_array], dtype=np.float32)

        return audio_array, audio_sr

    def _resample_audio(self, audio_array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio."""
        import librosa
        return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)

    def get_processed_count(self) -> int:
        """Get number of samples processed so far."""
        try:
            with shelve.open(self.cache_file) as cache:
                return cache.get("processed_count", 0)
        except:
            return self.processed_count

    def is_available(self, idx: int) -> bool:
        """Check if a sample is available in cache."""
        try:
            with shelve.open(self.cache_file, writeback=False) as cache:
                return str(idx) in cache
        except:
            return False

    def wait_for_data(self, min_items: int = 10, timeout: float = 60.0):
        """Wait until at least min_items are processed."""
        print(f"[Preprocessor] Waiting for {min_items} samples to be processed...")
        start = time.time()
        while self.get_processed_count() < min_items:
            time.sleep(0.5)
            if time.time() - start > timeout:
                print(f"[Preprocessor] Timeout waiting for data (processed={self.get_processed_count()})")
                break
        print(f"[Preprocessor] ✓ {self.get_processed_count()} samples ready for training")

    def stop(self):
        """Stop background processing."""
        print(f"\n[Preprocessor] Stopping background processing...")
        self.processing = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=5)
        print(f"[Preprocessor] Stopped (processed={self.processed_count} samples)")


class CachedDataLoader:
    """
    DataLoader that reads from continuously growing cache.

    This works with BackgroundPreprocessor to train on data
    as soon as it's processed, eliminating preprocessing delays.
    """

    def __init__(
        self,
        cache_file: str,
        processor,
        config,
        max_samples: Optional[int] = None,
        check_interval: float = 0.1,  # How often to check for new data
        ref_audio_path: Optional[str] = None,
    ):
        self.cache_file = cache_file
        self.processor = processor
        self.config = config
        self.max_samples = max_samples
        self.check_interval = check_interval
        self.ref_audio_path = ref_audio_path

        # Set default reference audio
        if self.ref_audio_path is None:
            self.ref_audio_path = os.path.join(
                os.path.dirname(__file__),
                "voices", "english_voice", "english_voice_24k.wav"
            )

        import librosa
        # Load reference audio once
        ref_audio, ref_sr = librosa.load(self.ref_audio_path, sr=None, mono=True)
        if ref_sr != 24000:
            ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=24000)
        self.ref_audio = ref_audio

    def __len__(self):
        """Get total expected samples (may increase as processing continues)."""
        try:
            with shelve.open(self.cache_file, writeback=False) as cache:
                return cache.get("processed_count", 0)
        except:
            return 0

    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from cache, waiting if not yet available."""
        timeout = 30.0  # Wait up to 30 seconds for a sample
        start = time.time()

        while not self.is_available(idx):
            time.sleep(self.check_interval)
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for sample {idx}")

        return self._process_item(self._read_from_cache(idx))

    def is_available(self, idx: int) -> bool:
        """Check if sample is available."""
        try:
            with shelve.open(self.cache_file, writeback=False) as cache:
                return str(idx) in cache
        except:
            return False

    def _read_from_cache(self, idx: int) -> Dict:
        """Read raw data from cache."""
        with shelve.open(self.cache_file, writeback=False) as cache:
            return cache[str(idx)]

    def _process_item(self, data: Dict) -> Dict:
        """Process cached item into training format."""
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

        # Build assistant text
        text = f"<|im_start|>assistant\n{data['text']}<|im_end|>\n<|im_start|>assistant\n"

        # Tokenize text
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id

        # Convert audio codes to tensor
        audio_codes = torch.tensor(data['audio_codes'], dtype=torch.long)

        # Extract reference mel spectrogram
        with torch.inference_mode():
            import librosa
            ref_mel = mel_spectrogram(
                torch.from_numpy(self.ref_audio).unsqueeze(0),
                n_fft=1024,
                num_mels=128,
                sampling_rate=24000,
                hop_size=256,
                win_size=1024,
                fmin=0,
                fmax=12000
            ).transpose(1, 2)

        return {
            "text_ids": input_id[:, :-5],
            "audio_codes": audio_codes,
            "ref_mel": ref_mel,
        }

    def collate_fn(self, batch):
        """Collate function to pad and batch variable-length sequences."""
        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
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
            text_ids = data['text_ids']
            audio_codec_0 = data['audio_codes'][:, 0]
            audio_codecs = data['audio_codes']

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            # text channel
            input_ids[i, :3, 0] = text_ids[0, :3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i, 7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8 + text_ids_len - 3, 0] = text_ids[0, 3:]
            input_ids[i, 8 + text_ids_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8 + text_ids_len - 2:8 + text_ids_len + codec_ids_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, :8 + text_ids_len + codec_ids_len] = True

            # codec channel
            input_ids[i, 3:8, 1] = torch.tensor([
                self.config.talker_config.codec_nothink_id,
                self.config.talker_config.codec_think_bos_id,
                self.config.talker_config.codec_think_eos_id,
                0,
                self.config.talker_config.codec_pad_id
            ])
            input_ids[i, 8:8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8 + text_ids_len - 2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8 + text_ids_len - 1 + codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8 + text_ids_len - 1 + codec_ids_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len, :] = audio_codecs

            codec_embedding_mask[i, 3:8 + text_ids_len + codec_ids_len] = True
            codec_embedding_mask[i, 6] = False

            codec_mask[i, 8 + text_ids_len - 1:8 + text_ids_len - 1 + codec_ids_len] = True
            attention_mask[i, :8 + text_ids_len + codec_ids_len] = True

        ref_mels = [data['ref_mel'] for data in batch]
        ref_mels = torch.cat(ref_mels, dim=0)

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


def get_cached_dataloader(
    dataset_name: str,
    split: str,
    tokenizer_path: str,
    processor,
    config,
    cache_dir: str = "./cache",
    max_samples: Optional[int] = None,
    batch_size: int = 4,
    num_workers: int = 0,
    num_preprocessing_workers: int = 4,
) -> Tuple[BackgroundPreprocessor, CachedDataLoader]:
    """
    Create a background preprocessor and cached dataloader.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split
        tokenizer_path: Tokenizer path
        processor: Model processor
        config: Model configuration
        cache_dir: Cache directory
        max_samples: Maximum samples to process
        batch_size: Training batch size
        num_workers: DataLoader workers (keep 0 for this dataloader)
        num_preprocessing_workers: Number of background preprocessing workers

    Returns:
        Tuple of (preprocessor, dataloader)
    """
    cache_file = os.path.join(cache_dir, f"{split}_cache.db")

    # Start background preprocessor
    preprocessor = BackgroundPreprocessor(
        dataset_name=dataset_name,
        split=split,
        tokenizer_path=tokenizer_path,
        cache_file=cache_file,
        max_samples=max_samples,
        num_workers=num_preprocessing_workers,
        batch_size=batch_size * 2,  # Process larger batches
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    preprocessor.start()

    # Create cached dataloader
    dataset = CachedDataLoader(
        cache_file=cache_file,
        processor=processor,
        config=config,
        max_samples=max_samples,
    )

    # Wait for initial data
    initial_batch = batch_size * 2  # Wait for 2 batches
    preprocessor.wait_for_data(min_items=initial_batch)

    # Create PyTorch DataLoader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep order for consistency
        collate_fn=dataset.collate_fn,
        num_workers=0,  # No multiprocessing - data already processed
        pin_memory=True,
    )

    return preprocessor, dataloader