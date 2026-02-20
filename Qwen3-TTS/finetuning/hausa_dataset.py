# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
from datasets import load_dataset
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]


class HausaTTSDataset(Dataset):
    """
    Dataset for Hausa TTS fine-tuning using data from Hugging Face.
    Loads data from vaghawan/hausa-tts-22k and prepares it for Qwen3-TTS training.
    """
    
    def __init__(
        self,
        split: str = "train",
        processor=None,
        config: Qwen3TTSConfig = None,
        ref_audio_path: str = None,
        ref_text: str = None,
        max_samples: int = None,
        lag_num: int = -1
    ):
        """
        Initialize Hausa TTS Dataset.
        
        Args:
            split: Dataset split - "train", "validation", or "test"
            processor: Qwen3TTS processor for tokenization
            config: Qwen3TTS configuration
            ref_audio_path: Path to reference audio file
            ref_text: Reference text for voice cloning
            max_samples: Maximum number of samples to use (for debugging)
            lag_num: Lag number for training
        """
        self.processor = processor
        self.config = config
        self.lag_num = lag_num
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text or "MTN Entertainment and Lifestyle. Entertainment and Lifestyle are at the heart of MTN's offering. We bring you music, movies, games and more through our digital platforms. With MTN musicals, you can stream your favorite"
        
        # Load dataset from Hugging Face
        print(f"Loading Hausa TTS dataset ({split} split)...")
        self.hf_dataset = load_dataset("vaghawan/hausa-tts-22k", split=split)
        
        # Limit samples if specified
        if max_samples is not None:
            self.hf_dataset = self.hf_dataset.select(range(min(max_samples, len(self.hf_dataset))))
        
        print(f"Loaded {len(self.hf_dataset)} samples")
        
        # Prepare data list
        self.data_list = self._prepare_data_list()
        
    def _prepare_data_list(self) -> List[dict]:
        """Prepare data list from Hugging Face dataset."""
        data_list = []
        
        for idx, item in enumerate(self.hf_dataset):
            # Extract audio array and sampling rate
            audio_data = item["audio"]
            if isinstance(audio_data, dict):
                audio_array = audio_data["array"]
                audio_sr = audio_data["sampling_rate"]
            else:
                audio_array = audio_data
                audio_sr = 22000  # Default for this dataset
            
            # Save audio to temporary file or use array directly
            # For now, we'll save the array and sr for processing
            data_list.append({
                "audio_array": audio_array.astype(np.float32),
                "audio_sr": audio_sr,
                "text": item["text"],
                "speaker_id": item.get("speaker_id", "unknown"),
                "language": item.get("language", "ha"),
                "gender": item.get("gender", "unknown"),
                "age_range": item.get("age_range", "unknown"),
                "phase": item.get("phase", "unknown"),
                "ref_audio": self.ref_audio_path,
                "ref_text": self.ref_text
            })
        
        return data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        """Load audio from file path."""
        audio, sr = librosa.load(x, sr=None, mono=True)
        
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        
        return audio.astype(np.float32), int(sr)
    
    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]
        
        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out
    
    def _build_assistant_text(self, text: str) -> str:
        """Build assistant text with special tokens."""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    def _ensure_list(self, x: MaybeList) -> List[Any]:
        """Ensure input is a list."""
        return x if isinstance(x, list) else [x]
    
    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        """Tokenize text using processor."""
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id
    
    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        """Extract mel spectrogram from audio."""
        # Resample to 24kHz if needed
        if sr != 24000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=24000)
            sr = 24000
        
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
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        item = self.data_list[idx]
        
        audio_array = item["audio_array"]
        audio_sr = item["audio_sr"]
        text = item["text"]
        audio_codes = item["audio_codes"]
        language = item.get('language', 'ha')
        ref_audio_path = item['ref_audio']
        
        # Build assistant text
        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)
        
        # Convert audio codes to tensor
        audio_codes = torch.tensor(audio_codes, dtype=torch.long)
        
        # Load and process reference audio
        ref_audio_list = self._ensure_list(ref_audio_path)
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav, sr = normalized[0]
        
        ref_mel = self.extract_mels(audio=wav, sr=sr)
        
        return {
            "text_ids": text_ids[:, :-5],    # 1, t
            "audio_codes": audio_codes,      # t, 16
            "ref_mel": ref_mel
        }
    
    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        assert self.lag_num == -1
        
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
            input_ids[i, 8:8+text_ids_len-3, 0] = text_ids[0, 3:]
            input_ids[i, 8+text_ids_len-3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8+text_ids_len-2:8+text_ids_len+codec_ids_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, :8+text_ids_len+codec_ids_len] = True
            
            # codec channel
            input_ids[i, 3:8, 1] = torch.tensor(
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,     # for speaker embedding
                    self.config.talker_config.codec_pad_id       
                ]
            )
            input_ids[i, 8:8+text_ids_len-3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8+text_ids_len-3, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, 8+text_ids_len-2, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len, 1] = audio_codec_0
            input_ids[i, 8+text_ids_len-1+codec_ids_len, 1] = self.config.talker_config.codec_eos_token_id
            
            codec_0_labels[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = audio_codec_0
            codec_0_labels[i, 8+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id
            
            codec_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len, :] = audio_codecs
            
            codec_embedding_mask[i, 3:8+text_ids_len+codec_ids_len] = True
            codec_embedding_mask[i, 6] = False       # for speaker embedding
            
            codec_mask[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = True
            attention_mask[i, :8+text_ids_len+codec_ids_len] = True
        
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
    
    def set_audio_codes(self, audio_codes_list: List[np.ndarray]):
        """
        Set pre-computed audio codes for the dataset.
        
        Args:
            audio_codes_list: List of audio codes arrays, one per sample
        """
        assert len(audio_codes_list) == len(self.data_list), \
            f"Number of audio codes ({len(audio_codes_list)}) must match number of samples ({len(self.data_list)})"
        
        for i, codes in enumerate(audio_codes_list):
            self.data_list[i]["audio_codes"] = codes
        
        print(f"Set audio codes for {len(audio_codes_list)} samples")
