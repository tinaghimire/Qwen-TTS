#!/usr/bin/env python3
"""Test the background preprocessor."""

import torch
from transformers import AutoConfig
from background_preprocessor import get_cached_dataloader

print("="*60)
print("Testing Background Preprocessor")
print("="*60)

# Load config (normally from model)
# For testing, we'll create a minimal config
class MockConfig:
    def __init__(self):
        self._name_or_path = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        self.tts_pad_token_id = 151645
        self.tts_bos_token_id = 151644
        self.tts_eos_token_id = 151645
        self.talker_config = type('obj', (object,), {
            'codec_nothink_id': 151653,
            'codec_think_bos_id': 151648,
            'codec_think_eos_id': 151649,
            'codec_pad_id': 151653,
            'codec_bos_id': 151650,
            'codec_eos_token_id': 151652
        })()

config = MockConfig()

# Load processor
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(config._name_or_path)

print("\n1. Starting background preprocessor...")
preprocessor, dataloader = get_cached_dataloader(
    dataset_name="vaghawan/hausa-tts-22k",
    split="train",
    tokenizer_path="Qwen/Qwen3-TTS-Tokenizer-12Hz",
    processor=processor,
    config=config,
    cache_dir="./test_cache",
    max_samples=50,  # Test with small dataset
    batch_size=4,
    num_preprocessing_workers=2,
)

print(f"\n2. Preprocessor started, processed: {preprocessor.get_processed_count()} samples")

print("\n3. Testing DataLoader iteration...")
for i, batch in enumerate(dataloader):
    print(f"   Batch {i}: {batch['input_ids'].shape}")
    print(f"   Processed: {preprocessor.get_processed_count()} samples")
    print(f"   Loss sample: {batch.get('codec_0_labels', 'N/A')[:2]}")
    if i >= 2:  # Test 3 batches
        break

print("\n4. Stopping preprocessor...")
preprocessor.stop()

print("\n✓ Test completed successfully!")
print("="*60)