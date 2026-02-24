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
"""
Layer replacement utility for Qwen3TTSTalker model.
Provides functions to replace and add new layers to the model for fine-tuning.
"""

import os
import sys

# Add Qwen3-TTS directory to Python path to find qwen_tts module
script_dir = os.path.dirname(os.path.abspath(__file__))
qwen3_tts_dir = os.path.dirname(script_dir)
if qwen3_tts_dir not in sys.path:
    sys.path.insert(0, qwen3_tts_dir)

import torch
import torch.nn as nn
from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSTalkerDecoderLayer


def initialize_decoder_layer(layer: Qwen3TTSTalkerDecoderLayer):
    """
    Initialize a decoder layer with fresh weights using Xavier/Glorot initialization.

    Args:
        layer: Qwen3TTSTalkerDecoderLayer instance to initialize
    """
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Normal initialization for embeddings
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    layer.apply(_init_weights)


def replace_and_add_layers(
    model,
    replace_last_n: int = 2,
    add_new_layers: int = 4,
    freeze_original_layers: bool = True,
    verbose: bool = True
):
    """
    Replace the last N layers of Qwen3TTSTalker with newly initialized layers
    and add M additional layers.

    Args:
        model: Qwen3TTSForConditionalGeneration model
        replace_last_n: Number of last layers to replace (default: 2)
        add_new_layers: Number of new layers to add (default: 4)
        freeze_original_layers: Whether to freeze the original layers (default: True)
        verbose: Whether to print detailed information (default: True)

    Returns:
        Modified model with replaced and added layers
    """
    # Access the talker model's layers
    talker_model = model.talker.model
    original_layers = talker_model.layers
    original_num_layers = len(original_layers)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Layer Replacement and Addition")
        print(f"{'='*60}")
        print(f"Original number of layers: {original_num_layers}")
        print(f"Replacing last {replace_last_n} layers")
        print(f"Adding {add_new_layers} new layers")
        print(f"Total new layers: {original_num_layers - replace_last_n + replace_last_n + add_new_layers}")

    # Validate inputs
    if replace_last_n >= original_num_layers:
        raise ValueError(
            f"Cannot replace {replace_last_n} layers from a model with {original_num_layers} layers. "
            f"replace_last_n must be less than original_num_layers."
        )

    # Keep the first (original_num_layers - replace_last_n) layers
    num_keep = original_num_layers - replace_last_n
    kept_layers = list(original_layers[:num_keep])

    if verbose:
        print(f"\nKeeping layers 1-{num_keep} (frozen: {freeze_original_layers})")

    # Freeze the kept layers if requested
    if freeze_original_layers:
        for i, layer in enumerate(kept_layers):
            for param in layer.parameters():
                param.requires_grad = False
            if verbose:
                print(f"  - Layer {i+1}: Frozen")

    # Create new layers (replacements + additions)
    total_new_layers = replace_last_n + add_new_layers
    config = talker_model.config

    new_layers = []
    for i in range(total_new_layers):
        layer_idx = num_keep + i
        new_layer = Qwen3TTSTalkerDecoderLayer(config, layer_idx)
        initialize_decoder_layer(new_layer)
        new_layers.append(new_layer)

        if verbose:
            layer_type = "Replacement" if i < replace_last_n else "Additional"
            print(f"  - Layer {layer_idx+1}: {layer_type} (freshly initialized)")

    # Combine kept layers with new layers
    all_layers = kept_layers + new_layers

    # Replace the model's layers
    talker_model.layers = nn.ModuleList(all_layers)

    # Update the configuration
    config.num_hidden_layers = len(all_layers)
    model.config.talker_config.num_hidden_layers = len(all_layers)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Layer replacement complete!")
        print(f"New total number of layers: {len(all_layers)}")
        print(f"Configuration updated: num_hidden_layers = {config.num_hidden_layers}")
        print(f"{'='*60}\n")

    return model


def get_trainable_params(model):
    """
    Get the number of trainable and frozen parameters in the model.

    Args:
        model: Qwen3TTSForConditionalGeneration model

    Returns:
        Tuple of (trainable_params, frozen_params, total_params)
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen
    return trainable, frozen, total


def print_model_summary(model):
    """
    Print a summary of the model including layer information and parameter counts.

    Args:
        model: Qwen3TTSForConditionalGeneration model
    """
    print(f"\n{'='*60}")
    print(f"Model Summary")
    print(f"{'='*60}")

    talker_model = model.talker.model
    num_layers = len(talker_model.layers)

    print(f"Total layers: {num_layers}")
    print(f"Hidden size: {talker_model.config.hidden_size}")
    print(f"Intermediate size: {talker_model.config.intermediate_size}")
    print(f"Attention heads: {talker_model.config.num_attention_heads}")

    trainable, frozen, total = get_trainable_params(model)

    print(f"\nParameter counts:")
    print(f"  Trainable: {trainable:,} ({100 * trainable / total:.2f}%)")
    print(f"  Frozen: {frozen:,} ({100 * frozen / total:.2f}%)")
    print(f"  Total: {total:,}")

    # Count trainable layers
    trainable_layers = sum(1 for layer in talker_model.layers if any(p.requires_grad for p in layer.parameters()))
    frozen_layers = num_layers - trainable_layers

    print(f"\nLayer status:")
    print(f"  Trainable layers: {trainable_layers}")
    print(f"  Frozen layers: {frozen_layers}")
    print(f"{'='*60}\n")
