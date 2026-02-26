#!/usr/bin/env python3
"""
Plot training and validation metrics from training_log.jsonl and validation_log.jsonl.
Charts loss, main_loss, and sub_talker_loss separately.

Usage:
  python -m finetuning.plot_logs [--output_dir ./output] [--out charts.png]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_jsonl(path: str) -> list[dict]:
    out = []
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def main():
    p = argparse.ArgumentParser(description="Plot loss, main_loss, sub_talker_loss from training/validation logs")
    p.add_argument("--output_dir", default="./output", help="Directory containing training_log.jsonl and validation_log.jsonl")
    p.add_argument("--out", default=None, help="Output image path (default: output_dir/training_charts.png)")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    out_path = args.out or str(output_dir / "training_charts.png")

    train_path = output_dir / "training_log.jsonl"
    val_path = output_dir / "validation_log.jsonl"

    train_logs = load_jsonl(str(train_path))
    val_logs = load_jsonl(str(val_path))

    if not train_logs and not val_logs:
        print(f"No logs found in {output_dir}. Check training_log.jsonl and validation_log.jsonl.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- Training: loss, main_loss, sub_talker_loss ---
    ax = axes[0, 0]
    if train_logs:
        steps = [r["step"] for r in train_logs]
        ax.plot(steps, [r["loss"] for r in train_logs], label="loss", color="C0")
        if any(r.get("main_loss") is not None for r in train_logs):
            ax.plot(steps, [r.get("main_loss") for r in train_logs], label="main_loss", color="C1")
        if any(r.get("sub_talker_loss") is not None for r in train_logs):
            ax.plot(steps, [r.get("sub_talker_loss") for r in train_logs], label="sub_talker_loss", color="C2")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training: loss, main_loss, sub_talker_loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Training: learning_rate and grad_norm ---
    ax = axes[0, 1]
    if train_logs:
        steps = [r["step"] for r in train_logs]
        if any(r.get("learning_rate") is not None for r in train_logs):
            ax.plot(steps, [r.get("learning_rate") for r in train_logs], label="learning_rate", color="C0")
        if any(r.get("grad_norm") is not None for r in train_logs):
            ax.plot(steps, [r.get("grad_norm") for r in train_logs], label="grad_norm", color="C1")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title("Training: learning_rate, grad_norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Validation: loss, main_loss, sub_talker_loss ---
    ax = axes[1, 0]
    if val_logs:
        steps = [r["step"] for r in val_logs]
        ax.plot(steps, [r["loss"] for r in val_logs], label="loss", color="C0")
        if any(r.get("main_loss") is not None for r in val_logs):
            ax.plot(steps, [r.get("main_loss") for r in val_logs], label="main_loss", color="C1")
        if any(r.get("sub_talker_loss") is not None for r in val_logs):
            ax.plot(steps, [r.get("sub_talker_loss") for r in val_logs], label="sub_talker_loss", color="C2")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Validation: loss, main_loss, sub_talker_loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Validation: perplexity and speaker_embedding_consistency ---
    ax = axes[1, 1]
    if val_logs:
        steps = [r["step"] for r in val_logs]
        if any((r.get("metrics") or {}).get("perplexity") is not None for r in val_logs):
            ax.plot(steps, [float((r.get("metrics") or {}).get("perplexity", 0)) for r in val_logs], label="perplexity", color="C0")
        if any((r.get("metrics") or {}).get("speaker_embedding_consistency") is not None for r in val_logs):
            ax.plot(steps, [(r.get("metrics") or {}).get("speaker_embedding_consistency") for r in val_logs], label="speaker_embedding_consistency", color="C1")
        ax.set_title("Validation: perplexity, speaker_embedding_consistency")
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    print(f"Saved charts to {out_path}")


if __name__ == "__main__":
    main()
