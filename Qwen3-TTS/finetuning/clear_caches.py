#!/usr/bin/env python3
# coding=utf-8
"""
Clear GPU and caches on the server.

Usage:
  python clear_caches.py          # Kill GPU processes + clear PyTorch CUDA cache
  python clear_caches.py --all    # Also clear project cache dir (Qwen3-TTS/cache)
  python clear_caches.py --kill-only   # Only kill GPU processes (no Python cache clear)
"""
import argparse
import os
import subprocess
import sys

def kill_gpu_processes():
    """Kill all processes using the GPU (nvidia-smi compute apps)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if out.returncode != 0:
            return
        pids = [x.strip() for x in out.stdout.strip().splitlines() if x.strip() and x.strip().isdigit()]
        for pid in pids:
            try:
                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=2)
                print(f"  Killed GPU process {pid}")
            except Exception:
                pass
        if not pids:
            print("  No GPU processes to kill")
    except FileNotFoundError:
        print("  nvidia-smi not found (skip killing)")
    except Exception as e:
        print(f"  Error killing GPU processes: {e}")


def clear_pytorch_cuda():
    """Clear PyTorch CUDA cache in this process."""
    try:
        import torch
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
                torch.cuda.reset_accumulated_memory_stats()
            print("  PyTorch CUDA: cache cleared, stats reset")
            for i in range(torch.cuda.device_count()):
                a = torch.cuda.memory_allocated(i) / 1e9
                r = torch.cuda.memory_reserved(i) / 1e9
                print(f"    GPU {i}: allocated={a:.2f} GiB, reserved={r:.2f} GiB")
        else:
            print("  CUDA not available (skip PyTorch clear)")
    except ImportError:
        print("  PyTorch not available (skip PyTorch clear)")


def clear_project_cache(cache_dir: str):
    """Remove project cache directory (e.g. downloaded HF datasets)."""
    if not cache_dir or not os.path.isdir(cache_dir):
        print(f"  Project cache dir not found: {cache_dir}")
        return
    try:
        import shutil
        shutil.rmtree(cache_dir)
        print(f"  Removed project cache: {cache_dir}")
    except Exception as e:
        print(f"  Error removing project cache: {e}")


def main():
    ap = argparse.ArgumentParser(description="Clear GPU and caches")
    ap.add_argument("--all", action="store_true", help="Also clear project cache dir (finetuning/cache or CACHE_DIR)")
    ap.add_argument("--kill-only", action="store_true", help="Only kill GPU processes, do not run PyTorch clear")
    args = ap.parse_args()

    print("=" * 60)
    print("Clear GPU and caches")
    print("=" * 60)

    print("\n1. Killing GPU processes...")
    kill_gpu_processes()

    if not args.kill_only:
        print("\n2. Clearing PyTorch CUDA cache (this process)...")
        clear_pytorch_cuda()

    if args.all:
        print("\n3. Clearing project cache dir...")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.environ.get("CACHE_DIR", os.path.join(root, "cache"))
        clear_project_cache(cache_dir)

    print("\n" + "=" * 60)
    print("Done. (GPU memory may take a few seconds to show as free in nvidia-smi.)")
    print("=" * 60)


if __name__ == "__main__":
    main()
