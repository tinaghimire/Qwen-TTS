#!/usr/bin/env python3
"""
Upload pre-processed TTS data (with audio_codes) to HuggingFace.
This allows you to train directly without intermediate JSONL files.
"""
import argparse
import json
import os
from pathlib import Path
from datasets import Dataset, DatasetDict, load_from_disk

def jsonl_to_hf_dataset(jsonl_path, output_dir, dataset_name, split="train"):
    """
    Convert JSONL file to HuggingFace dataset and save locally.
    """
    print(f"Loading JSONL from {jsonl_path}")
    
    # Load data from JSONL
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"✓ Loaded {len(data)} samples")
    
    # Create HuggingFace dataset
    print("Creating HuggingFace dataset...")
    dataset = Dataset.from_list(data)
    
    # Save locally
    output_path = os.path.join(output_dir, dataset_name)
    print(f"Saving to {output_path}")
    dataset.save_to_disk(output_path)
    
    print(f"✓ Dataset saved locally to {output_path}")
    print(f"\nTo upload to HuggingFace, run:")
    print(f"  huggingface-cli repo create {dataset_name} --type dataset --private")
    print(f"  cd {output_path}")
    print(f"  git init")
    print(f"  git lfs install")
    print(f"  git add .")
    print(f"  git commit -m 'Add TTS dataset with audio_codes'")
    print(f"  git push origin main")
    
    return dataset

def upload_to_huggingface(dataset_path, repo_id, private=False):
    """
    Upload local dataset to HuggingFace.
    """
    from huggingface_hub import login, upload_folder
    
    print(f"Uploading to HuggingFace: {repo_id}")
    
    # Load the dataset
    dataset = load_from_disk(dataset_path)
    
    # Push to hub
    dataset.push_to_hub(repo_id, private=private)
    
    print(f"✓ Successfully uploaded to https://huggingface.co/datasets/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Upload TTS data to HuggingFace")
    parser.add_argument("--jsonl_path", type=str, help="Path to JSONL file")
    parser.add_argument("--output_dir", type=str, default="./hf_datasets", help="Output directory")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace after saving")
    parser.add_argument("--hf_token", type=str, help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    
    args = parser.parse_args()
    
    if not args.jsonl_path:
        # Try to find default JSONL
        default_paths = [
            "./data/train.jsonl",
            "./data/hausa_train.jsonl",
        ]
        for path in default_paths:
            if os.path.exists(path):
                args.jsonl_path = path
                print(f"Using default JSONL path: {path}")
                break
        
        if not args.jsonl_path:
            print("Error: Must specify --jsonl_path")
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Convert JSONL to HF dataset
    dataset = jsonl_to_hf_dataset(args.jsonl_path, args.output_dir, args.dataset_name)
    
    # Display sample
    print("\nSample data:")
    print(json.dumps(dataset[0], indent=2, default=str))
    
    # Upload if requested
    if args.upload:
        if args.hf_token:
            os.environ["HF_TOKEN"] = args.hf_token
        
        dataset_path = os.path.join(args.output_dir, args.dataset_name)
        repo_id = args.dataset_name
        
        try:
            upload_to_huggingface(dataset_path, repo_id, private=args.private)
            
            print(f"\n✓ Everything done!")
            print(f"\nYou can now train directly with:")
            print(f"  .venv/bin/python train_from_hf.py \\")
            print(f"    --hf_dataset {repo_id} \\")
            print(f"    --split train \\")
            print(f"    --ref_audio ./voices/english_voice/english_voice_24k.wav \\")
            print(f"    --batch_size 1 \\")
            print(f"    --max_samples 100")
            
        except Exception as e:
            print(f"Error uploading to HuggingFace: {e}")
            print(f"Note: Make sure you have 'huggingface_hub' installed and valid token")
            print(f"Install: .venv/bin/pip install huggingface_hub")
    else:
        print(f"\n✓ Dataset saved locally")
        print(f"Path: {os.path.join(args.output_dir, args.dataset_name)}")
        print(f"\nTo upload to HuggingFace later, run:")
        print(f"  .venv/bin/python {os.path.basename(__file__)} \\")
        print(f"    --dataset_name {args.dataset_name} \\")
        print(f"    --jsonl_path {args.jsonl_path} \\")
        print(f"    --upload")

if __name__ == "__main__":
    main()