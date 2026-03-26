#!/usr/bin/env python3
"""
Prepare and upload GUIDE dataset to HuggingFace.

Usage:
    # Step 1: Prepare the dataset directory
    python scripts/prepare_hf_dataset.py --prepare \
        --videos_dir /path/to/video_self_learning/videos \
        --urls_dir /path/to/video_self_learning/urls \
        --converted_json /path/to/test_nogdrive_queries_with_videos_with_converted.json \
        --output_dir ./hf_dataset

    # Step 2: Upload to HuggingFace
    python scripts/prepare_hf_dataset.py --upload \
        --dataset_dir ./hf_dataset \
        --repo_id sharryXR/GUIDE-dataset
"""

import argparse
import os
import shutil
import json


def prepare_dataset(videos_dir, urls_dir, converted_json, verification_json, output_dir):
    """Prepare the dataset directory structure for HuggingFace upload."""
    os.makedirs(output_dir, exist_ok=True)

    # Copy videos
    if videos_dir and os.path.exists(videos_dir):
        print(f"Copying videos from {videos_dir}...")
        dst = os.path.join(output_dir, "videos")
        if not os.path.exists(dst):
            shutil.copytree(videos_dir, dst)
        print(f"  Videos copied to {dst}")
    else:
        print(f"WARNING: Videos directory not found: {videos_dir}")

    # Copy URLs
    if urls_dir and os.path.exists(urls_dir):
        print(f"Copying URLs from {urls_dir}...")
        dst = os.path.join(output_dir, "urls")
        if not os.path.exists(dst):
            shutil.copytree(urls_dir, dst)
        print(f"  URLs copied to {dst}")

    # Copy converted JSON
    if converted_json and os.path.exists(converted_json):
        print(f"Copying converted results...")
        dst_dir = os.path.join(output_dir, "converted_results")
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(converted_json, dst_dir)
        print(f"  Converted JSON copied to {dst_dir}")

    # Copy verification report
    if verification_json and os.path.exists(verification_json):
        shutil.copy2(verification_json, output_dir)
        print("  Verification report copied")

    # Print summary
    print("\n=== Dataset Summary ===")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(output_dir, '').count(os.sep)
        if level <= 1:
            indent = '  ' * level
            print(f"{indent}{os.path.basename(root)}/")

    print(f"\nDataset prepared at: {output_dir}")


def upload_dataset(dataset_dir, repo_id):
    """Upload the prepared dataset to HuggingFace."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return

    api = HfApi()

    print(f"Creating dataset repository: {repo_id}")
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    print(f"Uploading dataset from {dataset_dir} to {repo_id}...")
    print("This may take a while for large files (videos ~21GB)...")

    api.upload_folder(
        folder_path=dataset_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload GUIDE dataset: videos, annotations, URLs, and converted results",
    )

    print(f"\nDataset uploaded successfully!")
    print(f"View at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Prepare and upload GUIDE dataset to HuggingFace")
    parser.add_argument("--prepare", action="store_true", help="Prepare the dataset directory")
    parser.add_argument("--upload", action="store_true", help="Upload to HuggingFace")

    # Prepare arguments
    parser.add_argument("--videos_dir", type=str, help="Path to videos/ directory")
    parser.add_argument("--urls_dir", type=str, help="Path to urls/ directory")
    parser.add_argument("--converted_json", type=str, help="Path to converted JSON file")
    parser.add_argument("--verification_json", type=str, help="Path to video_verification_report.json")
    parser.add_argument("--output_dir", type=str, default="./hf_dataset", help="Output directory for prepared dataset")

    # Upload arguments
    parser.add_argument("--dataset_dir", type=str, help="Path to prepared dataset directory")
    parser.add_argument("--repo_id", type=str, help="HuggingFace repo ID (e.g., username/GUIDE-dataset)")

    args = parser.parse_args()

    if args.prepare:
        prepare_dataset(
            videos_dir=args.videos_dir,
            urls_dir=args.urls_dir,
            converted_json=args.converted_json,
            verification_json=args.verification_json,
            output_dir=args.output_dir,
        )

    if args.upload:
        if not args.dataset_dir:
            args.dataset_dir = args.output_dir
        if not args.repo_id:
            print("ERROR: --repo_id is required for upload")
            return
        upload_dataset(args.dataset_dir, args.repo_id)

    if not args.prepare and not args.upload:
        parser.print_help()


if __name__ == "__main__":
    main()
