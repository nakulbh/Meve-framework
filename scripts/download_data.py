#!/usr/bin/env python3
"""Download HotpotQA dataset for MeVe Framework"""

import json
import os
from pathlib import Path

from datasets import load_dataset


def download_hotpotqa(data_dir="data", split="validation", max_samples=None):
    """
    Download HotpotQA dataset from Hugging Face

    Args:
        data_dir: Directory to save data
        split: 'train' or 'validation'
        max_samples: Limit number of samples (None for all)
    """
    print("📥 Downloading HotpotQA dataset from Hugging Face...")

    # Create data directory
    Path(data_dir).mkdir(exist_ok=True)

    # Load dataset
    print(f"Loading {split} split...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)

    # Limit samples if specified
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples")

    # Convert to format expected by meve_engine.py
    hotpot_data = []
    for item in dataset:
        hotpot_data.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "type": item["type"],
                "level": item["level"],
                "supporting_facts": item["supporting_facts"],
                "context": item["context"],
            }
        )

    # Save to JSON file
    output_file = os.path.join(
        data_dir,
        "hotpot_dev_distractor_v1.json" if split == "validation" else "hotpot_train_v1.1.json",
    )

    print(f"💾 Saving {len(hotpot_data)} examples to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(hotpot_data, f, ensure_ascii=False, indent=2)

    print("✅ Successfully downloaded HotpotQA data!")
    print(f"   File: {output_file}")
    print(f"   Samples: {len(hotpot_data)}")
    return output_file


if __name__ == "__main__":
    # Download validation set with limited samples for faster setup
    download_hotpotqa(data_dir="data", split="validation", max_samples=100)
    print("\n🎉 Data ready! You can now run: make run")
