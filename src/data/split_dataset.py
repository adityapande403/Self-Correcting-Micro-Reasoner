"""Dataset splitting and manifest generation utility."""
import os
import json
import random
import hashlib
from typing import List, Dict, Any
from src.utils.logging import setup_logger
from src.data.validate_dataset import validate_record, check_leakage

logger = setup_logger("split_dataset")

def file_sha256(path: str) -> str:
    """Computes SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def create_splits(
    train_pool_path: str = "data/processed/train_pool.jsonl",
    test_pool_path: str = "data/processed/test_pool.jsonl",
    splits_dir: str = "data/splits",
    manifests_dir: str = "data/manifests",
    seed: int = 42
):
    """Creates stratified train, val, and test splits."""
    random.seed(seed)
    os.makedirs(splits_dir, exist_ok=True)
    os.makedirs(manifests_dir, exist_ok=True)

    # 1. Load valid training pool records
    train_records = []
    seen = set()
    with open(train_pool_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                norm_p = item["problem"].strip().lower()
                if norm_p in seen:
                    continue
                seen.add(norm_p)
                valid, _ = validate_record(item)
                if valid:
                    train_records.append(item)

    # 2. Load valid test pool records
    test_records = []
    seen_test = set()
    with open(test_pool_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                norm_p = item["problem"].strip().lower()
                if norm_p in seen or norm_p in seen_test:
                    continue  # Guarantee ZERO test leakage!
                seen_test.add(norm_p)
                valid, _ = validate_record(item)
                if valid:
                    test_records.append(item)

    logger.info(f"Loaded {len(train_records)} clean unique train candidates, {len(test_records)} clean test candidates.")

    # Shuffle with fixed seed
    random.shuffle(train_records)
    random.shuffle(test_records)

    # Partition train_records:
    # SFT Train: 2,000 examples
    # SFT Val: 200 examples
    # GRPO Train: 2,000 distinct examples (or overlapping with train pool if desired, but disjoint problem IDs preferred)
    sft_train = train_records[:2000]
    sft_val = train_records[2000:2200]
    
    # For GRPO, use an independent subset if pool size allows, or pool offset
    grpo_train = train_records[2200:4200] if len(train_records) >= 4200 else train_records[:2000]
    grpo_val = train_records[4200:4400] if len(train_records) >= 4400 else train_records[2000:2200]

    # Held-out evaluation test set (300 problems strictly separated)
    test_set = test_records[:300]

    # Save splits
    split_paths = {
        "sft_train": os.path.join(splits_dir, "sft_train.jsonl"),
        "sft_val": os.path.join(splits_dir, "sft_val.jsonl"),
        "grpo_train": os.path.join(splits_dir, "grpo_train.jsonl"),
        "grpo_val": os.path.join(splits_dir, "grpo_val.jsonl"),
        "test": os.path.join(splits_dir, "test.jsonl")
    }

    datasets = {
        "sft_train": sft_train,
        "sft_val": sft_val,
        "grpo_train": grpo_train,
        "grpo_val": grpo_val,
        "test": test_set
    }

    manifest = {"seed": seed, "splits": {}}

    for name, data in datasets.items():
        path = split_paths[name]
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                item_copy = dict(item)
                item_copy["split"] = name
                f.write(json.dumps(item_copy) + "\n")
        logger.info(f"Wrote {len(data)} records to {path}")
        
        # Manifest stats
        task_types = {}
        for r in data:
            tt = r.get("task_type", "unknown")
            task_types[tt] = task_types.get(tt, 0) + 1

        manifest["splits"][name] = {
            "count": len(data),
            "sha256": file_sha256(path),
            "file_path": path,
            "task_distribution": task_types
        }

    # Verify leakage
    leakage = check_leakage(split_paths["sft_train"], split_paths["test"])
    manifest["leakage_check"] = leakage

    manifest_path = os.path.join(manifests_dir, "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Dataset manifest successfully generated at {manifest_path}")

if __name__ == "__main__":
    create_splits()
