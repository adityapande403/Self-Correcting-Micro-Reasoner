"""Script to evaluate the unmodified baseline model on the held-out test set."""
import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_yaml
from src.utils.reproducibility import set_seed, get_environment_info
from src.models.load_model import load_base_model, load_tokenizer
from src.evaluation.evaluate_model import evaluate_dataset
from src.utils.logging import setup_logger

logger = setup_logger("run_baseline")

def main():
    parser = argparse.ArgumentParser(description="Run Baseline Evaluation on Unmodified Model")
    parser.add_argument("--config", type=str, default="configs/evaluation.yaml", help="Path to evaluation config")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model identifier")
    parser.add_argument("--limit", type=int, default=100, help="Number of test samples to evaluate (default: 100)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    logger.info("==================================================")
    logger.info("STAGE A: BASELINE EVALUATION (UNMODIFIED MODEL)")
    logger.info("==================================================")
    
    env_info = get_environment_info()
    logger.info(f"Environment: {env_info['device_name']}, PyTorch {env_info['torch_version']}")

    cfg = load_yaml(args.config) if os.path.exists(args.config) else {}
    test_path = cfg.get("test_dataset_path", "data/splits/test.jsonl")
    output_dir = cfg.get("output_dir", "outputs/reports")

    tokenizer = load_tokenizer(args.model)
    model = load_base_model(args.model)

    summary = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        test_data_path=test_path,
        batch_size=args.batch_size,
        limit=args.limit,
        output_prefix="baseline",
        output_dir=output_dir
    )

    logger.info("Baseline evaluation complete!")
    logger.info(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
