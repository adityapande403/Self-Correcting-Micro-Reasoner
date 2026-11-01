"""Unified evaluation CLI script for all training stages."""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.load_model import load_base_model, load_tokenizer, load_peft_checkpoint
from src.evaluation.evaluate_model import evaluate_dataset
from src.evaluation.evaluate_correction import run_self_correction_experiments
from src.utils.logging import setup_logger

logger = setup_logger("run_evaluation")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Model or Checkpoint")
    parser.add_argument("--stage", type=str, default="baseline", choices=["baseline", "sft", "grpo", "grpo_scale"], help="Evaluation stage")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model identifier")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--data", type=str, default="data/splits/test.jsonl", help="Test dataset path")
    parser.add_argument("--limit", type=int, default=100, help="Test sample limit")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--correction", action="store_true", help="Run self-correction experiments")
    args = parser.parse_args()

    # Determine adapter path based on stage if not explicitly given
    adapter = args.adapter
    if adapter is None:
        if args.stage == "sft":
            adapter = "outputs/checkpoints/sft/final"
        elif args.stage == "grpo":
            adapter = "outputs/checkpoints/grpo/final"
        elif args.stage == "grpo_scale":
            adapter = "outputs/checkpoints/grpo_scale/final"

    logger.info(f"Evaluating stage: {args.stage} (adapter={adapter})")

    if adapter and os.path.exists(adapter):
        logger.info(f"Loading PEFT model with adapter from: {adapter}")
        model, tokenizer = load_peft_checkpoint(args.model, adapter)
    else:
        logger.info(f"Loading base model: {args.model}")
        tokenizer = load_tokenizer(args.model)
        model = load_base_model(args.model)

    # 1. Standard Benchmark Evaluation
    summary = evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        test_data_path=args.data,
        batch_size=args.batch_size,
        limit=args.limit,
        output_prefix=args.stage
    )
    print("\nBenchmark Evaluation Results:")
    print(json.dumps(summary, indent=2))

    # 2. Self-Correction Evaluation if requested
    if args.correction:
        corr_summary = run_self_correction_experiments(
            model=model,
            tokenizer=tokenizer,
            test_data_path=args.data,
            limit=min(args.limit, 50),
            batch_size=min(args.batch_size, 2),
            stage_name=args.stage
        )
        print("\nSelf-Correction Results:")
        print(json.dumps(corr_summary, indent=2))

if __name__ == "__main__":
    main()
