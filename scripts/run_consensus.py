"""CLI runner for Self-Consistency (Consensus@K) evaluation."""
import os
import sys
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.load_model import load_peft_checkpoint
from src.evaluation.evaluate_consensus import evaluate_self_consistency
from src.utils.logging import setup_logger

logger = setup_logger("run_consensus")

def main():
    parser = argparse.ArgumentParser(description="Run Self-Consistency Consensus Evaluation")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=str, default="outputs/checkpoints/grpo/final")
    parser.add_argument("--data", type=str, default="data/splits/test.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--k", type=int, default=8, help="Number of sampled rollouts per problem")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--output-prefix", type=str, default="grpo_consensus")
    args = parser.parse_args()

    logger.info("==================================================")
    logger.info(f"SELF-CONSISTENCY CONSENSUS@{args.k} BENCHMARK")
    logger.info("==================================================")

    logger.info(f"Loading checkpoint from: {args.adapter}")
    model, tokenizer = load_peft_checkpoint(args.model, args.adapter)

    summary = evaluate_self_consistency(
        model=model,
        tokenizer=tokenizer,
        test_data_path=args.data,
        k_samples=args.k,
        temperature=args.temperature,
        limit=args.limit,
        output_prefix=args.output_prefix
    )

    print("\nConsensus Benchmark Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

