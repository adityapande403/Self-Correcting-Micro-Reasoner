"""Evaluation script for benchmarking models on held-out test datasets."""
import os
import json
import time
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.models.load_model import load_base_model, load_tokenizer, load_peft_checkpoint
from src.models.inference import InferenceEngine
from src.verifiers import verify_solution
from src.evaluation.metrics import compute_evaluation_metrics
from src.utils.logging import setup_logger

logger = setup_logger("evaluate_model")

def evaluate_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_data_path: str = "data/splits/test.jsonl",
    batch_size: int = 4,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    limit: Optional[int] = None,
    output_prefix: str = "eval",
    output_dir: str = "outputs/reports"
) -> Dict[str, Any]:
    """Runs evaluation across the test set and saves full results."""
    os.makedirs(output_dir, exist_ok=True)
    engine = InferenceEngine(model=model, tokenizer=tokenizer)

    # Load test data
    records = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if limit is not None and limit > 0:
        records = records[:limit]

    logger.info(f"Evaluating {len(records)} samples from {test_data_path} (batch_size={batch_size})")

    detailed_results = []
    
    # Process in batches
    for i in tqdm(range(0, len(records), batch_size), desc="Evaluating"):
        batch_records = records[i:i + batch_size]
        prompts = [engine.format_prompt(r["problem"]) for r in batch_records]
        
        gen_outputs = engine.generate(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )

        for record, gen in zip(batch_records, gen_outputs):
            v_res = verify_solution(
                prediction=gen["completion"],
                ground_truth=record["final_answer"],
                verifier_type=record.get("verifier_type", "numeric")
            )

            res_entry = {
                "id": record["id"],
                "problem": record["problem"],
                "final_answer": record["final_answer"],
                "completion": gen["completion"],
                "extracted_answer": v_res.extracted_answer,
                "correct": v_res.correct,
                "reward": v_res.reward,
                "parse_status": v_res.parse_status,
                "failure_reason": v_res.failure_reason,
                "task_type": record.get("task_type", "unknown"),
                "difficulty": record.get("difficulty", "unknown"),
                "input_tokens": gen["input_tokens"],
                "output_tokens": gen["output_tokens"],
                "latency_sec": gen["latency_sec"],
                "peak_vram_mb": gen["peak_vram_mb"]
            }
            detailed_results.append(res_entry)

    summary_metrics = compute_evaluation_metrics(detailed_results)
    summary_metrics["stage"] = output_prefix
    summary_metrics["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save detailed JSON and summary JSON
    detailed_file = os.path.join(output_dir, f"{output_prefix}_detailed.json")
    summary_file = os.path.join(output_dir, f"{output_prefix}_summary.json")

    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info(
        f"[{output_prefix}] Exact Accuracy: {summary_metrics['exact_accuracy']:.2%}, "
        f"Parse Rate: {summary_metrics['parse_success_rate']:.2%}, "
        f"Mean Tokens: {summary_metrics['mean_output_tokens']:.1f}, "
        f"Mean Latency: {summary_metrics['mean_latency_sec']:.3f}s"
    )
    return summary_metrics

def run_evaluation_cli(
    model_name_or_path: str,
    adapter_path: Optional[str] = None,
    test_data_path: str = "data/splits/test.jsonl",
    batch_size: int = 4,
    limit: Optional[int] = None,
    stage_name: str = "baseline"
):
    """CLI runner for evaluation."""
    if adapter_path and os.path.exists(adapter_path):
        model, tokenizer = load_peft_checkpoint(model_name_or_path, adapter_path)
    else:
        tokenizer = load_tokenizer(model_name_or_path)
        model = load_base_model(model_name_or_path)

    evaluate_dataset(
        model=model,
        tokenizer=tokenizer,
        test_data_path=test_data_path,
        batch_size=batch_size,
        limit=limit,
        output_prefix=stage_name
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--data", type=str, default="data/splits/test.jsonl")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stage", type=str, default="baseline")
    args = parser.parse_args()

    run_evaluation_cli(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        test_data_path=args.data,
        batch_size=args.batch_size,
        limit=args.limit,
        stage_name=args.stage
    )
