"""Comprehensive evaluation metric aggregators."""
import numpy as np
from typing import List, Dict, Any
from src.verifiers.base import VerificationResult

def compute_evaluation_metrics(
    eval_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes summary metrics across an evaluation run.
    Each item in eval_results is expected to contain:
    - 'correct': bool
    - 'parse_status': str ('success', 'unparseable', etc.)
    - 'output_tokens': int
    - 'latency_sec': float
    - 'peak_vram_mb': float
    - 'task_type': str
    - 'difficulty': str
    """
    if not eval_results:
        return {}

    n = len(eval_results)
    correct_count = sum(1 for r in eval_results if r["correct"])
    parse_success_count = sum(1 for r in eval_results if r["parse_status"] == "success")
    tokens = [r["output_tokens"] for r in eval_results]
    latencies = [r["latency_sec"] for r in eval_results]
    vrams = [r["peak_vram_mb"] for r in eval_results if r.get("peak_vram_mb") is not None]

    metrics = {
        "num_samples": n,
        "exact_accuracy": float(correct_count / n),
        "parse_success_rate": float(parse_success_count / n),
        "invalid_output_rate": float((n - parse_success_count) / n),
        "mean_output_tokens": float(np.mean(tokens)),
        "median_output_tokens": float(np.median(tokens)),
        "p95_output_tokens": float(np.percentile(tokens, 95)),
        "min_output_tokens": int(np.min(tokens)),
        "max_output_tokens": int(np.max(tokens)),
        "mean_latency_sec": float(np.mean(latencies)),
        "median_latency_sec": float(np.median(latencies)),
        "peak_vram_mb": float(np.max(vrams)) if vrams else 0.0
    }

    # Accuracy by task_type
    by_task = {}
    for r in eval_results:
        tt = r.get("task_type", "unknown")
        if tt not in by_task:
            by_task[tt] = {"total": 0, "correct": 0}
        by_task[tt]["total"] += 1
        if r["correct"]:
            by_task[tt]["correct"] += 1

    metrics["accuracy_by_task"] = {
        tt: float(stats["correct"] / stats["total"]) for tt, stats in by_task.items()
    }
    metrics["counts_by_task"] = {
        tt: stats["total"] for tt, stats in by_task.items()
    }

    # Accuracy by difficulty
    by_diff = {}
    for r in eval_results:
        diff = r.get("difficulty", "unknown")
        if diff not in by_diff:
            by_diff[diff] = {"total": 0, "correct": 0}
        by_diff[diff]["total"] += 1
        if r["correct"]:
            by_diff[diff]["correct"] += 1

    metrics["accuracy_by_difficulty"] = {
        d: float(stats["correct"] / stats["total"]) for d, stats in by_diff.items()
    }

    return metrics
