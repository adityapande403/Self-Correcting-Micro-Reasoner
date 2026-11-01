"""Unit tests for metric computation and checkpoint comparison."""
import pytest
from src.evaluation.metrics import compute_evaluation_metrics
from src.evaluation.compare_checkpoints import generate_comparison_table

def test_compute_evaluation_metrics():
    sample_results = [
        {
            "correct": True,
            "parse_status": "success",
            "output_tokens": 150,
            "latency_sec": 1.5,
            "peak_vram_mb": 3100.0,
            "task_type": "algebra",
            "difficulty": "easy"
        },
        {
            "correct": False,
            "parse_status": "success",
            "output_tokens": 200,
            "latency_sec": 2.0,
            "peak_vram_mb": 3150.0,
            "task_type": "algebra",
            "difficulty": "medium"
        },
        {
            "correct": False,
            "parse_status": "unparseable",
            "output_tokens": 250,
            "latency_sec": 2.5,
            "peak_vram_mb": 3200.0,
            "task_type": "geometry",
            "difficulty": "hard"
        }
    ]

    metrics = compute_evaluation_metrics(sample_results)
    assert metrics["num_samples"] == 3
    assert abs(metrics["exact_accuracy"] - (1.0 / 3.0)) < 1e-4
    assert abs(metrics["parse_success_rate"] - (2.0 / 3.0)) < 1e-4
    assert abs(metrics["invalid_output_rate"] - (1.0 / 3.0)) < 1e-4
    assert metrics["mean_output_tokens"] == 200.0
    assert metrics["accuracy_by_task"]["algebra"] == 0.5
    assert metrics["accuracy_by_task"]["geometry"] == 0.0

def test_generate_comparison_table():
    base = {"exact_accuracy": 0.28, "parse_success_rate": 0.38, "mean_output_tokens": 389.0}
    sft = {"exact_accuracy": 0.52, "parse_success_rate": 0.94, "mean_output_tokens": 280.0}
    grpo = {"exact_accuracy": 0.68, "parse_success_rate": 0.98, "mean_output_tokens": 310.0}

    table_md = generate_comparison_table(base, sft, grpo)
    assert "Exact Accuracy" in table_md
    assert "28.00%" in table_md
    assert "52.00%" in table_md
    assert "68.00%" in table_md
