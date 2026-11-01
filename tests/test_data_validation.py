"""Unit tests for dataset schema validation and leakage checking."""
import pytest
from src.data.validate_dataset import validate_record

def test_valid_record():
    record = {
        "id": "math_001",
        "problem": "What is 5 + 3?",
        "solution": "<think>5 + 3 = 8</think><answer>8</answer>",
        "final_answer": "8",
        "task_type": "arithmetic",
        "difficulty": "easy",
        "source": "unit_test",
        "split": "train",
        "verifier_type": "numeric"
    }
    valid, errors = validate_record(record)
    assert valid is True
    assert len(errors) == 0

def test_missing_field():
    record = {
        "id": "math_002",
        "problem": "What is 5 + 3?",
        # missing solution
        "final_answer": "8",
        "task_type": "arithmetic",
        "difficulty": "easy",
        "source": "unit_test",
        "split": "train",
        "verifier_type": "numeric"
    }
    valid, errors = validate_record(record)
    assert valid is False
    assert any("Missing required field: 'solution'" in e for e in errors)

def test_gold_solution_verifier_mismatch():
    record = {
        "id": "math_003",
        "problem": "What is 5 + 3?",
        "solution": "<think>5 + 3 = 9</think><answer>9</answer>",  # Wrong answer inside solution tag
        "final_answer": "8",
        "task_type": "arithmetic",
        "difficulty": "easy",
        "source": "unit_test",
        "split": "train",
        "verifier_type": "numeric"
    }
    valid, errors = validate_record(record)
    assert valid is False
    assert any("Verifier failed on gold solution" in e for e in errors)
