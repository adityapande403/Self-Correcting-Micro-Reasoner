"""Dataset validation and leak detection utility."""
import os
import json
from typing import List, Dict, Any, Tuple
from src.utils.logging import setup_logger
from src.verifiers import verify_solution

logger = setup_logger("validate_dataset")

REQUIRED_FIELDS = [
    "id", "problem", "solution", "final_answer",
    "task_type", "difficulty", "source", "split", "verifier_type"
]

def validate_record(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validates an individual record against schema and verification consistency."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"Missing required field: '{field}'")
        elif record[field] is None or (isinstance(record[field], str) and not record[field].strip()):
            errors.append(f"Empty or null field: '{field}'")

    if errors:
        return False, errors

    # Check that solution correctly parses and satisfies its own final_answer
    v_res = verify_solution(
        prediction=record["solution"],
        ground_truth=record["final_answer"],
        verifier_type=record.get("verifier_type", "numeric")
    )
    if not v_res.correct:
        errors.append(
            f"Verifier failed on gold solution: parse_status={v_res.parse_status}, "
            f"reason={v_res.failure_reason}, extracted={v_res.extracted_answer}, gold={record['final_answer']}"
        )

    return len(errors) == 0, errors

def validate_file(file_path: str) -> Dict[str, Any]:
    """Validates an entire JSONL file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    total = 0
    valid = 0
    invalid = 0
    malformed = []
    seen_problems = set()
    duplicates = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                malformed.append({"line": line_no, "errors": [f"JSONDecodeError: {e}"]})
                invalid += 1
                continue

            # Duplicate check
            norm_prob = data.get("problem", "").strip().lower()
            if norm_prob in seen_problems:
                duplicates += 1
            else:
                seen_problems.add(norm_prob)

            ok, errors = validate_record(data)
            if ok:
                valid += 1
            else:
                invalid += 1
                malformed.append({"line": line_no, "id": data.get("id"), "errors": errors})

    report = {
        "file_path": file_path,
        "total_records": total,
        "valid_records": valid,
        "invalid_records": invalid,
        "duplicate_problems": duplicates,
        "malformed_examples": malformed[:10]  # First 10 errors
    }
    logger.info(
        f"Validation for {file_path}: Total={total}, Valid={valid}, "
        f"Invalid={invalid}, Duplicates={duplicates}"
    )
    return report

def check_leakage(train_file: str, test_file: str) -> Dict[str, Any]:
    """Checks for exact question leakage between train and test splits."""
    train_problems = set()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_problems.add(json.loads(line)["problem"].strip().lower())

    leaked = []
    test_total = 0
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                test_total += 1
                p = json.loads(line)["problem"].strip().lower()
                if p in train_problems:
                    leaked.append(p)

    leakage_rate = len(leaked) / max(1, test_total)
    logger.info(f"Leakage check: {len(leaked)} / {test_total} test problems leaked into training ({leakage_rate:.2%}).")
    return {
        "test_total": test_total,
        "leaked_count": len(leaked),
        "leakage_rate": leakage_rate,
        "leaked_samples": leaked[:5]
    }

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "data/processed/train_pool.jsonl"
    report = validate_file(target)
    print(json.dumps(report, indent=2))
