"""Dataset preparation script: downloads GSM8K and synthesizes multi-step reasoning problems."""
import os
import re
import json
import random
import requests
from typing import List, Dict, Any
from src.utils.logging import setup_logger

logger = setup_logger("prepare_dataset")

GSM8K_TRAIN_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
GSM8K_TEST_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

def download_file(url: str, dest_path: str):
    """Downloads a file if it does not already exist."""
    if os.path.exists(dest_path):
        logger.info(f"File already exists at {dest_path}, skipping download.")
        return
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    logger.info(f"Downloading from {url} to {dest_path}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(resp.text)
    logger.info(f"Downloaded {len(resp.text.splitlines())} lines to {dest_path}.")

def format_solution(raw_solution: str, gold_answer: str) -> str:
    """Formats reasoning steps inside <think> tags and the final answer inside <answer> tags."""
    # Clean reasoning text by removing <<expr=res>> annotations
    clean_reasoning = re.sub(r"<<.*?>>", "", raw_solution).strip()
    # Remove #### <answer> from reasoning body
    clean_reasoning = re.sub(r"####\s*.*", "", clean_reasoning).strip()
    return f"<think>\n{clean_reasoning}\n</think>\n<answer>\n{gold_answer}\n</answer>"

def parse_gsm8k_record(record_dict: Dict[str, str], idx: int, split_name: str) -> Dict[str, Any]:
    """Parses a raw GSM8K record into our standardized schema."""
    problem = record_dict["question"].strip()
    raw_sol = record_dict["answer"].strip()
    
    # GSM8K format has '#### <final_answer>'
    parts = raw_sol.split("####")
    if len(parts) >= 2:
        gold_answer = parts[-1].strip().replace(",", "")
        reasoning = parts[0].strip()
    else:
        gold_answer = ""
        reasoning = raw_sol

    solution = format_solution(reasoning, gold_answer)
    return {
        "id": f"gsm8k_{split_name}_{idx:06d}",
        "problem": problem,
        "solution": solution,
        "final_answer": gold_answer,
        "task_type": "word_problem",
        "difficulty": "medium",
        "source": "gsm8k",
        "split": split_name,
        "verifier_type": "numeric"
    }

def generate_algebra_problem(seed_val: int) -> Dict[str, Any]:
    """Generates a multi-step algebraic word problem."""
    rng = random.Random(seed_val)
    a = rng.randint(2, 9)
    x = rng.randint(3, 25)
    b = rng.randint(5, 50)
    c = a * x + b
    
    problem = f"Solve for x in the equation: {a}x + {b} = {c}."
    reasoning = (
        f"We start with the linear equation {a}x + {b} = {c}.\n"
        f"Step 1: Subtract {b} from both sides:\n"
        f"{a}x = {c} - {b} = {c - b}.\n"
        f"Step 2: Divide both sides by {a}:\n"
        f"x = {c - b} / {a} = {x}.\n"
        f"Check: {a} * {x} + {b} = {a * x} + {b} = {c}, which matches."
    )
    solution = f"<think>\n{reasoning}\n</think>\n<answer>\n{x}\n</answer>"
    return {
        "id": f"alg_{seed_val:06d}",
        "problem": problem,
        "solution": solution,
        "final_answer": str(x),
        "task_type": "algebra",
        "difficulty": "easy" if a < 5 else "medium",
        "source": "synthetic_algebra",
        "split": "train",
        "verifier_type": "numeric"
    }

def generate_geometry_problem(seed_val: int) -> Dict[str, Any]:
    """Generates a geometry perimeter/area problem."""
    rng = random.Random(seed_val)
    width = rng.randint(4, 30)
    length = width + rng.randint(2, 15)
    area = width * length
    perimeter = 2 * (length + width)
    
    mode = rng.choice(["area", "perimeter", "width_from_area"])
    if mode == "area":
        problem = f"A rectangular garden has a length of {length} meters and a width of {width} meters. What is its area in square meters?"
        reasoning = (
            f"The formula for the area of a rectangle is Area = length * width.\n"
            f"Given length = {length} meters and width = {width} meters.\n"
            f"Area = {length} * {width} = {area} square meters."
        )
        ans = str(area)
    elif mode == "perimeter":
        problem = f"A rectangular field has a length of {length} meters and a width of {width} meters. What is the perimeter of the field in meters?"
        reasoning = (
            f"The formula for the perimeter of a rectangle is P = 2 * (length + width).\n"
            f"Given length = {length} meters and width = {width} meters.\n"
            f"P = 2 * ({length} + {width}) = 2 * {length + width} = {perimeter} meters."
        )
        ans = str(perimeter)
    else:
        problem = f"A rectangle has an area of {area} square meters and a length of {length} meters. What is its width in meters?"
        reasoning = (
            f"The formula for the area of a rectangle is Area = length * width.\n"
            f"So width = Area / length = {area} / {length} = {width} meters."
        )
        ans = str(width)

    solution = f"<think>\n{reasoning}\n</think>\n<answer>\n{ans}\n</answer>"
    return {
        "id": f"geom_{seed_val:06d}",
        "problem": problem,
        "solution": solution,
        "final_answer": ans,
        "task_type": "geometry",
        "difficulty": "medium",
        "source": "synthetic_geometry",
        "split": "train",
        "verifier_type": "numeric"
    }

def generate_rate_problem(seed_val: int) -> Dict[str, Any]:
    """Generates speed, distance, time or work rate problems."""
    rng = random.Random(seed_val)
    speed = rng.randint(30, 90)
    time = rng.randint(2, 8)
    distance = speed * time
    
    problem = f"A train travels at a constant speed of {speed} miles per hour for {time} hours. How many miles does it travel?"
    reasoning = (
        f"Distance = speed * time.\n"
        f"Given speed = {speed} mph and time = {time} hours.\n"
        f"Distance = {speed} * {time} = {distance} miles."
    )
    solution = f"<think>\n{reasoning}\n</think>\n<answer>\n{distance}\n</answer>"
    return {
        "id": f"rate_{seed_val:06d}",
        "problem": problem,
        "solution": solution,
        "final_answer": str(distance),
        "task_type": "rate_and_motion",
        "difficulty": "medium",
        "source": "synthetic_rate",
        "split": "train",
        "verifier_type": "numeric"
    }

def prepare_all_data(output_dir: str = "data"):
    """Main data preparation pipeline."""
    raw_dir = os.path.join(output_dir, "raw")
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    gsm8k_train_file = os.path.join(raw_dir, "gsm8k_train.jsonl")
    gsm8k_test_file = os.path.join(raw_dir, "gsm8k_test.jsonl")

    download_file(GSM8K_TRAIN_URL, gsm8k_train_file)
    download_file(GSM8K_TEST_URL, gsm8k_test_file)

    train_records = []
    with open(gsm8k_train_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                train_records.append(parse_gsm8k_record(json.loads(line), idx, "train"))

    test_records = []
    with open(gsm8k_test_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                test_records.append(parse_gsm8k_record(json.loads(line), idx, "test"))

    logger.info(f"Loaded {len(train_records)} GSM8K train records and {len(test_records)} GSM8K test records.")

    # Generate synthetic diverse mathematical problems
    logger.info("Generating diverse multi-step mathematical problems (algebra, geometry, rate)...")
    synthetic_records = []
    for i in range(1, 401):
        synthetic_records.append(generate_algebra_problem(1000 + i))
        synthetic_records.append(generate_geometry_problem(2000 + i))
        synthetic_records.append(generate_rate_problem(3000 + i))

    logger.info(f"Generated {len(synthetic_records)} diverse mathematical problems.")

    # Combine into raw pool
    combined_train = train_records + synthetic_records
    
    processed_train_path = os.path.join(processed_dir, "train_pool.jsonl")
    processed_test_path = os.path.join(processed_dir, "test_pool.jsonl")

    with open(processed_train_path, "w", encoding="utf-8") as f:
        for r in combined_train:
            f.write(json.dumps(r) + "\n")

    with open(processed_test_path, "w", encoding="utf-8") as f:
        for r in test_records:
            f.write(json.dumps(r) + "\n")

    logger.info(f"Saved {len(combined_train)} records to {processed_train_path}")
    logger.info(f"Saved {len(test_records)} records to {processed_test_path}")

if __name__ == "__main__":
    prepare_all_data()
