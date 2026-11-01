"""Self-Correction evaluation module implementing Experiments A, B, C, and D."""
import os
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.models.inference import InferenceEngine
from src.verifiers import verify_solution
from src.verifiers.base import extract_think_tag
from src.utils.logging import setup_logger

logger = setup_logger("evaluate_correction")

def run_self_correction_experiments(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_data_path: str = "data/splits/test.jsonl",
    limit: Optional[int] = 50,
    batch_size: int = 2,
    output_dir: str = "outputs/reports",
    stage_name: str = "eval"
) -> Dict[str, Any]:
    """Runs all 4 self-correction experiments and computes rigorous comparative metrics."""
    os.makedirs(output_dir, exist_ok=True)
    engine = InferenceEngine(model=model, tokenizer=tokenizer)

    records = []
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if limit:
        records = records[:limit]

    logger.info(f"Running self-correction experiments on {len(records)} test samples...")

    exp_c_results = []
    exp_b_results = []

    # Prompt templates
    prompt_single = "You are a mathematical reasoning assistant. First think in <think>...</think>, then give the final answer in <answer>...</answer>.\n\nProblem: {p}"
    prompt_verify = (
        "You are a mathematical reasoning assistant. "
        "Solve the problem step by step, and explicitly verify your calculation before giving your final answer. "
        "First think in <think>...</think>, then give the final answer in <answer>...</answer>.\n\nProblem: {p}"
    )

    # 1. Run Single-Pass (Exp A) and Verification Prompt (Exp B)
    for i in tqdm(range(0, len(records), batch_size), desc="Correction Pass 1 & Exp B"):
        batch = records[i:i + batch_size]
        
        # Single-pass prompts
        prompts_a = [engine.format_prompt(r["problem"]) for r in batch]
        gens_a = engine.generate(prompts_a, max_new_tokens=512, temperature=0.0)

        # Explicit verification prompt (Exp B)
        prompts_b = [prompt_verify.format(p=r["problem"]) for r in batch]
        gens_b = engine.generate(prompts_b, max_new_tokens=512, temperature=0.0)

        for r, gen_a, gen_b in zip(batch, gens_a, gens_b):
            v_a = verify_solution(gen_a["completion"], r["final_answer"], r.get("verifier_type", "numeric"))
            v_b = verify_solution(gen_b["completion"], r["final_answer"], r.get("verifier_type", "numeric"))

            exp_b_results.append({
                "id": r["id"],
                "correct_a": v_a.correct,
                "correct_b": v_b.correct,
                "tokens_a": gen_a["output_tokens"],
                "tokens_b": gen_b["output_tokens"]
            })

            exp_c_results.append({
                "id": r["id"],
                "problem": r["problem"],
                "final_answer": r["final_answer"],
                "verifier_type": r.get("verifier_type", "numeric"),
                "pass1_completion": gen_a["completion"],
                "pass1_correct": v_a.correct,
                "pass1_extracted": v_a.extracted_answer,
                "pass1_tokens": gen_a["output_tokens"],
                "pass1_latency": gen_a["latency_sec"]
            })

    # 2. Run Revision Pass (Exp C Pass 2)
    logger.info("Running Revision Pass (Exp C)...")
    for i in tqdm(range(0, len(exp_c_results), batch_size), desc="Correction Pass 2"):
        batch = exp_c_results[i:i + batch_size]
        revision_prompts = []
        for item in batch:
            rev_prompt = (
                f"Question: {item['problem']}\n\n"
                f"Your previous attempt was:\n{item['pass1_completion']}\n\n"
                f"Please carefully review this solution. Check every arithmetic step. "
                f"If you find an error, correct it. Write your thought process in <think>...</think> "
                f"and your final answer in <answer>...</answer>."
            )
            revision_prompts.append(engine.format_prompt(rev_prompt))

        gens_rev = engine.generate(revision_prompts, max_new_tokens=512, temperature=0.0)

        for item, gen_rev in zip(batch, gens_rev):
            v_rev = verify_solution(gen_rev["completion"], item["final_answer"], item["verifier_type"])
            item["pass2_completion"] = gen_rev["completion"]
            item["pass2_correct"] = v_rev.correct
            item["pass2_extracted"] = v_rev.extracted_answer
            item["pass2_tokens"] = gen_rev["output_tokens"]
            item["pass2_latency"] = gen_rev["latency_sec"]

    # 3. Analyze Transition Outcomes
    total = len(exp_c_results)
    initial_correct = sum(1 for r in exp_c_results if r["pass1_correct"])
    revised_correct = sum(1 for r in exp_c_results if r["pass2_correct"])

    # Transitions:
    # False -> True (Successful correction)
    # True -> False (Harmful regression)
    # True -> True (Maintained correct)
    # False -> False (Persistent failure)
    wrong_to_right = sum(1 for r in exp_c_results if not r["pass1_correct"] and r["pass2_correct"])
    right_to_wrong = sum(1 for r in exp_c_results if r["pass1_correct"] and not r["pass2_correct"])
    right_to_right = sum(1 for r in exp_c_results if r["pass1_correct"] and r["pass2_correct"])
    wrong_to_wrong = sum(1 for r in exp_c_results if not r["pass1_correct"] and not r["pass2_correct"])

    initial_wrong = total - initial_correct

    correction_rate = (wrong_to_right / initial_wrong) if initial_wrong > 0 else 0.0
    regression_rate = (right_to_wrong / initial_correct) if initial_correct > 0 else 0.0
    initial_acc = initial_correct / total
    revised_acc = revised_correct / total
    net_acc_change = revised_acc - initial_acc

    # Token and latency overhead
    mean_p1_tokens = float(np.mean([r["pass1_tokens"] for r in exp_c_results]))
    mean_p2_tokens = float(np.mean([r["pass2_tokens"] for r in exp_c_results]))
    mean_total_tokens = mean_p1_tokens + mean_p2_tokens

    # Exp B accuracy
    exp_b_acc = sum(1 for r in exp_b_results if r["correct_b"]) / total

    # Exp D: Natural self-correction audit in Pass 1
    # Check if think text contains internal reconsideration markers followed by correct answers
    natural_correction_keywords = ["wait,", "actually,", "mistake", "recalculat", "check again", "let's re-evaluate"]
    natural_revisions_detected = 0
    natural_revisions_correct = 0
    for r in exp_c_results:
        think = extract_think_tag(r["pass1_completion"])
        if think:
            think_lower = think.lower()
            if any(kw in think_lower for kw in natural_correction_keywords):
                natural_revisions_detected += 1
                if r["pass1_correct"]:
                    natural_revisions_correct += 1

    summary = {
        "stage": stage_name,
        "sample_count": total,
        "experiment_a_initial_accuracy": initial_acc,
        "experiment_b_verification_prompt_accuracy": exp_b_acc,
        "experiment_c_revised_accuracy": revised_acc,
        "net_accuracy_change": net_acc_change,
        "wrong_to_right_count": wrong_to_right,
        "right_to_wrong_count": right_to_wrong,
        "right_to_right_count": right_to_right,
        "wrong_to_wrong_count": wrong_to_wrong,
        "correction_rate": correction_rate,
        "regression_rate": regression_rate,
        "mean_pass1_tokens": mean_p1_tokens,
        "mean_pass2_tokens": mean_p2_tokens,
        "mean_total_two_pass_tokens": mean_total_tokens,
        "token_overhead_ratio": mean_total_tokens / max(1, mean_p1_tokens),
        "natural_reconsideration_detected_count": natural_revisions_detected,
        "natural_reconsideration_correct_count": natural_revisions_correct
    }

    report_path = os.path.join(output_dir, f"{stage_name}_correction_summary.json")
    detailed_path = os.path.join(output_dir, f"{stage_name}_correction_detailed.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(detailed_path, "w", encoding="utf-8") as f:
        json.dump(exp_c_results, f, indent=2)

    logger.info(
        f"[{stage_name} Self-Correction] Initial Acc: {initial_acc:.2%}, "
        f"Revised Acc: {revised_acc:.2%}, Net Change: {net_acc_change:+.2%}, "
        f"Correction Rate: {correction_rate:.2%}, Regression Rate: {regression_rate:.2%}"
    )
    return summary
