"""Self-Consistency and Majority Voting (Consensus@K) evaluation module."""
import os
import json
import time
from collections import Counter
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.models.inference import InferenceEngine
from src.verifiers import verify_solution
from src.verifiers.numeric import normalize_numeric_string
from src.utils.logging import setup_logger

logger = setup_logger("evaluate_consensus")

def evaluate_self_consistency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    test_data_path: str = "data/splits/test.jsonl",
    k_samples: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.9,
    limit: Optional[int] = 100,
    output_prefix: str = "grpo_consensus",
    output_dir: str = "outputs/reports"
) -> Dict[str, Any]:
    """
    Evaluates self-consistency via majority voting over K sampled reasoning rollouts.
    Computes:
      - pass@1: Average accuracy across all individual sampled rollouts
      - cons@K (Majority Vote): Accuracy of the plurality/mode extracted answer
      - pass@K: Upper bound accuracy (at least 1 of the K samples was correct)
    """
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

    logger.info(f"Running Self-Consistency (K={k_samples}, T={temperature}) on {len(records)} test samples...")

    detailed_results = []
    total_samples_evaluated = 0
    total_individual_correct = 0
    total_consensus_correct = 0
    total_pass_at_k_correct = 0
    total_tokens_generated = 0

    for record in tqdm(records, desc=f"Consensus@{k_samples}"):
        prompt = engine.format_prompt(record["problem"])
        gt = record["final_answer"]
        verifier_type = record.get("verifier_type", "numeric")

        # Sample K completions
        prompts_batch = [prompt] * k_samples
        gen_outputs = engine.generate(
            prompts=prompts_batch,
            max_new_tokens=512,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )

        sample_details = []
        candidate_votes = []
        correct_in_k = 0

        for gen in gen_outputs:
            completion = gen["completion"]
            v_res = verify_solution(completion, gt, verifier_type=verifier_type)
            extracted = v_res.extracted_answer
            is_correct = v_res.correct

            total_tokens_generated += gen["output_tokens"]
            if is_correct:
                correct_in_k += 1
                total_individual_correct += 1
            total_samples_evaluated += 1

            # Normalization key for voting
            vote_key = None
            if extracted:
                norm_val = normalize_numeric_string(extracted)
                vote_key = str(norm_val) if norm_val is not None else extracted.strip().lower()
                candidate_votes.append((vote_key, extracted, completion))

            sample_details.append({
                "completion": completion,
                "extracted_answer": extracted,
                "normalized_vote_key": vote_key,
                "correct": is_correct,
                "tokens": gen["output_tokens"]
            })

        # Determine Majority / Consensus Winner
        consensus_winner_correct = False
        winning_answer = None
        agreement_count = 0
        winning_completion = None

        if candidate_votes:
            key_counts = Counter([v[0] for v in candidate_votes])
            most_common_key, agreement_count = key_counts.most_common(1)[0]
            # Find the representative extracted answer for this key
            for key, raw_ext, comp in candidate_votes:
                if key == most_common_key:
                    winning_answer = raw_ext
                    winning_completion = comp
                    break
            
            # Verify the winning answer
            v_consensus = verify_solution(
                prediction=f"<answer>{winning_answer}</answer>",
                ground_truth=gt,
                verifier_type=verifier_type
            )
            consensus_winner_correct = v_consensus.correct
        else:
            # No valid candidate answer parsed from any sample
            consensus_winner_correct = False
            winning_answer = None
            agreement_count = 0

        if consensus_winner_correct:
            total_consensus_correct += 1

        has_at_least_one_correct = (correct_in_k > 0)
        if has_at_least_one_correct:
            total_pass_at_k_correct += 1

        agreement_ratio = agreement_count / k_samples

        detailed_results.append({
            "id": record["id"],
            "problem": record["problem"],
            "final_answer": gt,
            "winning_answer": winning_answer,
            "consensus_correct": consensus_winner_correct,
            "agreement_count": agreement_count,
            "agreement_ratio": agreement_ratio,
            "correct_in_k_count": correct_in_k,
            "pass_at_k": has_at_least_one_correct,
            "samples": sample_details
        })

    num_problems = len(records)
    summary = {
        "num_problems": num_problems,
        "k_samples": k_samples,
        "temperature": temperature,
        "top_p": top_p,
        "pass_at_1_accuracy": total_individual_correct / max(total_samples_evaluated, 1),
        "consensus_accuracy": total_consensus_correct / max(num_problems, 1),
        "pass_at_k_accuracy": total_pass_at_k_correct / max(num_problems, 1),
        "mean_agreement_ratio": sum(r["agreement_ratio"] for r in detailed_results) / max(num_problems, 1),
        "mean_tokens_per_problem": total_tokens_generated / max(num_problems, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": output_prefix
    }

    # Save results
    detailed_file = os.path.join(output_dir, f"{output_prefix}_detailed.json")
    summary_file = os.path.join(output_dir, f"{output_prefix}_summary.json")

    with open(detailed_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2)

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Self-Consistency Results (K={k_samples}):")
    logger.info(f"  Single-Sample Pass@1: {summary['pass_at_1_accuracy']:.2%}")
    logger.info(f"  Majority Vote (Cons@{k_samples}): {summary['consensus_accuracy']:.2%}")
    logger.info(f"  Upper Bound (Pass@{k_samples}): {summary['pass_at_k_accuracy']:.2%}")
    logger.info(f"  Mean Agreement Ratio: {summary['mean_agreement_ratio']:.2%}")

    return summary

