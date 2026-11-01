"""Reward functions and rollout management for GRPO training."""
import re
from typing import List, Dict, Any, Union
from src.verifiers import verify_solution
from src.verifiers.base import extract_answer_tag
from src.rl.reward_logger import RewardLogger
from src.utils.logging import setup_logger

logger = setup_logger("rollout_manager")

# Global logger instance for training session
active_reward_logger = RewardLogger()

def extract_completion_text(completion: Union[str, List[Dict[str, str]]]) -> str:
    """Normalizes completion from string or message dict list."""
    if isinstance(completion, list):
        if completion and isinstance(completion[0], dict) and "content" in completion[0]:
            return completion[0]["content"]
        elif completion and isinstance(completion[-1], dict) and "content" in completion[-1]:
            return completion[-1]["content"]
    return str(completion)

def math_accuracy_reward_func(
    completions: List[Any],
    final_answer: List[str],
    verifier_type: List[str] = None,
    **kwargs
) -> List[float]:
    """
    Deterministic mathematical accuracy reward function for GRPOTrainer.
    Returns 1.0 for correct final answer, 0.0 otherwise.
    """
    rewards = []
    v_types = verifier_type if verifier_type is not None else ["numeric"] * len(completions)

    for i, (comp, gold, v_type) in enumerate(zip(completions, final_answer, v_types)):
        text = extract_completion_text(comp)
        v_res = verify_solution(
            prediction=text,
            ground_truth=gold,
            verifier_type=v_type
        )
        rewards.append(float(v_res.reward))

    return rewards

def reasoning_format_reward_func(
    completions: List[Any],
    **kwargs
) -> List[float]:
    """
    Format compliance reward: awards a small bounded reward (0.1) if both <think>
    and <answer> tags are present with non-empty content.
    Prevents degenerate completions while keeping accuracy as the dominant signal (1.0).
    """
    rewards = []
    for comp in completions:
        text = extract_completion_text(comp)
        has_think = bool(re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE))
        has_answer = bool(re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE))
        
        if has_think and has_answer:
            rewards.append(0.1)
        elif has_answer:
            rewards.append(0.05)
        else:
            rewards.append(0.0)
    return rewards
