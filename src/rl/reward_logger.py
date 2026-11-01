"""Rollout reward tracking, statistics, and zero-variance group logging."""
import numpy as np
from typing import List, Dict, Any, Optional

class RewardLogger:
    """Logs and tracks group-level rewards, advantages, and verifier diagnostics."""
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.zero_variance_groups = 0
        self.total_groups = 0
        self.positive_rewards = 0
        self.total_rollouts = 0
        self.invalid_outputs = 0

    def record_group(
        self,
        prompt: str,
        completions: List[str],
        rewards: List[float],
        verification_details: List[Dict[str, Any]]
    ):
        """Records a single prompt's rollout group."""
        group_size = len(rewards)
        r_arr = np.array(rewards, dtype=np.float32)
        mean_r = float(np.mean(r_arr))
        std_r = float(np.std(r_arr))

        self.total_groups += 1
        self.total_rollouts += group_size
        self.positive_rewards += int(np.sum(r_arr > 0.0))

        # Check zero-variance
        is_zero_var = (std_r < 1e-6)
        if is_zero_var:
            self.zero_variance_groups += 1

        for v in verification_details:
            if v.get("parse_status") != "success":
                self.invalid_outputs += 1

        self.history.append({
            "prompt": prompt,
            "rewards": rewards,
            "mean_reward": mean_r,
            "std_reward": std_r,
            "is_zero_variance": is_zero_var,
            "completions": completions,
            "diagnostics": verification_details
        })

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates running statistics across all groups."""
        if not self.history:
            return {}

        all_means = [g["mean_reward"] for g in self.history]
        all_stds = [g["std_reward"] for g in self.history]

        return {
            "total_groups": self.total_groups,
            "total_rollouts": self.total_rollouts,
            "overall_mean_reward": float(np.mean(all_means)),
            "overall_std_reward": float(np.mean(all_stds)),
            "positive_reward_fraction": float(self.positive_rewards / max(1, self.total_rollouts)),
            "invalid_output_fraction": float(self.invalid_outputs / max(1, self.total_rollouts)),
            "zero_variance_group_fraction": float(self.zero_variance_groups / max(1, self.total_groups)),
            "zero_variance_groups_count": self.zero_variance_groups
        }
