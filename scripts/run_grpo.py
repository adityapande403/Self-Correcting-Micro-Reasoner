"""CLI runner for GRPO training."""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.train_grpo import train_grpo
from src.utils.logging import setup_logger

logger = setup_logger("run_grpo")

def main():
    parser = argparse.ArgumentParser(description="Run GRPO Training with Verifiable Rewards")
    parser.add_argument("--config", type=str, default="configs/grpo.yaml", help="Path to GRPO config")
    parser.add_argument("--sft-checkpoint", type=str, default="outputs/checkpoints/sft/final", help="Path to SFT model")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 4-step rollout smoke test")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("STAGE D: GRPO REINFORCEMENT LEARNING WITH RLVR")
    logger.info("==================================================")

    train_grpo(
        config_path=args.config,
        sft_checkpoint_path=args.sft_checkpoint,
        smoke_test=args.smoke_test,
        max_steps=args.max_steps
    )

if __name__ == "__main__":
    main()
