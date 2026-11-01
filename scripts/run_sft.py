"""CLI runner for Supervised Fine-Tuning."""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sft.train_sft import train_sft
from src.utils.logging import setup_logger

logger = setup_logger("run_sft")

def main():
    parser = argparse.ArgumentParser(description="Run SFT Training")
    parser.add_argument("--config", type=str, default="configs/sft.yaml", help="Path to SFT config")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="Path to model config")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick 5-step smoke test")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max training steps")
    args = parser.parse_args()

    logger.info("==================================================")
    logger.info("STAGE B: SUPERVISED FINE-TUNING (SFT COLD-START)")
    logger.info("==================================================")

    train_sft(
        config_path=args.config,
        model_config_path=args.model_config,
        max_steps=args.max_steps,
        smoke_test=args.smoke_test
    )

if __name__ == "__main__":
    main()
