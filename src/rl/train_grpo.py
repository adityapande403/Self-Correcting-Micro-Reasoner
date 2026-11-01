"""GRPO training pipeline using TRL GRPOTrainer and deterministic reward functions."""
import os
import json
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, TaskType
from src.utils.config import load_yaml
from src.utils.logging import setup_logger
from src.utils.memory import MemoryTracker
from src.models.load_model import load_tokenizer, load_base_model
from src.rl.rollout_manager import math_accuracy_reward_func, reasoning_format_reward_func

logger = setup_logger("train_grpo")

def prepare_grpo_dataset(file_path: str, tokenizer, max_samples: int = None) -> Dataset:
    """Loads JSONL records and creates prompt and ground-truth columns for GRPOTrainer."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if max_samples:
        records = records[:max_samples]

    formatted = []
    for r in records:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful mathematical reasoning assistant. "
                    "First, think through the problem step by step inside <think>...</think> tags. "
                    "Then, write your final answer inside <answer>...</answer> tags."
                )
            },
            {"role": "user", "content": r["problem"]}
        ]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted.append({
            "prompt": prompt_text,
            "final_answer": r["final_answer"],
            "verifier_type": r.get("verifier_type", "numeric"),
            "problem": r["problem"]
        })

    logger.info(f"Loaded {len(formatted)} examples for GRPO training from {file_path}")
    return Dataset.from_list(formatted)

def train_grpo(
    config_path: str = "configs/grpo.yaml",
    model_config_path: str = "configs/model.yaml",
    sft_checkpoint_path: str = "outputs/checkpoints/sft/final",
    max_steps: int = None,
    smoke_test: bool = False
):
    """Executes GRPO training on the cold-start SFT checkpoint."""
    cfg = load_yaml(config_path)
    model_cfg = load_yaml(model_config_path)
    training_args_dict = cfg.get("training", {})

    output_dir = cfg.get("output_dir", "outputs/checkpoints/grpo")
    os.makedirs(output_dir, exist_ok=True)

    base_model_name = model_cfg.get("model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer = load_tokenizer(base_model_name)

    # Determine starting model: use SFT checkpoint if available, otherwise base model
    start_model_path = sft_checkpoint_path if (sft_checkpoint_path and os.path.exists(sft_checkpoint_path)) else base_model_name
    logger.info(f"Initializing GRPO policy from: {start_model_path}")

    base_model = load_base_model(
        model_name_or_path=base_model_name,
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        device_map=model_cfg.get("device_map", "auto"),
        use_gradient_checkpointing=True
    )

    if start_model_path != base_model_name and os.path.exists(start_model_path):
        from peft import PeftModel
        logger.info(f"Merging SFT adapter from {start_model_path} into base model for GRPO...")
        sft_peft = PeftModel.from_pretrained(base_model, start_model_path)
        model = sft_peft.merge_and_unload()
    else:
        model = base_model

    # Attach LoRA adapter
    lora_dict = model_cfg.get("lora", {})
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_dict.get("r", 16),
        lora_alpha=lora_dict.get("lora_alpha", 32),
        lora_dropout=lora_dict.get("lora_dropout", 0.05),
        target_modules=lora_dict.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        bias="none"
    )

    # Datasets
    train_file = cfg.get("dataset_path", "data/splits/grpo_train.jsonl")
    val_file = cfg.get("val_dataset_path", "data/splits/grpo_val.jsonl")

    max_samples = 20 if smoke_test else None
    train_dataset = prepare_grpo_dataset(train_file, tokenizer, max_samples=max_samples)
    eval_dataset = prepare_grpo_dataset(val_file, tokenizer, max_samples=8 if smoke_test else 40)

    steps = 4 if smoke_test else (max_steps or training_args_dict.get("max_steps", 60))
    num_gens = 2 if smoke_test else training_args_dict.get("num_generations", 4)

    grpo_args = GRPOConfig(
        output_dir=output_dir,
        max_steps=steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=training_args_dict.get("gradient_accumulation_steps", 8 if not smoke_test else 2),
        learning_rate=float(training_args_dict.get("learning_rate", 1e-5)),
        num_generations=num_gens,
        max_completion_length=training_args_dict.get("max_completion_length", 512),
        temperature=float(training_args_dict.get("temperature", 0.7)),
        top_p=float(training_args_dict.get("top_p", 0.9)),
        beta=float(training_args_dict.get("beta", 0.04)),
        warmup_steps=max(1, int(steps * 0.05)),
        logging_steps=training_args_dict.get("logging_steps", 5 if not smoke_test else 1),
        save_steps=training_args_dict.get("save_steps", 30 if not smoke_test else 2),
        save_total_limit=2,
        seed=training_args_dict.get("seed", 42),
        bf16=True,
        gradient_checkpointing=True,
        report_to="none"
    )

    # Initialize GRPOTrainer with multi-component verifiable reward
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[math_accuracy_reward_func, reasoning_format_reward_func],
        args=grpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config
    )

    tracker = MemoryTracker()
    tracker.reset()
    logger.info("Starting GRPO training...")
    trainer.train()

    tracker.print_summary(prefix="GRPO Training Finished")

    final_save_path = os.path.join(output_dir, "final")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    logger.info(f"Saved final GRPO model and tokenizer to: {final_save_path}")

    # Verify reloadability
    logger.info("Verifying checkpoint reloadability...")
    from peft import PeftModel
    reloaded_base = load_base_model(base_model_name)
    reloaded_peft = PeftModel.from_pretrained(reloaded_base, final_save_path)
    reloaded_peft.eval()
    logger.info("Successfully verified reload of GRPO checkpoint!")
    return final_save_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grpo.yaml")
    parser.add_argument("--sft-checkpoint", type=str, default="outputs/checkpoints/sft/final")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    train_grpo(
        config_path=args.config,
        sft_checkpoint_path=args.sft_checkpoint,
        smoke_test=args.smoke_test,
        max_steps=args.max_steps
    )
