"""Supervised Fine-Tuning (SFT) implementation for format and reasoning cold-start."""
import os
import json
import torch
from datasets import Dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType
from src.utils.config import load_yaml
from src.utils.logging import setup_logger
from src.utils.memory import MemoryTracker
from src.models.load_model import load_tokenizer, load_base_model

logger = setup_logger("train_sft")

def prepare_sft_dataset(file_path: str, tokenizer, max_samples: int = None) -> Dataset:
    """Loads JSONL records and converts them to formatted chat messages."""
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
            {"role": "user", "content": r["problem"]},
            {"role": "assistant", "content": r["solution"]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        formatted.append({"text": text, "problem": r["problem"], "final_answer": r["final_answer"]})

    logger.info(f"Loaded {len(formatted)} formatted training examples from {file_path}")
    return Dataset.from_list(formatted)

def train_sft(
    config_path: str = "configs/sft.yaml",
    model_config_path: str = "configs/model.yaml",
    max_steps: int = None,
    smoke_test: bool = False
):
    """Executes SFT training with LoRA and gradient checkpointing."""
    cfg = load_yaml(config_path)
    model_cfg = load_yaml(model_config_path)
    training_args_dict = cfg.get("training", {})

    output_dir = cfg.get("output_dir", "outputs/checkpoints/sft")
    os.makedirs(output_dir, exist_ok=True)

    base_model_name = model_cfg.get("model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer = load_tokenizer(base_model_name)
    model = load_base_model(
        model_name_or_path=base_model_name,
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        device_map=model_cfg.get("device_map", "auto"),
        use_gradient_checkpointing=True
    )

    # Dataset
    train_file = cfg.get("dataset_path", "data/splits/sft_train.jsonl")
    val_file = cfg.get("val_dataset_path", "data/splits/sft_val.jsonl")

    max_samples = 40 if smoke_test else None
    train_dataset = prepare_sft_dataset(train_file, tokenizer, max_samples=max_samples)
    eval_dataset = prepare_sft_dataset(val_file, tokenizer, max_samples=10 if smoke_test else 50)

    # LoRA config
    lora_dict = model_cfg.get("lora", {})
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_dict.get("r", 16),
        lora_alpha=lora_dict.get("lora_alpha", 32),
        lora_dropout=lora_dict.get("lora_dropout", 0.05),
        target_modules=lora_dict.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        bias="none"
    )

    steps = 5 if smoke_test else (max_steps or training_args_dict.get("max_steps", 200))

    sft_args = SFTConfig(
        output_dir=output_dir,
        max_steps=steps,
        per_device_train_batch_size=training_args_dict.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=training_args_dict.get("gradient_accumulation_steps", 4),
        learning_rate=float(training_args_dict.get("learning_rate", 2e-4)),
        lr_scheduler_type=training_args_dict.get("lr_scheduler_type", "cosine"),
        warmup_steps=max(1, int(steps * 0.05)),
        weight_decay=float(training_args_dict.get("weight_decay", 0.01)),
        logging_steps=training_args_dict.get("logging_steps", 10 if not smoke_test else 1),
        save_steps=training_args_dict.get("save_steps", 100 if not smoke_test else 5),
        eval_strategy="steps",
        eval_steps=training_args_dict.get("eval_steps", 100 if not smoke_test else 5),
        save_total_limit=training_args_dict.get("save_total_limit", 2),
        seed=training_args_dict.get("seed", 42),
        bf16=True,
        max_length=model_cfg.get("max_seq_length", 1024),
        dataset_text_field="text",
        packing=False,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config
    )

    tracker = MemoryTracker()
    tracker.reset()
    logger.info("Starting SFT training...")
    trainer.train()

    tracker.print_summary(prefix="SFT Training Finished")

    # Save final checkpoint
    final_save_path = os.path.join(output_dir, "final")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    logger.info(f"Saved final SFT model and tokenizer to: {final_save_path}")

    # Verify reloadability
    logger.info("Verifying checkpoint reloadability...")
    from peft import PeftModel
    reloaded_base = load_base_model(base_model_name)
    reloaded_peft = PeftModel.from_pretrained(reloaded_base, final_save_path)
    reloaded_peft.eval()
    logger.info("Successfully verified reload of SFT checkpoint!")
    return final_save_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sft.yaml")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    train_sft(config_path=args.config, smoke_test=args.smoke_test, max_steps=args.max_steps)
