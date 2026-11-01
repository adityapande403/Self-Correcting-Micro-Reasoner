"""Configuration loader and schema validation."""
import os
import yaml
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    model_name_or_path: str = Field(default="Qwen/Qwen2.5-1.5B-Instruct")
    tokenizer_name_or_path: Optional[str] = None
    torch_dtype: str = Field(default="bfloat16")
    device_map: str = Field(default="auto")
    max_seq_length: int = Field(default=2048)
    use_gradient_checkpointing: bool = Field(default=True)
    load_in_4bit: bool = Field(default=False)

class LoRAConfig(BaseModel):
    enabled: bool = Field(default=True)
    r: int = Field(default=16)
    lora_alpha: int = Field(default=32)
    lora_dropout: float = Field(default=0.05)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    bias: str = Field(default="none")
    task_type: str = Field(default="CAUSAL_LM")

class SFTConfig(BaseModel):
    dataset_path: str = Field(default="data/splits/sft_train.jsonl")
    val_dataset_path: str = Field(default="data/splits/sft_val.jsonl")
    output_dir: str = Field(default="outputs/checkpoints/sft")
    num_train_epochs: int = Field(default=1)
    max_steps: int = Field(default=200)
    per_device_train_batch_size: int = Field(default=2)
    per_device_eval_batch_size: int = Field(default=2)
    gradient_accumulation_steps: int = Field(default=4)
    learning_rate: float = Field(default=2e-4)
    lr_scheduler_type: str = Field(default="cosine")
    warmup_ratio: float = Field(default=0.05)
    weight_decay: float = Field(default=0.01)
    logging_steps: int = Field(default=10)
    save_steps: int = Field(default=100)
    eval_steps: int = Field(default=100)
    save_total_limit: int = Field(default=2)
    seed: int = Field(default=42)

class GRPOConfigModel(BaseModel):
    dataset_path: str = Field(default="data/splits/grpo_train.jsonl")
    val_dataset_path: str = Field(default="data/splits/grpo_val.jsonl")
    output_dir: str = Field(default="outputs/checkpoints/grpo")
    model_name_or_path: str = Field(default="outputs/checkpoints/sft/final")
    num_train_epochs: int = Field(default=1)
    max_steps: int = Field(default=100)
    per_device_train_batch_size: int = Field(default=1)
    gradient_accumulation_steps: int = Field(default=8)
    learning_rate: float = Field(default=1e-5)
    num_generations: int = Field(default=4)
    max_prompt_length: int = Field(default=512)
    max_completion_length: int = Field(default=512)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)
    beta: float = Field(default=0.04)
    logging_steps: int = Field(default=5)
    save_steps: int = Field(default=50)
    seed: int = Field(default=42)

class EvaluationConfig(BaseModel):
    test_dataset_path: str = Field(default="data/splits/test.jsonl")
    batch_size: int = Field(default=4)
    max_new_tokens: int = Field(default=512)
    temperature: float = Field(default=0.0)
    top_p: float = Field(default=1.0)
    output_dir: str = Field(default="outputs/reports")

def load_yaml(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def save_yaml(config_data: Dict[str, Any], output_path: str):
    """Saves a dictionary to a YAML file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f, sort_keys=False, default_flow_style=False)
