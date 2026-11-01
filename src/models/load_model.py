"""Model and tokenizer initialization utilities with LoRA and PEFT support."""
import os
import torch
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from src.utils.logging import setup_logger
from src.utils.memory import MemoryTracker

logger = setup_logger("load_model")

def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizer:
    """Loads and configures the tokenizer."""
    logger.info(f"Loading tokenizer from: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # For batched decoder generation
    return tokenizer

def load_base_model(
    model_name_or_path: str,
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    use_gradient_checkpointing: bool = True
) -> PreTrainedModel:
    """Loads the base causal language model onto the device."""
    logger.info(f"Loading base model: {model_name_or_path} (dtype={torch_dtype}, device_map={device_map})")
    tracker = MemoryTracker()
    tracker.reset()

    dtype = torch.bfloat16 if torch_dtype == "bfloat16" else (torch.float16 if torch_dtype == "float16" else torch.float32)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    if use_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    tracker.print_summary(prefix="Model Loaded")
    return model

def apply_lora(
    model: PreTrainedModel,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None
) -> PeftModel:
    """Wraps the base model with LoRA adapters."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    logger.info(f"Applying LoRA (r={r}, alpha={lora_alpha}, dropout={lora_dropout}, targets={target_modules})")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

def load_peft_checkpoint(
    base_model_name: str,
    adapter_path: str,
    sft_adapter_path: Optional[str] = None
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Loads a base model and attaches a saved LoRA adapter checkpoint, merging prior SFT weights if needed."""
    logger.info(f"Loading base model '{base_model_name}' with adapter '{adapter_path}'")
    tokenizer = load_tokenizer(adapter_path if os.path.exists(os.path.join(adapter_path, "tokenizer_config.json")) else base_model_name)
    base_model = load_base_model(base_model_name)

    # If this is a GRPO checkpoint, merge the prior SFT checkpoint first
    sft_path = sft_adapter_path or ("outputs/checkpoints/sft/final" if "grpo" in adapter_path.lower() else None)
    if sft_path and os.path.exists(sft_path):
        logger.info(f"Merging intermediate SFT adapter from {sft_path}...")
        sft_peft = PeftModel.from_pretrained(base_model, sft_path)
        base_model = sft_peft.merge_and_unload()

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer
