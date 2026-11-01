"""Inference engine with latency, token count, and peak VRAM measurement."""
import time
import torch
from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.utils.memory import MemoryTracker
from src.utils.logging import setup_logger

logger = setup_logger("inference")

class InferenceEngine:
    """Handles text generation and records execution diagnostics."""
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device
        self.tracker = MemoryTracker(device=str(self.device))

    def format_prompt(self, problem: str, system_prompt: Optional[str] = None) -> str:
        """Formats the problem using the tokenizer chat template if available, or structured prompt."""
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({
                    "role": "system",
                    "content": (
                        "You are a careful mathematical reasoning assistant. "
                        "First, think through the problem step by step inside <think>...</think> tags. "
                        "Then, write your final answer inside <answer>...</answer> tags."
                    )
                })
            messages.append({"role": "user", "content": problem})
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return f"Problem: {problem}\nAnswer:"

    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        do_sample: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates completions for a batch of prompts and tracks performance.
        Returns a list of dicts with 'completion', 'new_tokens', 'latency_sec', 'tokens_per_sec', 'peak_vram_mb'.
        """
        if do_sample is None:
            do_sample = temperature > 0.0

        self.tracker.reset()
        start_time = time.perf_counter()

        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536
        ).to(self.device)

        prompt_lengths = [len(ids) for ids in encoded.input_ids]

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = max(temperature, 1e-4)
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                **gen_kwargs
            )

        elapsed = time.perf_counter() - start_time
        peak_vram = self.tracker.get_stats().get("gpu_peak_allocated_mb", 0.0)

        results = []
        for i, out_ids in enumerate(outputs):
            # Extract newly generated tokens only
            input_len = prompt_lengths[i]
            new_ids = out_ids[input_len:]
            gen_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            num_tokens = len(new_ids)
            tps = num_tokens / max(elapsed, 1e-6)

            results.append({
                "prompt": prompts[i],
                "completion": gen_text,
                "input_tokens": input_len,
                "output_tokens": num_tokens,
                "latency_sec": elapsed / len(prompts),
                "tokens_per_sec": tps,
                "peak_vram_mb": peak_vram
            })

        return results
