"""Base abstractions for verifiable deterministic rewards."""
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class VerificationResult(BaseModel):
    correct: bool = Field(default=False)
    reward: float = Field(default=0.0)
    verifier: str = Field(default="base")
    parse_status: str = Field(default="unknown")  # "success", "unparseable", "no_answer_tag", "multiple_answers"
    failure_reason: Optional[str] = None
    extracted_answer: Optional[str] = None
    gold_answer: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BaseVerifier(ABC):
    """Abstract verifier interface."""
    @abstractmethod
    def verify(self, prediction: str, ground_truth: str) -> VerificationResult:
        """Verifies if the prediction matches the ground truth deterministically."""
        pass

def extract_answer_tag(text: str) -> Dict[str, Any]:
    """
    Extracts the answer from structured text.
    Prioritizes:
    1. <answer>(.*?)</answer>
    2. \\boxed{(.*?)}
    3. '### Final Answer:' or '### Answer:'
    4. Last line if explicitly formatted
    """
    if not text or not isinstance(text, str):
        return {"status": "unparseable", "answer": None, "reason": "empty_or_non_string_input"}

    # 1. Look for <answer>...</answer> tags
    answer_matches = list(re.finditer(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE))
    if len(answer_matches) == 1:
        extracted = answer_matches[0].group(1).strip()
        if extracted:
            return {"status": "success", "answer": extracted, "tag_type": "answer_tag"}
        return {"status": "unparseable", "answer": None, "reason": "empty_answer_tag"}
    elif len(answer_matches) > 1:
        # Multiple answer tags: take the last one or flag warning
        extracted = answer_matches[-1].group(1).strip()
        return {"status": "success", "answer": extracted, "tag_type": "multiple_answer_tags_used_last"}

    # 2. Look for \boxed{...} (handling nested braces up to 3 levels)
    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*)\}"
    boxed_matches = list(re.finditer(boxed_pattern, text))
    if boxed_matches:
        extracted = boxed_matches[-1].group(1).strip()
        if extracted:
            return {"status": "success", "answer": extracted, "tag_type": "boxed"}

    # 3. Look for '### Answer:' or '### Final Answer:'
    answer_prefix_match = re.search(r"###\s*(?:Final\s+)?Answer\s*:\s*([^\n\r]+)", text, flags=re.IGNORECASE)
    if answer_prefix_match:
        extracted = answer_prefix_match.group(1).strip()
        # Clean trailing period if numeric
        if extracted:
            return {"status": "success", "answer": extracted.rstrip("."), "tag_type": "prefix_header"}

    # 4. Look for 'The answer is ...' at end of sentence
    the_answer_match = re.search(r"[Tt]he (?:final )?answer is\s*:?\s*(\$?[^\n\r.]+\$?)", text)
    if the_answer_match:
        extracted = the_answer_match.group(1).strip().strip("$").rstrip(".")
        if extracted:
            return {"status": "success", "answer": extracted, "tag_type": "natural_language_phrase"}

    return {"status": "no_answer_tag", "answer": None, "reason": "no_verifiable_answer_anchor_found"}

def extract_think_tag(text: str) -> Optional[str]:
    """Extracts internal reasoning from <think>...</think> tags if present."""
    if not text:
        return None
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None
