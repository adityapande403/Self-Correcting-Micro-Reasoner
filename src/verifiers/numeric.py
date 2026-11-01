"""Deterministic numeric and rational number verifier."""
import re
from fractions import Fraction
from typing import Optional, Tuple
from src.verifiers.base import BaseVerifier, VerificationResult, extract_answer_tag

def normalize_numeric_string(raw: str) -> Optional[float | Fraction]:
    """
    Normalizes numbers, currency, commas, percentages, and LaTeX fractions.
    Returns a Python float or Fraction if valid, else None.
    """
    if not raw or not isinstance(raw, str):
        return None

    cleaned = raw.strip()
    
    # Strip enclosing LaTeX delimiters: \( ... \), $ ... $, [ ... ]
    cleaned = re.sub(r"^\$+(.*?)\$+$", r"\1", cleaned).strip()
    cleaned = re.sub(r"^\\\((.*?)\\\)$", r"\1", cleaned).strip()
    cleaned = re.sub(r"^\\\[(.*?)\\\]$", r"\1", cleaned).strip()

    # Strip currency symbols and common units
    cleaned = re.sub(r"^[\$€£¥₹]\s*", "", cleaned).strip()
    cleaned = re.sub(r"\s*(?:dollars?|cents?|km|m|cm|mm|kg|g|lbs?|hours?|hrs?|minutes?|mins?|seconds?|secs?|USD)$", "", cleaned, flags=re.IGNORECASE).strip()

    # Handle percentage: e.g. "25%" -> 0.25 (or 25 depending on prompt convention, but keep Fraction if fractional)
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1].strip()

    # Remove thousands commas: "1,250" -> "1250"
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", cleaned)

    # Check LaTeX fraction: \frac{a}{b} or \dfrac{a}{b}
    frac_match = re.search(r"\\(?:d)?frac\{([^{}]+)\}\{([^{}]+)\}", cleaned)
    if frac_match:
        num_str = frac_match.group(1).strip()
        den_str = frac_match.group(2).strip()
        try:
            num = float(num_str)
            den = float(den_str)
            if den != 0:
                val = Fraction(int(num), int(den)) if num.is_integer() and den.is_integer() else (num / den)
                return val * Fraction(1, 100) if is_percent and isinstance(val, Fraction) else (val / 100.0 if is_percent else val)
        except (ValueError, ZeroDivisionError):
            pass

    # Check standard fraction: a/b
    if "/" in cleaned:
        parts = cleaned.split("/")
        if len(parts) == 2:
            try:
                num = int(parts[0].strip())
                den = int(parts[1].strip())
                if den != 0:
                    val = Fraction(num, den)
                    return val * Fraction(1, 100) if is_percent else val
            except ValueError:
                pass

    # Standard float / int parsing
    try:
        val = float(cleaned)
        if is_percent:
            val = val / 100.0
        return val
    except ValueError:
        return None

def are_numbers_equivalent(val1: float | Fraction, val2: float | Fraction, tolerance: float = 1e-4) -> bool:
    """Checks equality between two numerical representations."""
    if isinstance(val1, Fraction) and isinstance(val2, Fraction):
        return val1 == val2

    # Float comparison with relative/absolute tolerance
    f1 = float(val1)
    f2 = float(val2)
    return abs(f1 - f2) <= max(tolerance, tolerance * max(abs(f1), abs(f2)))

class NumericVerifier(BaseVerifier):
    """Deterministic verifier for numeric and fractional answers."""
    def __init__(self, tolerance: float = 1e-4):
        self.tolerance = tolerance

    def verify(self, prediction: str, ground_truth: str) -> VerificationResult:
        # Extract prediction answer
        extraction = extract_answer_tag(prediction)
        if extraction["status"] != "success":
            return VerificationResult(
                correct=False,
                reward=0.0,
                verifier="numeric",
                parse_status=extraction["status"],
                failure_reason=extraction.get("reason", "failed_to_extract_answer"),
                extracted_answer=None,
                gold_answer=ground_truth
            )

        extracted_raw = extraction["answer"]

        # Also extract ground truth if gold is embedded in tags
        gold_extraction = extract_answer_tag(ground_truth)
        gold_raw = gold_extraction["answer"] if gold_extraction["status"] == "success" else ground_truth

        pred_val = normalize_numeric_string(extracted_raw)
        gold_val = normalize_numeric_string(gold_raw)

        if gold_val is None:
            # Fallback string comparison if gold isn't pure numeric
            is_match = extracted_raw.strip().lower() == gold_raw.strip().lower()
            return VerificationResult(
                correct=is_match,
                reward=1.0 if is_match else 0.0,
                verifier="numeric_str_fallback",
                parse_status="success",
                failure_reason=None if is_match else "string_mismatch",
                extracted_answer=extracted_raw,
                gold_answer=gold_raw
            )

        if pred_val is None:
            return VerificationResult(
                correct=False,
                reward=0.0,
                verifier="numeric",
                parse_status="unparseable_numeric",
                failure_reason=f"could_not_parse_numeric: '{extracted_raw}'",
                extracted_answer=extracted_raw,
                gold_answer=gold_raw
            )

        # Compare values
        match = are_numbers_equivalent(pred_val, gold_val, tolerance=self.tolerance)

        return VerificationResult(
            correct=match,
            reward=1.0 if match else 0.0,
            verifier="numeric",
            parse_status="success",
            failure_reason=None if match else f"value_mismatch: {pred_val} != {gold_val}",
            extracted_answer=extracted_raw,
            gold_answer=gold_raw,
            metadata={"parsed_pred": float(pred_val), "parsed_gold": float(gold_val)}
        )
