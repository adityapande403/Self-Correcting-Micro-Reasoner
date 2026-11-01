"""Deterministic symbolic verifier using SymPy for algebraic equivalence."""
import re
import sympy
from typing import Optional
from src.verifiers.base import BaseVerifier, VerificationResult, extract_answer_tag

def parse_symbolic_expression(expr_str: str) -> Optional[sympy.Expr]:
    """Parses a string into a SymPy expression."""
    if not expr_str or not isinstance(expr_str, str):
        return None

    cleaned = expr_str.strip()
    # Strip LaTeX delimiters
    cleaned = re.sub(r"^\$+(.*?)\$+$", r"\1", cleaned).strip()
    cleaned = re.sub(r"^\\\((.*?)\\\)$", r"\1", cleaned).strip()
    cleaned = re.sub(r"^\\\[(.*?)\\\]$", r"\1", cleaned).strip()

    # Convert LaTeX fractions \frac{a}{b} -> (a)/(b)
    cleaned = re.sub(r"\\(?:d)?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", cleaned)
    
    # Replace LaTeX power ^ with **
    cleaned = cleaned.replace("^", "**")
    
    # Replace \cdot or \times with *
    cleaned = cleaned.replace(r"\cdot", "*").replace(r"\times", "*")

    try:
        # Use sympy sympify with strict sandbox
        parsed = sympy.sympify(cleaned, evaluate=False)
        return parsed
    except Exception:
        return None

class SymbolicVerifier(BaseVerifier):
    """Verifies symbolic algebraic expressions using SymPy."""
    def verify(self, prediction: str, ground_truth: str) -> VerificationResult:
        extraction = extract_answer_tag(prediction)
        if extraction["status"] != "success":
            return VerificationResult(
                correct=False,
                reward=0.0,
                verifier="symbolic",
                parse_status=extraction["status"],
                failure_reason=extraction.get("reason", "failed_to_extract_answer"),
                extracted_answer=None,
                gold_answer=ground_truth
            )

        extracted_raw = extraction["answer"]
        gold_extraction = extract_answer_tag(ground_truth)
        gold_raw = gold_extraction["answer"] if gold_extraction["status"] == "success" else ground_truth

        pred_expr = parse_symbolic_expression(extracted_raw)
        gold_expr = parse_symbolic_expression(gold_raw)

        if gold_expr is None or pred_expr is None:
            # Fall back to string matching if either cannot be parsed into a symbolic AST
            is_match = extracted_raw.strip().lower() == gold_raw.strip().lower()
            return VerificationResult(
                correct=is_match,
                reward=1.0 if is_match else 0.0,
                verifier="symbolic_fallback_str",
                parse_status="partial_parse",
                failure_reason=None if is_match else "symbolic_parse_failed_and_str_mismatch",
                extracted_answer=extracted_raw,
                gold_answer=gold_raw
            )

        try:
            diff = sympy.simplify(pred_expr - gold_expr)
            is_equiv = (diff == 0)
            return VerificationResult(
                correct=is_equiv,
                reward=1.0 if is_equiv else 0.0,
                verifier="symbolic",
                parse_status="success",
                failure_reason=None if is_equiv else f"symbolic_inequality: diff={diff}",
                extracted_answer=extracted_raw,
                gold_answer=gold_raw
            )
        except Exception as e:
            # If simplification errors, fall back to equality
            is_equiv = (pred_expr == gold_expr)
            return VerificationResult(
                correct=is_equiv,
                reward=1.0 if is_equiv else 0.0,
                verifier="symbolic",
                parse_status="success",
                failure_reason=None if is_equiv else f"simplification_exception: {e}",
                extracted_answer=extracted_raw,
                gold_answer=gold_raw
            )
