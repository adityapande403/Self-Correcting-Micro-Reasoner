"""Unified verifier interface."""
from src.verifiers.base import BaseVerifier, VerificationResult, extract_answer_tag, extract_think_tag
from src.verifiers.numeric import NumericVerifier
from src.verifiers.symbolic import SymbolicVerifier
from src.verifiers.sql import SQLVerifier

_numeric_verifier = NumericVerifier()
_symbolic_verifier = SymbolicVerifier()
_sql_verifier = SQLVerifier()

def verify_solution(
    prediction: str,
    ground_truth: str,
    verifier_type: str = "numeric",
    **kwargs
) -> VerificationResult:
    """Dispatches verification to the designated verifier."""
    verifier_type = (verifier_type or "numeric").lower()
    if verifier_type in ("numeric", "math", "arithmetic", "numeric_exact"):
        return _numeric_verifier.verify(prediction, ground_truth)
    elif verifier_type in ("symbolic", "algebra"):
        return _symbolic_verifier.verify(prediction, ground_truth)
    elif verifier_type in ("sql", "sqlite"):
        schema = kwargs.get("schema", "")
        return _sql_verifier.verify(prediction, ground_truth, schema=schema)
    else:
        # Default to numeric with string fallback
        return _numeric_verifier.verify(prediction, ground_truth)
