"""Unit tests for symbolic algebraic verifier."""
import pytest
from src.verifiers.symbolic import SymbolicVerifier

@pytest.fixture
def verifier():
    return SymbolicVerifier()

def test_symbolic_identity(verifier):
    pred = "<answer>2*x + 3</answer>"
    res = verifier.verify(pred, "2*x + 3")
    assert res.correct is True
    assert res.reward == 1.0

def test_symbolic_algebraic_equivalence(verifier):
    pred = "<answer>(x + 1)*(x - 1)</answer>"
    res = verifier.verify(pred, "x**2 - 1")
    assert res.correct is True
    assert res.reward == 1.0

def test_symbolic_latex_fraction(verifier):
    pred = "<answer>\\frac{x}{2}</answer>"
    res = verifier.verify(pred, "0.5*x")
    assert res.correct is True
    assert res.reward == 1.0

def test_symbolic_inequality(verifier):
    pred = "<answer>x**2 + 1</answer>"
    res = verifier.verify(pred, "x**2 - 1")
    assert res.correct is False
    assert res.reward == 0.0
