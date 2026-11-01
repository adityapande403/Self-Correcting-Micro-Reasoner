"""Unit tests for the deterministic numeric verifier."""
import pytest
from src.verifiers.numeric import NumericVerifier

@pytest.fixture
def verifier():
    return NumericVerifier(tolerance=1e-4)

def test_exact_integer(verifier):
    pred = "<think>2+2=4</think><answer>4</answer>"
    res = verifier.verify(pred, "4")
    assert res.correct is True
    assert res.reward == 1.0

def test_incorrect_integer(verifier):
    pred = "<think>2+2=5</think><answer>5</answer>"
    res = verifier.verify(pred, "4")
    assert res.correct is False
    assert res.reward == 0.0

def test_negative_values(verifier):
    pred = "<answer>-15</answer>"
    res = verifier.verify(pred, "-15")
    assert res.correct is True

def test_comma_in_number(verifier):
    pred = "<answer>1,250</answer>"
    res = verifier.verify(pred, "1250")
    assert res.correct is True

def test_currency_and_units(verifier):
    pred = "<answer>$45.00</answer>"
    res = verifier.verify(pred, "45")
    assert res.correct is True

def test_decimal_tolerance(verifier):
    pred = "<answer>3.14159</answer>"
    res = verifier.verify(pred, "3.1416")
    assert res.correct is True

def test_fraction_and_latex(verifier):
    pred = "<answer>\\frac{1}{2}</answer>"
    res = verifier.verify(pred, "0.5")
    assert res.correct is True

def test_fraction_slash(verifier):
    pred = "<answer>3/4</answer>"
    res = verifier.verify(pred, "0.75")
    assert res.correct is True

def test_whitespace_tolerance(verifier):
    pred = "<answer>  84   </answer>"
    res = verifier.verify(pred, "84")
    assert res.correct is True

def test_adversarial_mention_in_text(verifier):
    # Model mentions 42 in reasoning, but puts 99 in answer tag
    pred = "<think>The gold answer is 42, but I output something else</think><answer>99</answer>"
    res = verifier.verify(pred, "42")
    assert res.correct is False
    assert res.reward == 0.0

def test_malformed_unparseable(verifier):
    pred = "<answer>not a number</answer>"
    res = verifier.verify(pred, "42")
    assert res.correct is False
    assert res.reward == 0.0
