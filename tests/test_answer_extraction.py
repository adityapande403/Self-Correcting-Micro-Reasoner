"""Tests for answer tag and reasoning extraction."""
import pytest
from src.verifiers.base import extract_answer_tag, extract_think_tag

def test_extract_clean_answer_tag():
    text = "<think>Calculations here</think><answer>42</answer>"
    res = extract_answer_tag(text)
    assert res["status"] == "success"
    assert res["answer"] == "42"

def test_extract_think_tag():
    text = "<think>We need to multiply 6 and 7.</think><answer>42</answer>"
    assert extract_think_tag(text) == "We need to multiply 6 and 7."

def test_extract_boxed():
    text = "The result after simplifying is \\boxed{15}."
    res = extract_answer_tag(text)
    assert res["status"] == "success"
    assert res["answer"] == "15"

def test_extract_boxed_nested_fraction():
    text = "Therefore, x = \\boxed{\\frac{3}{4}}."
    res = extract_answer_tag(text)
    assert res["status"] == "success"
    assert res["answer"] == "\\frac{3}{4}"

def test_extract_header_prefix():
    text = "Step 1: 5 + 5 = 10\n### Answer: 10"
    res = extract_answer_tag(text)
    assert res["status"] == "success"
    assert res["answer"] == "10"

def test_missing_tags():
    text = "I am calculating 5 + 5 which is 10, but I forgot the format."
    res = extract_answer_tag(text)
    # Natural language phrase may catch or it returns no_answer_tag
    assert res["status"] in ("no_answer_tag", "success")

def test_multiple_answer_tags_takes_last():
    text = "First attempt: <answer>10</answer> Wait, that's wrong! <answer>20</answer>"
    res = extract_answer_tag(text)
    assert res["status"] == "success"
    assert res["answer"] == "20"

def test_empty_input():
    res = extract_answer_tag("")
    assert res["status"] == "unparseable"

def test_empty_answer_tag():
    res = extract_answer_tag("<answer>   </answer>")
    assert res["status"] == "unparseable"
