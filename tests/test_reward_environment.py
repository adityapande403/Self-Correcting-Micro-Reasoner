"""Unit tests for unified reward environment and SQL verifier."""
import pytest
from src.verifiers import verify_solution

def test_unified_numeric_dispatch():
    res = verify_solution("<answer>42</answer>", "42", verifier_type="numeric")
    assert res.correct is True
    assert res.reward == 1.0

def test_unified_symbolic_dispatch():
    res = verify_solution("<answer>x + x</answer>", "2*x", verifier_type="symbolic")
    assert res.correct is True
    assert res.reward == 1.0

def test_sql_execution_success():
    schema = """
    CREATE TABLE employees (id INT, name TEXT, salary INT);
    INSERT INTO employees VALUES (1, 'Alice', 5000), (2, 'Bob', 7000), (3, 'Charlie', 6000);
    """
    pred = "<answer>SELECT name FROM employees WHERE salary > 5500 ORDER BY salary DESC</answer>"
    gold = "SELECT name FROM employees WHERE salary > 5500 ORDER BY salary DESC"
    res = verify_solution(pred, gold, verifier_type="sql", schema=schema)
    assert res.correct is True
    assert res.reward == 1.0

def test_sql_forbidden_keywords():
    schema = "CREATE TABLE test (id INT);"
    pred = "<answer>DROP TABLE test;</answer>"
    res = verify_solution(pred, "SELECT * FROM test", verifier_type="sql", schema=schema)
    assert res.correct is False
    assert res.reward == 0.0
    assert "Disallowed keyword" in res.failure_reason

def test_sql_syntax_error():
    schema = "CREATE TABLE test (id INT);"
    pred = "<answer>SELECT FROM test;</answer>"
    res = verify_solution(pred, "SELECT * FROM test", verifier_type="sql", schema=schema)
    assert res.correct is False
    assert res.reward == 0.0
    assert res.parse_status == "execution_error"
