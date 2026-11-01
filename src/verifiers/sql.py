"""Deterministic and safe SQLite execution verifier."""
import re
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
from src.verifiers.base import BaseVerifier, VerificationResult, extract_answer_tag

# Disallowed dangerous SQL commands
FORBIDDEN_SQL_KEYWORDS = {
    "ATTACH", "DETACH", "PRAGMA", "LOAD_EXTENSION", "VACUUM", 
    "ALTER", "DROP", "UPDATE", "DELETE", "INSERT", "CREATE"
}

def is_safe_query(sql: str) -> Tuple[bool, Optional[str]]:
    """Checks whether the SQL query is a read-only SELECT query."""
    normalized = sql.strip().upper()
    tokens = re.findall(r"\b[A-Z_]+\b", normalized)
    for kw in FORBIDDEN_SQL_KEYWORDS:
        if kw in tokens:
            return False, f"Disallowed keyword: {kw}"
    if not normalized.startswith("SELECT") and not normalized.startswith("WITH"):
        return False, "Query must start with SELECT or WITH"
    return True, None

def execute_query(db_schema_script: str, query: str) -> Dict[str, Any]:
    """Executes a query in an isolated, in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    try:
        # Create schema and fixtures
        cursor.executescript(db_schema_script)
        # Execute query
        cursor.execute(query)
        rows = cursor.fetchall()
        description = [d[0] for d in cursor.description] if cursor.description else []
        return {"status": "success", "rows": rows, "columns": description, "error": None}
    except Exception as e:
        return {"status": "error", "rows": None, "columns": None, "error": str(e)}
    finally:
        conn.close()

def are_results_equivalent(rows1: List[Any], rows2: List[Any], ignore_order: bool = True) -> bool:
    """Compares SQL execution output rows for equivalence."""
    if rows1 is None or rows2 is None:
        return False
    if len(rows1) != len(rows2):
        return False
    
    if ignore_order:
        # Sort rows if possible (converting tuples to string representation if needed)
        try:
            return sorted(rows1) == sorted(rows2)
        except TypeError:
            return sorted(map(str, rows1)) == sorted(map(str, rows2))
    return rows1 == rows2

class SQLVerifier(BaseVerifier):
    """Safe SQLite execution verifier."""
    def __init__(self, db_schema: str = ""):
        self.db_schema = db_schema

    def verify(self, prediction: str, ground_truth: str, schema: Optional[str] = None) -> VerificationResult:
        active_schema = schema or self.db_schema
        extraction = extract_answer_tag(prediction)
        
        # If extraction failed, look for ```sql ... ``` code block
        if extraction["status"] != "success":
            code_block = re.search(r"```(?:sql)?\s*(.*?)\s*```", prediction, flags=re.DOTALL | re.IGNORECASE)
            if code_block:
                pred_sql = code_block.group(1).strip()
            else:
                return VerificationResult(
                    correct=False,
                    reward=0.0,
                    verifier="sql",
                    parse_status="unparseable",
                    failure_reason="No SQL query or answer tags extracted"
                )
        else:
            pred_sql = extraction["answer"].strip()

        # Extract gold SQL or expected output
        gold_sql = ground_truth.strip()

        safe, reason = is_safe_query(pred_sql)
        if not safe:
            return VerificationResult(
                correct=False,
                reward=0.0,
                verifier="sql",
                parse_status="unsafe_query",
                failure_reason=reason,
                extracted_answer=pred_sql
            )

        # Run prediction query
        pred_res = execute_query(active_schema, pred_sql)
        if pred_res["status"] != "success":
            return VerificationResult(
                correct=False,
                reward=0.0,
                verifier="sql",
                parse_status="execution_error",
                failure_reason=f"Execution failed: {pred_res['error']}",
                extracted_answer=pred_sql
            )

        # Run reference query
        gold_res = execute_query(active_schema, gold_sql)
        if gold_res["status"] != "success":
            return VerificationResult(
                correct=False,
                reward=0.0,
                verifier="sql",
                parse_status="gold_execution_error",
                failure_reason=f"Gold execution failed: {gold_res['error']}",
                extracted_answer=pred_sql
            )

        is_match = are_results_equivalent(pred_res["rows"], gold_res["rows"])
        return VerificationResult(
            correct=is_match,
            reward=1.0 if is_match else 0.0,
            verifier="sql",
            parse_status="success",
            failure_reason=None if is_match else "Query results mismatch",
            extracted_answer=pred_sql,
            gold_answer=gold_sql,
            metadata={"pred_rows": pred_res["rows"], "gold_rows": gold_res["rows"]}
        )
