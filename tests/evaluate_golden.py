"""
Golden Dataset Evaluation Script
--------------------------------
Runs all test cases from golden_dataset.json against the /answer endpoint
and generates an Excel report with detailed metrics.

Usage:
    python tests/evaluate_golden.py

Output:
    tests/evaluation_report.xlsx
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime
import requests

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas not installed. Run: pip install pandas openpyxl")
    sys.exit(1)

try:
    import openpyxl
except ImportError:
    print("ERROR: openpyxl not installed. Run: pip install openpyxl")
    sys.exit(1)


BASE_URL = "http://localhost:8000"
GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
OUTPUT_PATH = Path(__file__).parent / "evaluation_report.xlsx"


# Refusal indicators for auto-detection
REFUSAL_INDICATORS = [
    "INSUFFICIENT_CONTEXT",
    "cannot answer",
    "not found",
    "no information",
    "not contain",
    "unable to",
    "does not contain",
    "not available",
    "no relevant"
]


def load_golden_dataset() -> dict:
    """Load the golden dataset from JSON."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def call_answer_endpoint(question: str, k: int = 5) -> dict:
    """Call the /answer endpoint and return the response."""
    try:
        response = requests.post(
            f"{BASE_URL}/answer",
            json={
                "text": question,
                "access_level": "public",
                "k": k
            },
            timeout=180  # 3 min timeout for slow LLM
        )
        response.raise_for_status()
        return {
            "success": True,
            "data": response.json(),
            "request_id_header": response.headers.get("X-Request-ID")
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "data": None
        }


def is_refusal(answer: str) -> bool:
    """Check if the answer is a refusal."""
    answer_lower = answer.lower()
    return any(ind.lower() in answer_lower for ind in REFUSAL_INDICATORS)


def check_retrieval_hit(expected_sources: list, actual_sources: list) -> tuple:
    """
    Check if expected sources appear in retrieval results.
    Returns (hit: bool, rank: int or -1)
    """
    if not expected_sources:
        return True, -1  # No expected sources (refusal test)
    
    for expected in expected_sources:
        for i, actual in enumerate(actual_sources):
            if expected.lower() in actual.lower():
                return True, i + 1  # 1-indexed rank
    
    return False, -1


def evaluate_test(test: dict) -> dict:
    """Run a single test and return the evaluation result."""
    test_id = test["id"]
    question = test["question"]
    expected_sources = test.get("expected_sources", [])
    test_type = test.get("test_type", "FACTUAL")
    
    print(f"  Running {test_id}: {question[:50]}...")
    
    # Call API
    result = call_answer_endpoint(question)
    
    if not result["success"]:
        return {
            "test_id": test_id,
            "category": test["category"],
            "question": question,
            "expected_behavior": test["expected_behavior"],
            "expected_sources": ", ".join(expected_sources),
            "actual_answer": f"ERROR: {result['error']}",
            "answer_length": 0,
            "is_refusal": False,
            "citation_count": 0,
            "cited_sources": "",
            "cited_pages": "",
            "retrieval_sources": "",
            "retrieval_hit": False,
            "retrieval_rank": -1,
            "chunks_used": 0,
            "avg_retrieval_score": None,
            "retrieval_latency_ms": 0,
            "generation_latency_ms": 0,
            "total_latency_ms": 0,
            "model_name": "",
            "request_id": "",
            "pass_fail": "FAIL",
            "failure_reason": f"API Error: {result['error']}"
        }
    
    data = result["data"]
    answer = data["answer"]
    citations = data["citations"]
    metrics = data["metrics"]
    
    # Extract fields
    cited_sources = list(set(c["source"] for c in citations))
    cited_pages = [str(c.get("page_number", "N/A")) for c in citations]
    
    # Get retrieval sources from citations (approximation since we don't have raw retrieval results)
    retrieval_sources = cited_sources  # In this case, cited_sources approximates retrieval
    
    # Check retrieval hit
    hit, rank = check_retrieval_hit(expected_sources, cited_sources)
    
    # Determine pass/fail
    answer_is_refusal = is_refusal(answer)
    
    if test_type == "REFUSAL":
        # For refusal tests, PASS if model refused
        passed = answer_is_refusal
        failure_reason = "" if passed else "Expected refusal but got answer"
    else:
        # For factual tests, PASS if we have citations AND retrieval hit
        has_citations = len(citations) > 0
        if not has_citations:
            passed = False
            failure_reason = "No citations in answer"
        elif not hit:
            passed = False
            failure_reason = f"Expected source not found: {expected_sources}"
        elif answer_is_refusal:
            passed = False
            failure_reason = "Unexpected refusal"
        else:
            passed = True
            failure_reason = ""
    
    return {
        "test_id": test_id,
        "category": test["category"],
        "question": question,
        "expected_behavior": test["expected_behavior"],
        "expected_sources": ", ".join(expected_sources),
        "actual_answer": answer,  # Full answer, no truncation
        "answer_length": len(answer),
        "is_refusal": answer_is_refusal,
        "citation_count": len(citations),
        "cited_sources": ", ".join(cited_sources),
        "cited_pages": ", ".join(cited_pages),
        "retrieval_sources": ", ".join(retrieval_sources),
        "retrieval_hit": hit,
        "retrieval_rank": rank,
        "chunks_used": metrics["chunks_used"],
        "avg_retrieval_score": metrics.get("avg_retrieval_score"),
        "retrieval_latency_ms": round(metrics["retrieval_latency_ms"], 2),
        "generation_latency_ms": round(metrics["generation_latency_ms"], 2),
        "total_latency_ms": round(metrics["retrieval_latency_ms"] + metrics["generation_latency_ms"], 2),
        "model_name": metrics["model_name"],
        "request_id": data["request_id"],
        "pass_fail": "PASS" if passed else "FAIL",
        "failure_reason": failure_reason
    }


def run_evaluation():
    """Run all tests and generate the Excel report."""
    print("=" * 60)
    print("GOVERNED RAG - GOLDEN DATASET EVALUATION")
    print("=" * 60)
    
    # Check server health
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Server health: {health.json()}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Server not running. Start with:")
        print("  python -m uvicorn app.api.server:app --reload --port 8000")
        sys.exit(1)
    
    # Load dataset
    dataset = load_golden_dataset()
    tests = dataset["tests"]
    print(f"\nLoaded {len(tests)} test cases from golden dataset.\n")
    
    # Run all tests
    results = []
    start_time = time.time()
    
    # Delay between questions to avoid overwhelming LLM endpoint
    DELAY_BETWEEN_TESTS = 10  # seconds
    
    for i, test in enumerate(tests):
        result = evaluate_test(test)
        results.append(result)
        
        # Add delay between tests (skip after last test)
        if i < len(tests) - 1:
            time.sleep(DELAY_BETWEEN_TESTS)
    
    total_time = time.time() - start_time
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate summary stats
    total_tests = len(results)
    passed = sum(1 for r in results if r["pass_fail"] == "PASS")
    failed = total_tests - passed
    pass_rate = (passed / total_tests) * 100
    
    # Create summary DataFrame
    summary_data = {
        "Metric": [
            "Total Tests",
            "Passed",
            "Failed",
            "Pass Rate (%)",
            "Avg Retrieval Latency (ms)",
            "Avg Generation Latency (ms)",
            "Total Evaluation Time (s)",
            "Evaluation Timestamp"
        ],
        "Value": [
            total_tests,
            passed,
            failed,
            f"{pass_rate:.1f}%",
            f"{df['retrieval_latency_ms'].mean():.2f}",
            f"{df['generation_latency_ms'].mean():.2f}",
            f"{total_time:.2f}",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    
    # Write to Excel with multiple sheets
    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        # Summary sheet
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        
        # Detailed results sheet
        df.to_excel(writer, sheet_name="Results", index=False)
        
        # Failures only sheet
        failures_df = df[df["pass_fail"] == "FAIL"]
        if not failures_df.empty:
            failures_df.to_excel(writer, sheet_name="Failures", index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nResults: {passed}/{total_tests} PASSED ({pass_rate:.1f}%)")
    print(f"Total time: {total_time:.2f}s")
    print(f"\nReport saved to: {OUTPUT_PATH}")
    
    if failed > 0:
        print(f"\n⚠️ {failed} tests failed. See 'Failures' sheet for details.")
    
    return df


if __name__ == "__main__":
    run_evaluation()
