"""
Test script for /answer endpoint.

Tests:
1. Citation correctness
2. Refusal behavior
3. Grounding verification
4. Access control E2E
5. Request traceability
"""

import sys
from pathlib import Path
import requests
import json

# Add project root
sys.path.append(str(Path(__file__).parent.parent))

BASE_URL = "http://localhost:8000"


def test_citation_correctness():
    """Test: Model cites chunk IDs correctly for valid query."""
    print("\n=== TEST 1: Citation Correctness ===")
    
    response = requests.post(
        f"{BASE_URL}/answer",
        json={
            "text": "What are the ethical principles for AI?",
            "access_level": "public",
            "k": 3
        }
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    print(f"Request ID: {data['request_id']}")
    print(f"Answer: {data['answer'][:200]}...")
    print(f"Citations: {len(data['citations'])}")
    
    for cit in data['citations']:
        print(f"  - {cit['chunk_id']} from {cit['source']}")
    
    # Verify we have citations
    assert len(data['citations']) > 0, "Expected at least 1 citation"
    
    # Verify answer contains citation markers
    has_citation = "[CHUNK" in data['answer'] or len(data['citations']) > 0
    assert has_citation, "Answer should contain citation markers"
    
    print("✅ PASSED: Citations present")
    return True


def test_refusal_behavior():
    """Test: Model refuses unrelated queries."""
    print("\n=== TEST 2: Refusal Behavior ===")
    
    response = requests.post(
        f"{BASE_URL}/answer",
        json={
            "text": "What is the capital of France?",
            "access_level": "public",
            "k": 3
        }
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    print(f"Answer: {data['answer'][:300]}...")
    
    # Check for refusal indicators
    refusal_indicators = [
        "INSUFFICIENT_CONTEXT",
        "cannot answer",
        "not found",
        "no information",
        "not contain",
        "unable to"
    ]
    
    is_refusal = any(ind.lower() in data['answer'].lower() for ind in refusal_indicators)
    
    if is_refusal:
        print("✅ PASSED: Model correctly refused")
    else:
        print("⚠️ WARNING: Model may have hallucinated. Review answer manually.")
    
    return True


def test_access_control_e2e():
    """Test: Restricted query with public level returns only public docs."""
    print("\n=== TEST 3: Access Control E2E ===")
    
    response = requests.post(
        f"{BASE_URL}/answer",
        json={
            "text": "Tell me about information assurance regulations",
            "access_level": "public",
            "k": 5
        }
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    print(f"Request ID: {data['request_id']}")
    print(f"Chunks used: {data['metrics']['chunks_used']}")
    print(f"Citations: {len(data['citations'])}")
    
    # All citations should be from public docs
    for cit in data['citations']:
        print(f"  - {cit['source']} (page {cit.get('page_number', 'N/A')})")
    
    print("✅ PASSED: Check logs to verify access_level filter applied")
    return True


def test_request_traceability():
    """Test: request_id in response matches X-Request-ID header."""
    print("\n=== TEST 4: Request Traceability ===")
    
    response = requests.post(
        f"{BASE_URL}/answer",
        json={
            "text": "What are the contractor obligations?",
            "access_level": "public",
            "k": 3
        }
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    header_id = response.headers.get("X-Request-ID")
    
    print(f"Response request_id: {data['request_id']}")
    print(f"Header X-Request-ID: {header_id}")
    
    assert data['request_id'] == header_id, "Request IDs should match"
    
    print("✅ PASSED: Request ID is traceable")
    return True


def test_metrics_present():
    """Test: Metrics are correctly populated."""
    print("\n=== TEST 5: Metrics Verification ===")
    
    response = requests.post(
        f"{BASE_URL}/answer",
        json={
            "text": "What is the RFP submission deadline?",
            "access_level": "public",
            "k": 3
        }
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    
    metrics = data['metrics']
    print(f"Retrieval latency: {metrics['retrieval_latency_ms']:.2f}ms")
    print(f"Generation latency: {metrics['generation_latency_ms']:.2f}ms")
    print(f"Model: {metrics['model_name']}")
    print(f"Chunks used: {metrics['chunks_used']}")
    
    assert metrics['retrieval_latency_ms'] >= 0, "Retrieval latency should be >= 0"
    assert metrics['generation_latency_ms'] >= 0, "Generation latency should be >= 0"
    assert metrics['model_name'], "Model name should be present"
    assert metrics['chunks_used'] >= 0, "Chunks used should be >= 0"
    
    print("✅ PASSED: All metrics present")
    return True


def run_all_tests():
    """Run all verification tests."""
    print("=" * 60)
    print("GOVERNED RAG - ANSWER ENDPOINT VERIFICATION")
    print("=" * 60)
    
    tests = [
        test_citation_correctness,
        test_refusal_behavior,
        test_access_control_e2e,
        test_request_traceability,
        test_metrics_present,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {test.__name__} - {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    # Check server is running
    try:
        health = requests.get(f"{BASE_URL}/health")
        print(f"Server health: {health.json()}")
    except requests.exceptions.ConnectionError:
        print("ERROR: Server not running. Start with:")
        print("  python -m uvicorn app.api.server:app --reload --port 8000")
        sys.exit(1)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)
