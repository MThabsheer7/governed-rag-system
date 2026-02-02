import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from app.core.database import get_vector_store
from app.core.logger import get_logger

logger = get_logger(__name__)

def verify_retrieval(query: str):
    print(f"\n--- Verifying Retrieval for Query: '{query}' ---")
    
    vector_store = get_vector_store()
    
    # Retrieve top 3 results
    results = vector_store.similarity_search(query, k=3)
    
    if not results:
        print("No results found!")
        return

    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}]")
        print(f"Source: {doc.metadata.get('document_id', 'Unknown')}")
        print(f"Section: {doc.metadata.get('section_title', 'Unknown')}")
        print(f"Type: {doc.metadata.get('document_type', 'Unknown')}")
        print(f"Access: {doc.metadata.get('access_level', 'Unknown')}")
        print(f"Snippet: {doc.page_content[:150]}...")
        print("-" * 50)

if __name__ == "__main__":
    # Test queries relevant to the provided docs
    verify_retrieval("What are the principles of AI Ethics?")
    verify_retrieval("Contractor liability requirements")
