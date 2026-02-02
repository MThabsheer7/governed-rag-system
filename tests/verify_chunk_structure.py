import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import get_vector_store
from app.core.logger import get_logger

logger = get_logger(__name__)

def verify_structure():
    print("Initializing Vector Store Connection...")
    vector_store = get_vector_store()
    
    print("Fetching all stored documents to analyze structure...")
    # Chroma specific: .get() retrieves all documents including metadata
    # This might be heavy for production DBs but fine for this POC verification.
    try:
        data = vector_store.get()
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    if not data or not data['ids']:
        print("Vector store is empty! Did you run the ingestion pipeline?")
        return

    print(f"Total Chunks in DB: {len(data['ids'])}")

    # Group chunks by Document ID
    docs_map = {}
    for i, meta in enumerate(data['metadatas']):
        doc_id = meta.get('document_id', 'UNKNOWN')
        if doc_id not in docs_map:
            docs_map[doc_id] = []
        
        docs_map[doc_id].append({
            'text': data['documents'][i],
            'meta': meta,
            'id': data['ids'][i]
        })
    
    # Pick top 2 documents
    unique_docs = list(docs_map.keys())
    target_docs = unique_docs[:2]
    
    print(f"\nFound {len(unique_docs)} unique documents: {unique_docs}")
    print(f"Inspecting structure for: {target_docs}")

    for doc_id in target_docs:
        print(f"\n{'='*80}")
        print(f"DOCUMENT: {doc_id}")
        print(f"{'='*80}")
        
        chunks = docs_map[doc_id]
        
        # Group by Section Title
        sections = {}
        for chunk in chunks:
            sec = chunk['meta'].get('section_title', 'General')
            if sec not in sections:
                sections[sec] = []
            sections[sec].append(chunk)
            
        # Print Sections
        # Sort sections by name just for consistent display (though heuristics might result in weird sort)
        sorted_sections = sorted(sections.keys())
        
        for sec_title in sorted_sections:
            sec_chunks = sections[sec_title]
            print(f"\n  SECTION HEADER: '{sec_title}'")
            print(f"  {'-'*40}")
            
            for i, chunk in enumerate(sec_chunks):
                # formatting snippet
                snippet = chunk['text'][:80].replace('\n', ' ')
                access = chunk['meta'].get('access_level', 'N/A')
                print(f"    [Chunk {i+1}] [{access}] {snippet}...")

if __name__ == "__main__":
    verify_structure()
