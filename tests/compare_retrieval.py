import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.retrieval.hybrid import HybridRetriever
from app.core.logger import get_logger

logger = get_logger(__name__)

def compare_retrieval():
    print("Initializing Hybrid Retriever...")
    retriever = HybridRetriever()
    
    queries = [
        "What are the ethical principles for AI?",
        "Contractor liability clause 3.2", 
        "RFP submission deadline",
        "Data encryption standards AES-256", 
        "MOCAI guidelines",
    ]
    
    output_file = Path("tests/retrieval_comparison.md")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Qualitative Retrieval Validation (Top 3)\n")
        f.write(f"**Goal**: Validate that Hybrid retrieval captures both semantic meaning and specific keywords.\n\n")

        print(f"Running validation on {len(queries)} queries. Output: {output_file}")

        for q in queries:
            f.write(f"## Query: '{q}'\n")
            
            # 1. Dense
            dense_docs = retriever.dense_search(q, k=3)
            f.write(f"### DENSE (Semantic)\n")
            for i, doc in enumerate(dense_docs):
                snippet = doc.page_content[:150].replace('\n', ' ')
                source = doc.metadata.get('document_id')
                f.write(f"**{i+1}.** {snippet}... *({source})*\n\n")
            
            # 2. Sparse (BM25)
            sparse_docs = retriever.sparse_search(q, k=3)
            f.write(f"### SPARSE (Keyword)\n")
            for i, doc in enumerate(sparse_docs):
                snippet = doc.page_content[:150].replace('\n', ' ')
                source = doc.metadata.get('document_id')
                f.write(f"**{i+1}.** {snippet}... *({source})*\n\n")
                
            # 3. Hybrid (RRF)
            hybrid_docs = retriever.hybrid_search(q, k=3)
            f.write(f"### HYBRID (RRF)\n")
            for i, doc in enumerate(hybrid_docs):
                # Check overlap
                is_dense = any(d.page_content == doc.page_content for d in dense_docs)
                is_sparse = any(d.page_content == doc.page_content for d in sparse_docs)
                
                origin = []
                if is_dense: origin.append("Dense")
                if is_sparse: origin.append("Sparse")
                origin_str = " + ".join(origin) if origin else "Lower Ranked"
                
                snippet = doc.page_content[:150].replace('\n', ' ')
                source = doc.metadata.get('document_id')
                f.write(f"**{i+1}.** [{origin_str}] {snippet}... *({source})*\n\n")
            
            f.write("---\n\n")
            
    print(f"Done. Check {output_file} for results.")

if __name__ == "__main__":
    compare_retrieval()
