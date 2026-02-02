from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from app.core.logger import get_logger
from app.core.database import get_vector_store

logger = get_logger(__name__)

class HybridRetriever:
    def __init__(self):
        """
        Initializes the Hybrid Retriever.
        Hydrates BM25 index from local Vector Store (Memory-heavy for large datasets, fine for PoC).
        """
        self.vector_store = get_vector_store()
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Document] = []
        self._hydrate_bm25()

    def _hydrate_bm25(self):
        """
        Fetches all documents from ChromaDB to build the in-memory sparse index.
        """
        try:
            logger.info("Hydrating BM25 index from Vector Store...")
            # Fetch all documents
            # Note: In a real system, we'd paginate or use a generator. 
            # Chroma's get() returns dicts, need to reconstruct.
            data = self.vector_store.get()
            
            if not data or not data['documents']:
                logger.warning("Vector store is empty. BM25 index will be empty.")
                return

            self.documents = []
            corpus = []
            
            for i, text in enumerate(data['documents']):
                meta = data['metadatas'][i] if data['metadatas'] else {}
                doc_id = data['ids'][i]
                
                doc = Document(page_content=text, metadata=meta)
                # Attach the real ID for tracking
                doc.metadata['chunk_id'] = doc_id 
                
                self.documents.append(doc)
                # Simple tokenization for BM25 (split by space)
                # Production: Use a proper tokenizer (e.g. NLTK or the Embedding model's tokenizer)
                corpus.append(text.lower().split())

            self.bm25 = BM25Okapi(corpus)
            logger.info(f"BM25 index hydrated with {len(self.documents)} documents.")
            
        except Exception as e:
            logger.error("Failed to hydrate BM25", exc_info=True)

    def dense_search(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[Document]:
        """
        Run semantic vector search.
        """
        return self.vector_store.similarity_search(query, k=k, filter=filter)

    def sparse_search(self, query: str, k: int = 5, filter: Optional[Dict] = None) -> List[Document]:
        """
        Run keyword search using BM25.
        Simulate pre-filtering by retrieving more candidates and filtering in-memory.
        """
        if not self.bm25:
            logger.warning("BM25 not initialized.")
            return []

        tokenized_query = query.lower().split()
        
        # Retrieve 3x candidates to allow for filtering fallout
        # rank_bm25 returns the top-n documents directly
        top_n = self.bm25.get_top_n(tokenized_query, self.documents, n=k*3)
        
        results = []
        for doc in top_n:
            if len(results) >= k:
                break
                
            # Manual Filter Application
            if filter:
                match = True
                for key, val in filter.items():
                    # Simple check: metadata[key] == val
                    if doc.metadata.get(key) != str(val):
                        match = False
                        break
                if not match:
                    continue
            
            results.append(doc)
            
        return results

    def hybrid_search(self, query: str, k: int = 5, rrf_k: int = 60, filter: Optional[Dict] = None) -> List[Document]:
        """
        Combine Dense and Sparse results using Reciprocal Rank Fusion (RRF).
        RRF Score = 1 / (rank + k)
        """
        # 1. Get ranked lists
        dense_results = self.dense_search(query, k=k, filter=filter)
        sparse_results = self.sparse_search(query, k=k, filter=filter)
        
        # 2. Fuse scores
        doc_scores = {}
        
        # Helper to process a list
        def process_list(results: List[Document]):
            for rank, doc in enumerate(results):
                # unique ID for fusion logic
                # Ideally use chunk_id from metadata, but fallback to content hash if needed
                # We stored chunk_id in metadata during hydration
                doc_id = doc.metadata.get('chunk_id')
                if not doc_id:
                    continue
                    
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = {"doc": doc, "score": 0.0}
                
                doc_scores[doc_id]["score"] += 1.0 / (rrf_k + rank + 1)

        process_list(dense_results)
        process_list(sparse_results)
        
        # 3. Sort by aggregated score
        sorted_docs = sorted(doc_scores.values(), key=lambda x: x['score'], reverse=True)
        final_docs = []
        for item in sorted_docs[:k]:
            doc = item['doc']
            doc.metadata['score'] = item['score']
            final_docs.append(doc)
            
        return final_docs

