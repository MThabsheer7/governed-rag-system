import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Core configurations
PERSIST_DIRECTORY = "./data/chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def get_vector_store() -> Chroma:
    """
    Initialize and return the ChromaDB vector store.
    
    Governance Constraints:
    - Runs locally (no external API).
    - Persists to local filesystem.
    - Uses CPU-friendly embedding model (MiniLM).
    """
    embedding_function = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )
    
    vector_store = Chroma(
        collection_name="governed_docs",
        embedding_function=embedding_function,
        persist_directory=PERSIST_DIRECTORY
    )
    
    return vector_store
