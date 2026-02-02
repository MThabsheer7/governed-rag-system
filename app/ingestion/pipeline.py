import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document

# Fix path to ensure imports work
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from app.core.database import get_vector_store
from app.ingestion.chunker import RuleBasedChunker
from app.core.schema import DocumentType, AccessLevel
from app.core.logger import get_logger

logger = get_logger(__name__)

def ingest_documents(source_dir: str = "data/docs"):
    """
    Ingests all PDFs from source_dir into the local ChromaDB.
    """
    source_path = Path(source_dir)
    pdf_files = list(source_path.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {source_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDFs. Initializing Vector Store...")
    
    try:
        vector_store = get_vector_store()
        chunker = RuleBasedChunker()

        for file_path in pdf_files:
            try:
                logger.info(f"Processing document: {file_path.name}")
                
                # 1. Document Level Metadata
                doc_metadata = {
                    "document_id": file_path.name,
                    "source": str(file_path),
                    "ingested_at": datetime.utcnow().isoformat(),
                    "version": "1.0"
                }
                
                # 2. Determine Governance Type (Heuristic)
                filename_lower = file_path.name.lower()
                
                if "sop" in filename_lower:
                    doc_type = DocumentType.SOP
                elif "tender" in filename_lower or "rfp" in filename_lower:
                    doc_type = DocumentType.RFP
                else:
                    doc_type = DocumentType.POLICY
                
                access_level = AccessLevel.PUBLIC
                if "restricted" in filename_lower:
                    access_level = AccessLevel.RESTRICTED
                
                logger.info(f"Classified {file_path.name} as {doc_type.value} ({access_level.value})")
                
                # 3. Parse PDF & Chunk Page-by-Page
                reader = PdfReader(file_path)
                
                documents_to_add: List[Document] = []
                current_section_title = "General"  # Maintain context across pages
                
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if not text:
                        continue
                        
                    page_num = i + 1
                    
                    # Chunk this page
                    page_chunks = chunker.split_text(
                        text=text,
                        document_id=file_path.name,
                        document_type=doc_type,
                        access_level=access_level,
                        default_section_title=current_section_title,
                        page_number=page_num,
                        is_markdown=False
                    )
                    
                    if not page_chunks:
                        continue
                        
                    # Update context for next page (use the last section title found in this page)
                    # The last chunk of this page dictates the starting context of the next page
                    if page_chunks:
                        last_chunk = page_chunks[-1]
                        # Access nested metadata from Pydantic model dump
                        if "metadata" in last_chunk and "section_title" in last_chunk["metadata"]:
                            current_section_title = last_chunk["metadata"]["section_title"]

                    # Prepare Documents
                    for chunk in page_chunks:
                        # Merge document-level metadata
                        combined_metadata = {**doc_metadata, **chunk["metadata"]}
                        
                        # Fix Source to be just filename (Hygiene)
                        combined_metadata["source"] = file_path.name
                        
                        # Ensure Page Number is never None (Fallback -1)
                        if combined_metadata.get("page_number") is None:
                            combined_metadata["page_number"] = -1
                        
                        # Flatten for Chroma
                        cleaned_metadata = {}
                        for k, v in combined_metadata.items():
                            if isinstance(v, (str, int, float, bool)):
                                cleaned_metadata[k] = v
                            else:
                                cleaned_metadata[k] = str(v)
                        
                        doc = Document(
                            page_content=chunk["text"],
                            metadata=cleaned_metadata,
                            id=str(chunk["metadata"]["chunk_id"])
                        )
                        documents_to_add.append(doc)

                logger.info(f"Generated {len(documents_to_add)} chunks from {len(reader.pages)} pages.")

                # 3. Upsert
                if documents_to_add:
                    logger.info(f"Upserting chunks for {file_path.name}...")
                    vector_store.add_documents(documents_to_add)
                    logger.info("Upsert successful.")
                else:
                    logger.warning(f"No content extracted from {file_path.name}")
                    
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}", exc_info=True)

    except Exception as e:
        logger.critical("Critical failure in ingestion pipeline", exc_info=True)

if __name__ == "__main__":
    ingest_documents()
