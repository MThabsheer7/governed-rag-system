import sys
import os
from pathlib import Path
from pypdf import PdfReader

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.ingestion.chunker import RuleBasedChunker
from app.core.schema import DocumentType, AccessLevel

def test_real_pdf_extraction():
    # Path to a real PDF
    docs_dir = project_root / "data" / "docs"
    
    # Try to find a PDF
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDFs found in data/docs!")
        return
    
    # Prefer the Ethics or Compliance guide if available as they likely have structure
    target_pdf = next((f for f in pdf_files if "Ethics" in f.name or "Compliance" in f.name), pdf_files[0])
    
    print(f"Testing on file: {target_pdf.name}")
    
    try:
        reader = PdfReader(target_pdf)
        
        # Extract text from the first 20 pages (enough to catch Table of Contents + some content)
        full_text = ""
        page_limit = min(len(reader.pages), 20)
        
        print(f"Extracting text from first {page_limit} pages...")
        for i in range(page_limit):
            page_text = reader.pages[i].extract_text()
            if page_text:
                full_text += page_text + "\n"
        
        print(f"Extracted {len(full_text)} characters.")
        
        chunker = RuleBasedChunker()
        
        print("\n--- Running Chunker with Regex Header Extraction ---")
        chunks = chunker.split_text(
            text=full_text,
            document_id=target_pdf.name,
            document_type=DocumentType.POLICY,
            access_level=AccessLevel.PUBLIC,
            is_markdown=False
        )
        
        print(f"\nGenerated {len(chunks)} chunks.")
        
        unique_headers = set()
        print("\n--- Chunk Sample (Structure Verification) ---")
        # Print a sample of chunks from different parts
        for i, chunk in enumerate(chunks):
            section = chunk['metadata']['section_title']
            unique_headers.add(section)
            # Only print first few, then some from middle if many
            if i < 5 or (i > 10 and i < 15):
                print(f"Chunk {i} | Header: '{section}' | Text: {chunk['text'][:60].replace(chr(10), ' ')}...")
            
        print("\nUnique Headers Extracted:", list(unique_headers)[:10], "..." if len(unique_headers) > 10 else "")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")

if __name__ == "__main__":
    test_real_pdf_extraction()
