import os
import sys
from pathlib import Path
from pypdf import PdfReader
from app.ingestion.chunker import RuleBasedChunker
from app.core.schema import DocumentType, AccessLevel

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

def test_pdf_chunking():
    docs_dir = Path("data/docs")
    pdf_files = list(docs_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in data/docs")
        return

    # Pick the Ethics PDF as a good test case
    target_pdf = next((f for f in pdf_files if "Ethics" in f.name), pdf_files[0])
    print(f"Testing chunking on: {target_pdf.name}")

    reader = PdfReader(target_pdf)
    chunker = RuleBasedChunker()
    
    # Process first 3 pages
    print("\n--- Processing First 3 Pages ---")
    for i, page in enumerate(reader.pages[:3]):
        text = page.extract_text()
        page_num = i + 1
        
        chunks = chunker.split_text(
            text=text,
            document_id=target_pdf.name,
            document_type=DocumentType.POLICY, # Assuming policy for this test
            access_level=AccessLevel.PUBLIC,
            page_number=page_num
        )
        
        print(f"\nPage {page_num}: Extracted {len(chunks)} chunks")
        if chunks:
            print(f"Sample Chunk 1 Metadata: {chunks[0]['metadata']}")
            print(f"Sample Chunk 1 Text (truncated): {chunks[0]['text'][:100]}...")

if __name__ == "__main__":
    test_pdf_chunking()
