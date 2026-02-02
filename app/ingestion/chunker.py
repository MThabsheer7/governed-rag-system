from typing import List, Optional, Tuple
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from app.core.schema import ChunkMetadata, AccessLevel, DocumentType

class RuleBasedChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 75):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Base splitter for inner content
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 4,
            chunk_overlap=chunk_overlap * 4,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Headers to track for Markdown
        self.md_headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

    def split_text(
        self, 
        text: str, 
        document_id: str, 
        document_type: DocumentType, 
        access_level: AccessLevel,
        default_section_title: str = "General",
        page_number: Optional[int] = None,
        is_markdown: bool = False
    ) -> List[dict]:
        """
        Splits text and extracts headers if possible.
        """
        if is_markdown:
            return self._chunk_markdown(text, document_id, document_type, access_level)
        else:
            return self._chunk_text(text, document_id, document_type, access_level, default_section_title, page_number)

    def _chunk_markdown(
        self, text: str, document_id: str, document_type: DocumentType, access_level: AccessLevel
    ) -> List[dict]:
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.md_headers_to_split_on)
        md_docs = md_splitter.split_text(text)
        
        result_chunks = []
        for doc in md_docs:
            # Further split if a section is too long
            splits = self.recursive_splitter.split_documents([doc])
            
            for split in splits:
                # Construct composite section title from metadata
                headers = [v for k, v in split.metadata.items() if k in ["Header 1", "Header 2", "Header 3"]]
                section_title = " > ".join(headers) if headers else "General"

                result_chunks.append({
                    "text": split.page_content,
                    "metadata": ChunkMetadata(
                        document_id=document_id,
                        document_type=document_type,
                        access_level=access_level,
                        section_title=section_title,
                        page_number=None
                    ).model_dump()
                })
        return result_chunks

    def _chunk_text(
        self, 
        text: str, 
        document_id: str, 
        document_type: DocumentType, 
        access_level: AccessLevel, 
        default_section_title: str,
        page_number: Optional[int]
    ) -> List[dict]:
        """
        Splits text based on common document structure (headers) first,
        then chunks recursively within those sections.
        """
        # Regex for potential headers
        # These patterns cover common government/compliance document formats
        header_patterns = [
            # ARTICLE I, SECTION 2, PART III, CHAPTER 1
            r"^(?:ARTICLE|SECTION|PART|CHAPTER)\s+[IVX0-9]+.*$",  
            # 5.3 Management Controls, 14.1.5 Title
            r"^\d+\.\d*(?:\.\d+)*\s+[A-Z].*$",
            # M1 Strategy and Planning, M1.1 Entity Context, T5.2.2 Access Control
            # Pattern: Letter + Number(s) + optional decimals + Title starting with capital
            r"^[A-Z]\d+(?:\.\d+)*\s+[A-Z].*$",
        ]
        combined_pattern = "|".join(header_patterns)
        
        matches = list(re.finditer(combined_pattern, text, re.MULTILINE))
        
        sections: List[Tuple[str, str]] = []
        
        # If no matches, return whole text
        if not matches:
            sections.append((default_section_title, text))
            return self._finalize_chunks(sections, document_id, document_type, access_level, page_number)

        # Process matches
        current_header = default_section_title
        current_content_start = 0
        
        for i, match in enumerate(matches):
            raw_header = match.group().strip()
            cleaned_header = self._clean_header(raw_header)
            
            # 1. Validate the candidate header
            if not self._is_valid_header(cleaned_header):
                # If invalid, it's just content. Ignore this match boundaries as a split point?
                # Actually, if we ignore it, we just continue accumulating content until the next *valid* header.
                continue
            
            # It is a valid header. 
            # Capture content from previous start up to this header's start
            prev_content = text[current_content_start:match.start()]
            
            # Append previous section
            if prev_content.strip() or current_header != default_section_title:
                 sections.append((current_header, prev_content))

            # Update state
            current_header = cleaned_header
            current_content_start = match.end() # Start content after this header
            
            # (Optionally include the header text itself in the content? 
            # Usually redundant if it's in metadata, but safe to keep context)
            # For now, we skip adding it to content to avoid duplication, strictly using it as metadata.

        # Append remaining content after last valid header
        final_content = text[current_content_start:]
        sections.append((current_header, final_content))

        return self._finalize_chunks(sections, document_id, document_type, access_level, page_number)

    def _finalize_chunks(self, sections, document_id, document_type, access_level, page_number):
        result_chunks = []
        for section_title, section_content in sections:
            if not section_content.strip():
                continue
                
            chunks = self.recursive_splitter.create_documents([section_content])
            for chunk in chunks:
                result_chunks.append({
                    "text": chunk.page_content,
                    "metadata": ChunkMetadata(
                        document_id=document_id,
                        document_type=document_type,
                        access_level=access_level,
                        section_title=section_title,
                        page_number=page_number
                    ).model_dump()
                })
        return result_chunks

    def _clean_header(self, header: str) -> str:
        header = header.replace("\n", " ").strip()
        header = re.sub(r'\s+\d+$', '', header) # Remove trailing page numbers
        return header

    def _is_valid_header(self, header: str) -> bool:
        """
        Heuristics to filter out false positives (sentences starting with numbers).
        """
        # 1. Length check (Headers are usually short)
        if len(header) > 80:
            return False
            
        # 2. Stopword check (Headers don't start with these)
        # Check after the number, e.g. "1. The..." -> Check "The"
        # Remove leading numbers/dots
        title_text = re.sub(r'^[\d\.\s]+', '', header).strip()
        if not title_text:
            return True # Just numbers like "1.2.3" -> Valid
            
        first_word = title_text.split()[0].lower()
        stopwords = {"the", "a", "an", "this", "that", "these", "those", "if", "when", "we", "it", "there"}
        if first_word in stopwords:
            return False
            
        # 3. Punctuation check (Headers usually don't end in periods, unless it's just the number)
        if header.endswith(".") and len(title_text) > 0:
             return False
             
        # 4. Verb check (naive): headers rarely contain " is ", " are "
        if " is " in f" {title_text} " or " are " in f" {title_text} ":
            return False
            
        return True
