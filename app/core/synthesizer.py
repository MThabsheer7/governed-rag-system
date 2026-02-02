"""
Answer Synthesizer Module
-------------------------
Orchestrates grounded answer generation from retrieved chunks.

Core Responsibilities:
1. Build strict grounding prompts with 4 hard constraints
2. Call LLM with deterministic settings
3. Extract citations from LLM output
4. Return structured, auditable responses

CRITICAL CONSTRAINTS (Non-Negotiable):
- Use ONLY provided context
- Do NOT use outside knowledge
- CITE chunk IDs explicitly
- REFUSE if context insufficient
"""

import re
import time
from typing import List, Tuple, Optional

from langchain_core.documents import Document

from app.core.logger import get_logger
from app.models.llm_loader import get_llm_backend
from app.api.schemas import Citation

logger = get_logger(__name__)


# ============================================================
# Strict Grounding Prompt Template
# ============================================================

SYSTEM_PROMPT = """You are a Governed RAG Assistant for government and enterprise documents.

STRICT RULES:
1. USE ONLY THE PROVIDED CONTEXT - Never use knowledge from your training data.
2. NO HALLUCINATION - If context lacks the information, say so clearly.
3. ALWAYS CITE - After each fact, add a citation like [C1], [C2], etc.
4. REFUSE IF INSUFFICIENT - If you cannot fully answer, start with "INSUFFICIENT_CONTEXT:" then explain and cite what you reviewed.

FORMAT RULES:
- Write a natural language answer with inline citations.
- Use [C1], [C2], [C3] etc. to cite sources (matching the context labels).
- ALWAYS include citations, even when refusing - cite the chunks you reviewed.
- Do NOT copy chunk headers into your answer.
- Keep answers concise and factual.

EXAMPLES:

Answering:
"The policy requires background checks for all employees [C1]. The checks must include identity verification [C1] and criminal record review [C2]."

Refusing (with citations):
"INSUFFICIENT_CONTEXT: The provided documents discuss AI ethics [C1] and information security [C2] but do not contain information about Python installation."

You are being audited. Follow these rules exactly."""

USER_PROMPT_TEMPLATE = """### CONTEXT DOCUMENTS
Each context chunk is labeled [C1], [C2], etc. Cite using these labels.

{context_block}

### QUESTION
{question}

### YOUR TASK
1. Read the context carefully.
2. Answer the question using ONLY information from the context.
3. After each fact, cite the source with [C1], [C2], etc.
4. If the context does not contain the answer:
   - Start with "INSUFFICIENT_CONTEXT:"
   - Explain what topics ARE covered in the context
   - Cite the chunks you reviewed (e.g., [C1], [C2])
5. Write in clear, natural language.

### ANSWER"""


def build_context_block(chunks: List[Document]) -> Tuple[str, dict]:
    """
    Build a formatted context block from retrieved chunks.
    Uses indexed labels [C1], [C2] instead of long UUIDs.
    
    Args:
        chunks: List of retrieved Document objects
        
    Returns:
        Tuple of (formatted context string, index_map for UUID lookup)
    """
    context_parts = []
    index_map = {}  # Maps index -> chunk metadata for remapping
    
    for i, doc in enumerate(chunks):
        idx = i + 1  # 1-indexed: C1, C2, C3...
        chunk_id = doc.metadata.get("chunk_id", f"unknown_{i}")
        source = doc.metadata.get("document_id", "unknown")
        section = doc.metadata.get("section_title", "")
        page = doc.metadata.get("page_number", "")
        
        # Store mapping for later remapping
        index_map[idx] = {
            "chunk_id": chunk_id,
            "source": source,
            "section_title": section,
            "page_number": page
        }
        
        # Header with indexed label (short and reliable)
        header = f"[C{idx}] Source: {source}"
        if section:
            header += f" | Section: {section}"
        if page:
            header += f" | Page: {page}"
        
        context_parts.append(f"{header}\n{doc.page_content}\n")
    
    return "\n---\n".join(context_parts), index_map


def build_prompt(question: str, chunks: List[Document]) -> Tuple[str, dict]:
    """
    Construct the full prompt with system instructions and context.
    
    Args:
        question: User's question
        chunks: Retrieved chunks (frozen/immutable)
        
    Returns:
        Tuple of (complete prompt string, index_map for remapping)
    """
    context_block, index_map = build_context_block(chunks)
    
    user_prompt = USER_PROMPT_TEMPLATE.format(
        context_block=context_block,
        question=question
    )
    
    # For Qwen chat format, we structure as system + user
    full_prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""
    return full_prompt, index_map


def extract_citations(answer: str, index_map: dict) -> List[Citation]:
    """
    Extract indexed citations from the LLM's answer and remap to UUIDs.
    
    Looks for patterns like [C1], [C2], [C3]
    
    Args:
        answer: Raw LLM output
        index_map: Mapping from index -> chunk metadata (from build_context_block)
        
    Returns:
        List of Citation objects with real UUIDs
    """
    # Find all indexed citations: [C1], [C2], [C3], etc.
    pattern = r'\[C(\d+)\]'
    matches = re.findall(pattern, answer, re.IGNORECASE)
    
    # Deduplicate while preserving order
    seen = set()
    citations = []
    
    for idx_str in matches:
        try:
            idx = int(idx_str)
        except ValueError:
            continue
            
        if idx in seen:
            continue
        seen.add(idx)
        
        # Look up metadata from index_map
        meta = index_map.get(idx, {})
        if not meta:
            logger.warning(f"Citation [C{idx}] not found in index_map")
            continue
        
        # Parse page number safely
        page_num = meta.get("page_number")
        if page_num and str(page_num) != "None" and str(page_num) != "":
            try:
                page_num = int(page_num)
            except (ValueError, TypeError):
                page_num = None
        else:
            page_num = None
        
        citations.append(Citation(
            chunk_id=meta.get("chunk_id", f"unknown_{idx}"),
            source=meta.get("source", "unknown"),
            section_title=meta.get("section_title"),
            page_number=page_num
        ))
    
    return citations


def synthesize_answer(
    question: str,
    chunks: List[Document]
) -> Tuple[str, List[Citation], float, str]:
    """
    Generate a grounded answer from retrieved chunks.
    
    Args:
        question: User's question
        chunks: Retrieved chunks (FROZEN - do not modify)
        
    Returns:
        Tuple of (answer, citations, generation_latency_ms, model_name)
    """
    if not chunks:
        return (
            "INSUFFICIENT_CONTEXT: No relevant documents were retrieved.",
            [],
            0.0,
            "none"
        )
    
    # Get LLM backend
    llm = get_llm_backend()
    model_name = llm.model_name
    
    # Build prompt with frozen chunks (returns index_map for remapping)
    prompt, index_map = build_prompt(question, chunks)
    
    logger.info(f"[LLM] status=calling | model={model_name} | prompt_chars={len(prompt)} | chunks={len(chunks)}")
    
    # Generate with timing
    start_time = time.time()
    try:
        raw_answer = llm.generate(prompt, max_tokens=512)
    except Exception as e:
        logger.error(f"[LLM] status=failed | model={model_name} | error={str(e)}")
        return (
            f"ERROR: Generation failed - {str(e)}",
            [],
            (time.time() - start_time) * 1000,
            model_name
        )
    generation_latency = (time.time() - start_time) * 1000
    
    logger.info(f"[LLM] status=complete | model={model_name} | latency_ms={generation_latency:.2f} | output_chars={len(raw_answer)}")
    
    # Clean up answer (remove trailing special tokens)
    answer = raw_answer.strip()
    if "<|im_end|>" in answer:
        answer = answer.split("<|im_end|>")[0].strip()
    
    # Extract citations using index_map (remaps [C1] -> UUID)
    citations = extract_citations(answer, index_map)
    
    logger.info(f"Generated answer with {len(citations)} citations in {generation_latency:.2f}ms")
    
    return answer, citations, generation_latency, model_name
