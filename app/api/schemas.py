from typing import List, Dict, Any, Optional
from uuid import UUID
from pydantic import BaseModel, Field
from app.core.schema import AccessLevel

class QueryRequest(BaseModel):
    text: str = Field(..., min_length=3, description="The search query")
    access_level: AccessLevel = Field(default=AccessLevel.PUBLIC, description="User's security clearance")
    limit: int = Field(default=5, ge=1, le=20, description="Number of results to return")

class SearchResult(BaseModel):
    text: str
    source: str
    document_type: str = "unknown"
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    score: Optional[float] = None
    score_type: str = "rrf"
    metadata: Dict[str, Any]

class QueryMetrics(BaseModel):
    latency_ms: float
    result_count: int

class QueryResponse(BaseModel):
    request_id: str  # Kept as string for serialization ease
    results: List[SearchResult]
    metrics: QueryMetrics


# ============================================================
# Answer Synthesis Schemas (Day 6)
# ============================================================

class AnswerRequest(BaseModel):
    """Request for answer synthesis."""
    text: str = Field(..., min_length=3, description="The user's question")
    access_level: AccessLevel = Field(default=AccessLevel.PUBLIC, description="User's security clearance")
    k: int = Field(default=5, ge=1, le=10, description="Number of chunks to retrieve")


class Citation(BaseModel):
    """A citation to a specific chunk used in the answer."""
    chunk_id: str
    source: str  # Document filename
    section_title: Optional[str] = None
    page_number: Optional[int] = None


class AnswerMetrics(BaseModel):
    """Observability metrics for answer generation."""
    retrieval_latency_ms: float
    generation_latency_ms: float
    model_name: str
    chunks_used: int
    avg_retrieval_score: Optional[float] = None  # Average RRF score of retrieved chunks


class AnswerResponse(BaseModel):
    """Response from the /answer endpoint."""
    request_id: str
    answer: str
    citations: List[Citation]
    metrics: AnswerMetrics
