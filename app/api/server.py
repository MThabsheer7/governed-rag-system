import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.core.logger import get_logger
from app.retrieval.hybrid import HybridRetriever
from app.api.schemas import (
    QueryRequest, QueryResponse, SearchResult, QueryMetrics,
    AnswerRequest, AnswerResponse, Citation, AnswerMetrics
)
from app.core.synthesizer import synthesize_answer

logger = get_logger("api")

# Global Retriever Instance
retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handle startup and shutdown events.
    - Auto-ingests documents if vector store is empty (for fresh deployments)
    - Hydrates the BM25 index on startup
    """
    global retriever
    logger.info("Server Startup: Initializing...")
    start_time = time.time()
    
    try:
        # Check if we need to ingest (for fresh deployments like HF Spaces)
        from app.core.database import get_vector_store
        from pathlib import Path
        
        vector_store = get_vector_store()
        existing_docs = vector_store.get()
        
        if not existing_docs or not existing_docs.get('documents'):
            logger.info("Vector store is empty. Running auto-ingestion...")
            docs_path = Path(__file__).parent.parent.parent / "data" / "docs"
            
            if docs_path.exists() and any(docs_path.iterdir()):
                from app.ingestion.pipeline import ingest_all
                ingest_all()
                logger.info("Auto-ingestion complete.")
            else:
                logger.warning(f"No documents found in {docs_path}. Skipping ingestion.")
        else:
            logger.info(f"Vector store has {len(existing_docs['documents'])} documents. Skipping ingestion.")
        
        # Initialize retriever (will hydrate BM25 from vector store)
        retriever = HybridRetriever()
        duration = (time.time() - start_time) * 1000
        logger.info(f"Retriever initialized in {duration:.2f}ms")
        
    except Exception as e:
        logger.critical("Failed to initialize retriever", exc_info=True)
        # We might want to raise here to prevent startup if retrieval is broken
    
    yield
    
    logger.info("Server Shutdown")

app = FastAPI(title="Governed RAG API", version="1.0", lifespan=lifespan)

# CORS middleware for demo UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount demo UI static files
demo_path = Path(__file__).parent.parent.parent / "demo"
if demo_path.exists():
    app.mount("/demo", StaticFiles(directory=str(demo_path), html=True), name="demo")
    logger.info(f"Demo UI mounted at /demo from {demo_path}")


@app.middleware("http")
async def add_process_time_and_audit(request: Request, call_next):
    """
    Middleware for audit logging and latency tracking.
    Generates a unique request_id for every call.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Attach ID to request state for downstream use if needed
    request.state.request_id = request_id
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    
    # Attach ID to response headers for client tracing
    response.headers["X-Request-ID"] = request_id
    
    # Generic Access Log (Detailed query audit is inside the endpoint)
    # logger.info(f"API | method={request.method} | path={request.url.path} | status={response.status_code} | latency_ms={process_time:.2f} | request_id={request_id}")
    
    return response

@app.get("/health")
def health_check():
    status = "healthy" if retriever and retriever.bm25 else "degraded"
    return {"status": status, "retriever": "ready" if retriever else "not_ready"}

@app.post("/query", response_model=QueryResponse)
def search_documents(query_req: QueryRequest, request: Request):
    """
    Semantic Search Endpoint.
    Strictly performs retrieval only (No LLM generation).
    """
    global retriever
    if not retriever:
        return JSONResponse(status_code=503, content={"detail": "Retriever not initialized"})
    
    request_id = request.state.request_id
    start_time = time.time()
    
    try:
        # Governance Filter
        # Only allow documents that match the requested access_level (or are Public)
        # Simplified Policy: 
        #   - If user is RESTRICTED, they see (RESTRICTED + PUBLIC).
        #   - If user is PUBLIC, they strictly see PUBLIC.
        # Implementation in Chroma filter:
        #   Chroma filtering logic is strict AND. To do OR (Public OR Restricted), we need logic in the retriever.
        #   For v1 simplicity, we pass the strict filter. 
        #   Re-reading the requirement: "Access Level" usually implies clearance.
        #   If User=Restricted, they HAVE access to Restricted docs. (And implicitly Public).
        #   If User=Public, they ONLY have access to Public.
        
        # NOTE: Current implementation of hybrid.py takes a 'filter' dict which is passed to Chroma.
        # Chroma filter: {"access_level": "public"} matches ONLY public.
        # To match "Public OR Restricted", we'd need $or operator which might depend on Chroma version.
        # SAFE FALLBACK: If user request implies Higher Access, we might likely want to see everything?
        # Actually, let's stick to strict matching for the demo query param.
        # If user asks with access_level=RESTRICTED, we verify `access_level` field in docs matches strict requirements.
        # But wait, usually High Clearance users want to see Public docs too.
        # Solution: For this POC, let's make the filter explicitly match the requested level OR 'public' if possible.
        # But `filter` dict is usually exact match.
        # DECISION: To ensure determinism and clarity for the demo:
        # We will filter STRICTLY by the requested level. 
        # (e.g. "Show me Restricted docs about X"). 
        
        metadata_filter = {}
        if query_req.access_level == "public":
             metadata_filter = {"access_level": "public"}
        # If "restricted", we might technically allow "public" too, but for strict verification let's just search strict restricted implies checking restricted content.
        # Let's start with strict filtering based on input param to prove the point.
        else:
             metadata_filter = {"access_level": "restricted"}

        # Perform Hybrid Search
        results = retriever.hybrid_search(
            query=query_req.text,
            k=query_req.limit,
            filter=metadata_filter
        )
        
        latency = (time.time() - start_time) * 1000
        result_count = len(results)
        top_score = 0.0 # RRF doesn't return score easily in current method sig, might need update if critical.
        # For now, we log count.
        
        # Helper to safely parse int from metadata which might be string "None"
        def parse_int(val):
            if not val or str(val) == "None":
                return None
            try:
                return int(val)
            except ValueError:
                return None

        # Build Response
        response_model = QueryResponse(
            request_id=request_id,
            results=[
                SearchResult(
                    text=doc.page_content,
                    # Canonical Source is Document ID (Filename)
                    source=doc.metadata.get("document_id", "unknown"),
                    document_type=doc.metadata.get("document_type", "unknown"),
                    section_title=doc.metadata.get("section_title"),
                    page_number=parse_int(doc.metadata.get("page_number")) or -1, # Fallback to -1 if somehow None
                    score=doc.metadata.get("score"),
                    score_type="rrf",
                    # Clean Metadata: Remove leaked fields
                    metadata={
                        k: v for k, v in doc.metadata.items() 
                        if k not in ["score", "source", "page_content"]
                    }
                ) for doc in results
            ],
            metrics=QueryMetrics(latency_ms=latency, result_count=result_count)
        )
        
        # AUDIT LOG (Strict requirement)
        logger.info(
            f"AUDIT | request_id={request_id} | endpoint=/query | mode=hybrid "
            f"| latency_ms={latency:.2f} | user_level={query_req.access_level.value} "
            f"| results={result_count} | filter={metadata_filter}"
        )
        
        return response_model

    except Exception as e:
        logger.error(f"Query failed | request_id={request_id} | error={str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


# ============================================================
# Answer Synthesis Endpoint (Day 6)
# ============================================================

@app.post("/answer", response_model=AnswerResponse)
def answer_question(answer_req: AnswerRequest, request: Request):
    """
    Answer Synthesis Endpoint.
    
    Flow:
    1. Retrieve relevant chunks using existing hybrid search
    2. Pass frozen chunks to LLM for grounded synthesis
    3. Return answer with citations and audit trail
    
    Strict Governance:
    - Access level filtering is enforced
    - LLM uses ONLY retrieved context
    - All outputs are auditable via request_id
    """
    global retriever
    if not retriever:
        return JSONResponse(status_code=503, content={"detail": "Retriever not initialized"})
    
    request_id = request.state.request_id
    
    # ========== EVENT: Request Received ==========
    logger.info(
        f"[REQUEST] id={request_id} | endpoint=/answer | "
        f"query=\"{answer_req.text[:50]}...\" | access={answer_req.access_level.value} | k={answer_req.k}"
    )
    
    try:
        # 1. RETRIEVAL PHASE (reuse existing logic)
        retrieval_start = time.time()
        
        # Governance Filter (same as /query endpoint)
        metadata_filter = {}
        if answer_req.access_level == "public":
            metadata_filter = {"access_level": "public"}
        else:
            metadata_filter = {"access_level": "restricted"}
        
        # ========== EVENT: Retrieval Started ==========
        logger.info(f"[RETRIEVAL] id={request_id} | status=started | filter={metadata_filter}")
        
        # Retrieve chunks using hybrid search
        retrieved_chunks = retriever.hybrid_search(
            query=answer_req.text,
            k=answer_req.k,
            filter=metadata_filter
        )
        
        retrieval_latency = (time.time() - retrieval_start) * 1000
        
        # FREEZE chunks - no further modification allowed
        frozen_chunks = list(retrieved_chunks)
        
        # ========== EVENT: Retrieval Complete ==========
        chunk_sources = [c.metadata.get("document_id", "?") for c in frozen_chunks]
        logger.info(
            f"[RETRIEVAL] id={request_id} | status=complete | "
            f"chunks={len(frozen_chunks)} | latency_ms={retrieval_latency:.2f} | sources={chunk_sources}"
        )
        
        # ========== EVENT: Synthesis Started ==========
        logger.info(f"[SYNTHESIS] id={request_id} | status=started | chunks_in={len(frozen_chunks)}")
        
        # 2. SYNTHESIS PHASE
        answer, citations, generation_latency, model_name = synthesize_answer(
            question=answer_req.text,
            chunks=frozen_chunks
        )
        
        # ========== EVENT: Synthesis Complete ==========
        cited_ids = [c.chunk_id for c in citations]
        logger.info(
            f"[SYNTHESIS] id={request_id} | status=complete | "
            f"model={model_name} | latency_ms={generation_latency:.2f} | citations={len(citations)}"
        )
        
        # Compute average retrieval score from RRF
        scores = [c.metadata.get("score", 0) for c in frozen_chunks if c.metadata.get("score")]
        avg_score = sum(scores) / len(scores) if scores else None
        
        # 3. BUILD RESPONSE
        response = AnswerResponse(
            request_id=request_id,
            answer=answer,
            citations=citations,
            metrics=AnswerMetrics(
                retrieval_latency_ms=retrieval_latency,
                generation_latency_ms=generation_latency,
                model_name=model_name,
                chunks_used=len(frozen_chunks),
                avg_retrieval_score=avg_score
            )
        )
        
        # ========== EVENT: Audit Trail (Final) ==========
        logger.info(
            f"[AUDIT] id={request_id} | endpoint=/answer | "
            f"total_ms={retrieval_latency + generation_latency:.2f} | "
            f"model={model_name} | chunks={len(frozen_chunks)} | cited={cited_ids} | "
            f"access={answer_req.access_level.value}"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"[ERROR] id={request_id} | endpoint=/answer | error={str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
