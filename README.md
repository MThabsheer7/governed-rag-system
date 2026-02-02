# Governed, Auditable RAG System for Enterprise & Government

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-development-orange)

## 🎯 Project Overview

This project aims to build a **Governed, Auditable Retrieval-Augmented Generation (RAG)** system tailored for **Government agencies and regulated enterprises**. 

Unlike traditional LLM chat applications, this system prioritizes **data sovereignty**, **auditability**, and **determinism** to handle sensitive policy and contract documents without hallucination or leakage.

### The Problem
Regulated environments face unique challenges that generic AI tools cannot solve:
- **Hallucinations**: Fabrication of information is unacceptable in policy.
- **Lack of Citations**: Answers must be traceable to specific document clauses.
- **Data Privacy**: Sending sensitive docs to external APIs (OpenAI, Anthropic) is often prohibited.
- **Governance**: Access controls must be respected at the retrieval level.

---

## 🛡️ Core Constraints & Design Principles

This system is engineered around **6 Critical GovTech Constraints**:

### 1. Data Sovereignty
- **No External APIs**: The entire pipeline (Ingestion, Embedding, Retrieval, Generation) runs **locally**.
- **Offline Capable**: No document content leaves the secure environment.

### 2. Explainability & Auditability
- **Traceable Sources**: Every answer cites specific chunk IDs.
- **Human-Readable Logs**: Full visibility into the decision path.
- **Reproducibility**: System ensures consistent answers for identical queries.

### 3. Access Control Simulation
- **Metadata Filtering**: Retrieval strictly respects document-level permissions (e.g., specific departments or clearance levels).
- **Security First**: Restricted documents are never retrieved for unauthorized queries.

### 4. Determinism
- **Zero Temperature**: `temperature=0` is enforced to prevent creative drift.
- **No Random Sampling**: Outputs are consistent and factual.

### 5. Model Transparency
- **Open Source Foundation**: Powered by **Qwen 2.5**.
- **Resource Optimized**: 
    - **Development**: Qwen 2.5 (0.5B / 1.5B / 3B) 
    - **Production**: Qwen 2.5 (7B / 14B)
- **Deployment**: Optimized for CPU/Low-VRAM environments (GGUF Quantization).

### 6. Deployment Flexibility
- **Lightweight**: Capable of running on commodity hardware or cheap VMs (Render, Fly.io).
- **GPU Optional**: Fully functional on CPU-only infrastructure.

---

## 🏗️ Architecture

*(Architecture diagram placeholder)*

The system strictly adheres to a retrieval-first approach:
1.  **Ingest**: Documents are parsed, chunked, and embedded locally.
2.  **Retrieve**: Semantic & Keyword search finds relevant chunks based on query + governance filters.
    **Important Scale Note**:
    For the current dataset size, the BM25 index is hydrated in memory from the persisted vector store at system startup. In production-scale deployments, sparse and dense indices would be independently persisted and refreshed asynchronously to avoid heavy startup costs.
3.  **Generate**: Local LLM synthesizes an answer using *only* the retrieved context.

---

## Document Corpus
We selected a small but representative set of government-style documents to stress-test retrieval, governance, and citation accuracy.

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)
- LLM Backend (local llama.cpp server or Google Colab with ngrok)

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/thabsheer/governed-rag-system.git
cd governed-rag-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure LLM endpoint
cp .env.example .env
# Edit .env with your LLM_ENDPOINT (e.g., ngrok URL from Colab)

# Start the server (auto-ingests documents on first run)
uvicorn app.api.server:app --reload --port 8000
```

### Option 2: Docker

```bash
# Clone and configure
git clone https://github.com/thabsheer/governed-rag-system.git
cd governed-rag-system
cp .env.example .env

# Build and run
docker-compose up --build

# API available at http://localhost:8000
```

### Auto-Ingestion

The server automatically ingests documents on first startup if the vector store is empty. This enables:
- **Fresh deployments** (like HF Spaces) to work without pre-ingested data
- **Zero manual setup** - just add documents to `data/docs/` and restart

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/query` | POST | Retrieval only (no LLM) |
| `/answer` | POST | Full RAG: Retrieval + LLM synthesis |

### Example Request

```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"text": "What are the AI ethics principles?", "k": 5}'
```

---

## 📂 Project Structure

```
governed-rag-system/
├── app/
│   ├── api/          # FastAPI server & schemas
│   ├── core/         # Synthesizer, logger, database
│   ├── ingestion/    # Document loaders & chunkers
│   ├── models/       # LLM loader & backends
│   └── retrieval/    # Hybrid retriever (Dense + BM25)
├── data/
│   ├── docs/         # Source documents (PDF, MD)
│   └── chroma_db/    # Vector store (gitignored)
├── tests/            # Test suite & evaluation scripts
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run golden dataset evaluation (requires LLM)
python tests/evaluate_golden.py
```

---

## 📜 License
MIT

