"""FastAPI main application."""

import logging
import tempfile
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile
from src.rag_system.api.models import QueryRequest, QueryResponse, IngestResponse
from src.rag_system.evaluation.trulens_evaluator import SimulatedEvaluator

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="RAG System API",
    description="Closed-Loop RAG System with TruLens Evaluation",
    version="0.1.0",
)

# Initialize evaluator
evaluator = SimulatedEvaluator()


# Placeholder for RAG engine (will be initialized in Task 5)
# For now, we'll create a mock that will be replaced later
class MockRAGEngine:
    """Mock RAG engine for testing purposes."""

    async def query(self, query: str, top_k: int) -> dict:
        """Mock query method."""
        return {"answer": "Mock answer for testing", "sources": [], "retrieved_count": 0}


rag_engine = MockRAGEngine()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "RAG System API is running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query RAG system with evaluation.

    Args:
        request: QueryRequest containing query string and top_k parameter

    Returns:
        QueryResponse containing answer, sources, and evaluation metrics

    Raises:
        HTTPException: If query processing fails
    """
    try:
        # Query RAG engine
        result = await rag_engine.query(request.query, request.top_k)

        # Evaluate the response
        evaluation = await evaluator.evaluate_query(request.query, result)

        return {
            "query": request.query,
            "answer": result["answer"],
            "sources": result["sources"],
            "retrieved_count": result["retrieved_count"],
            "evaluation": evaluation,
        }
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile):
    """Upload and ingest a document.

    Args:
        file: Uploaded file (multipart/form-data).

    Returns:
        IngestResponse with document_id, status, chunk_count, source.

    Raises:
        HTTPException: If ingestion fails.
    """
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        # Generate unique document_id
        document_id = str(uuid.uuid4())

        # For now, return a mock response
        # In a real implementation, this would use DocumentIngester
        # to process the file through the ingestion pipeline
        result = {
            "document_id": document_id,
            "status": "completed",
            "chunk_count": 5,
            "source": file.filename,
        }
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if tmp_path.exists():
            tmp_path.unlink()
