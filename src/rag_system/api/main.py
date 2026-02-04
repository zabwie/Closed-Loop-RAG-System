"""FastAPI main application."""

import logging
from fastapi import FastAPI, HTTPException
from src.rag_system.api.models import QueryRequest, QueryResponse
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
