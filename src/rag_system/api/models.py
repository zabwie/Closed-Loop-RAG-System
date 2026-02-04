"""API models for request/response schemas."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request model for RAG query endpoint."""

    query: str = Field(..., description="The user's query string")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")


class QueryResponse(BaseModel):
    """Response model for RAG query endpoint."""

    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="The generated answer")
    sources: list = Field(..., description="List of retrieved sources")
    retrieved_count: int = Field(..., description="Number of documents retrieved")
    evaluation: dict = Field(..., description="Evaluation metrics for the response")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""

    document_id: str = Field(..., description="Unique identifier for the ingested document")
    status: str = Field(..., description="Ingestion status (completed/failed)")
    chunk_count: int | None = Field(None, description="Number of chunks created")
    source: str | None = Field(None, description="Source filename")
    error: str | None = Field(None, description="Error message if ingestion failed")
