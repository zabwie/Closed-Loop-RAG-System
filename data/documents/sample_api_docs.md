# FastAPI RAG System API Documentation

## Overview

This document provides comprehensive documentation for the FastAPI-based RAG (Retrieval-Augmented Generation) system API. The API exposes endpoints for document ingestion, query processing, and system management. Built with FastAPI, it offers automatic OpenAPI documentation, type validation, and high performance through async operations.

## Base URL

All API endpoints are prefixed with `/api/v1`. The base URL for local development is `http://localhost:8000/api/v1`.

## Authentication

Currently, the API operates without authentication for development purposes. In production, API key authentication or OAuth2 should be implemented. Authentication headers should be included in the format: `Authorization: Bearer <your-api-key>`.

## Endpoints

### Document Management

#### POST /documents/upload

Uploads a new document to the RAG system for indexing and retrieval.

**Request Body:**
```json
{
  "file": "binary file data",
  "metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "category": "Technical",
    "tags": ["rag", "vector-database", "llm"]
  }
}
```

**Response (200 OK):**
```json
{
  "document_id": "doc_1234567890",
  "status": "indexed",
  "chunks_created": 15,
  "message": "Document successfully uploaded and indexed"
}
```

**Response (400 Bad Request):**
```json
{
  "error": "Invalid file format",
  "details": "Only PDF, TXT, and MD files are supported"
}
```

**Response (413 Payload Too Large):**
```json
{
  "error": "File too large",
  "details": "Maximum file size is 10MB"
}
```

#### GET /documents/{document_id}

Retrieves metadata and status information for a specific document.

**Path Parameters:**
- `document_id` (string, required): The unique identifier of the document

**Response (200 OK):**
```json
{
  "document_id": "doc_1234567890",
  "title": "Document Title",
  "author": "Author Name",
  "category": "Technical",
  "tags": ["rag", "vector-database", "llm"],
  "status": "indexed",
  "chunks_created": 15,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:00Z"
}
```

**Response (404 Not Found):**
```json
{
  "error": "Document not found",
  "document_id": "doc_1234567890"
}
```

#### GET /documents

Lists all documents in the system with optional filtering and pagination.

**Query Parameters:**
- `category` (string, optional): Filter by document category
- `status` (string, optional): Filter by document status (indexed, processing, failed)
- `limit` (integer, optional): Maximum number of results (default: 20, max: 100)
- `offset` (integer, optional): Number of results to skip (default: 0)

**Response (200 OK):**
```json
{
  "documents": [
    {
      "document_id": "doc_1234567890",
      "title": "Document Title",
      "category": "Technical",
      "status": "indexed",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "total": 45,
  "limit": 20,
  "offset": 0
}
```

#### DELETE /documents/{document_id}

Deletes a document and all associated chunks from the system.

**Path Parameters:**
- `document_id` (string, required): The unique identifier of the document

**Response (200 OK):**
```json
{
  "document_id": "doc_1234567890",
  "status": "deleted",
  "message": "Document successfully deleted"
}
```

**Response (404 Not Found):**
```json
{
  "error": "Document not found",
  "document_id": "doc_1234567890"
}
```

### Query Processing

#### POST /query

Processes a natural language query and returns relevant documents along with an AI-generated response.

**Request Body:**
```json
{
  "query": "How does RAG work with vector databases?",
  "top_k": 5,
  "temperature": 0.3,
  "include_sources": true,
  "filters": {
    "category": "Technical"
  }
}
```

**Request Parameters:**
- `query` (string, required): The natural language query to process
- `top_k` (integer, optional): Number of relevant documents to retrieve (default: 5, max: 20)
- `temperature` (float, optional): Controls response randomness (default: 0.3, range: 0.0-1.0)
- `include_sources` (boolean, optional): Whether to include source documents in response (default: true)
- `filters` (object, optional): Metadata filters for document retrieval

**Response (200 OK):**
```json
{
  "query": "How does RAG work with vector databases?",
  "answer": "RAG systems work with vector databases by converting documents into vector representations using embedding models...",
  "sources": [
    {
      "document_id": "doc_1234567890",
      "chunk_id": "chunk_001",
      "content": "Vector databases play a crucial role in modern RAG systems...",
      "similarity_score": 0.92,
      "metadata": {
        "title": "RAG System Architecture",
        "category": "Technical"
      }
    }
  ],
  "retrieval_time_ms": 45,
  "generation_time_ms": 234,
  "total_time_ms": 279
}
```

**Response (400 Bad Request):**
```json
{
  "error": "Invalid query",
  "details": "Query must be at least 3 characters long"
}
```

**Response (500 Internal Server Error):**
```json
{
  "error": "Query processing failed",
  "details": "Unable to connect to vector database"
}
```

#### POST /query/stream

Processes a query and streams the response in real-time, useful for long-form answers.

**Request Body:** Same as `/query` endpoint

**Response:** Server-Sent Events (SSE) stream with chunks of the response

```
data: {"type": "chunk", "content": "RAG systems work with vector databases"}
data: {"type": "chunk", "content": " by converting documents into vector representations"}
data: {"type": "done", "total_time_ms": 279}
```

### System Management

#### GET /health

Health check endpoint to verify system status.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "api": "healthy",
    "vector_database": "healthy",
    "llm_service": "healthy"
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### GET /stats

Returns system statistics and metrics.

**Response (200 OK):**
```json
{
  "documents": {
    "total": 45,
    "indexed": 42,
    "processing": 2,
    "failed": 1
  },
  "queries": {
    "total": 1234,
    "last_24h": 56,
    "avg_response_time_ms": 275
  },
  "vector_database": {
    "total_vectors": 678,
    "index_size_mb": 12.5
  }
}
```

## Error Handling

All endpoints follow consistent error response formats:

```json
{
  "error": "Error type",
  "details": "Detailed error message",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

Common HTTP status codes:
- 200: Success
- 400: Bad Request (invalid input)
- 404: Not Found
- 413: Payload Too Large
- 429: Too Many Requests (rate limiting)
- 500: Internal Server Error

## Rate Limiting

The API implements rate limiting to prevent abuse:
- 100 requests per minute per IP address
- 1000 requests per hour per IP address

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Best Practices

1. **Batch Document Uploads**: For large document collections, use batch uploads to reduce API overhead
2. **Optimize Query Parameters**: Adjust `top_k` and `temperature` based on your use case
3. **Use Streaming for Long Responses**: The `/query/stream` endpoint provides better UX for complex queries
4. **Implement Caching**: Cache frequent queries to reduce latency and API costs
5. **Monitor Response Times**: Track `total_time_ms` to identify performance bottlenecks
6. **Handle Errors Gracefully**: Implement proper error handling and retry logic with exponential backoff

## Interactive Documentation

FastAPI automatically generates interactive API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

These interfaces allow you to explore and test all API endpoints directly from your browser.
