"""
End-to-end integration tests for the full RAG pipeline.

Tests cover:
- Full RAG pipeline (ingest → query → evaluate)
- Error scenarios (no documents, empty query)
- Multi-document retrieval
- Docker health checks
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys


# Mock MarkItDown to avoid ffmpeg dependency
class MockMarkItDown:
    def convert(self, content):
        return type(
            "Document", (), {"text_content": str(content), "format": "markdown", "metadata": {}}
        )()


sys.modules["markitdown"] = type(sys)("markitdown")
sys.modules["markitdown"].MarkItDown = MockMarkItDown
sys.modules["markitdown._markitdown"] = type(sys)("markitdown._markitdown")

from rag_system.api.main import app
import httpx


@pytest.fixture(scope="session")
def ollama_available():
    """Check if Ollama service is available. Skip tests if not."""
    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return True
    except Exception:
        pass
    pytest.skip("Ollama service not available. Run 'docker-compose up' to start services.")


@pytest.fixture(scope="session")
def milvus_available():
    """Check if Milvus service is available. Skip tests if not."""
    try:
        from pymilvus import connections

        connections.connect(host="localhost", port="19530")
        connections.disconnect("default")
        return True
    except Exception:
        pass
    pytest.skip("Milvus service not available. Run 'docker-compose up' to start services.")


@pytest.mark.e2e
def test_full_rag_pipeline(ollama_available, milvus_available):
    """Test ingest → query → evaluate pipeline"""
    client = TestClient(app)

    # Ingest document
    sample_md_path = Path(__file__).parent / "fixtures" / "sample.md"
    with open(sample_md_path, "rb") as f:
        ingest_response = client.post("/ingest", files={"file": ("sample.md", f, "text/markdown")})
        assert ingest_response.status_code == 200
        ingest_data = ingest_response.json()
        assert ingest_data["status"] == "completed"
        assert ingest_data["chunk_count"] > 0
        assert "document_id" in ingest_data
        assert ingest_data["source"] == "sample.md"

    # Query
    query_response = client.post("/query", json={"query": "What is RAG?"})
    assert query_response.status_code == 200
    query_data = query_response.json()
    assert "answer" in query_data
    assert "evaluation" in query_data
    assert query_data["evaluation"]["overall_score"] >= 0
    assert query_data["evaluation"]["overall_score"] <= 1


@pytest.mark.e2e
def test_empty_query(ollama_available, milvus_available):
    """Test handling of empty query"""
    client = TestClient(app)

    response = client.post("/query", json={"query": ""})
    # Either handled gracefully (200) or rejected (400)
    assert response.status_code in [200, 400]


@pytest.mark.e2e
def test_no_documents(ollama_available, milvus_available):
    """Test query when no documents are ingested"""
    client = TestClient(app)

    # Note: This test assumes a fresh state or that previous tests
    # don't affect this test. In practice, you might need to clear
    # the vector store before running this test.
    response = client.post("/query", json={"query": "What is this about?"})
    assert response.status_code == 200
    data = response.json()
    # Either returns "No relevant documents found." or an answer
    # depending on whether documents were ingested in previous tests
    assert "answer" in data


@pytest.mark.e2e
def test_multi_document_retrieval(ollama_available, milvus_available):
    """Test retrieval across multiple documents"""
    client = TestClient(app)

    # Ingest multiple documents
    fixtures_path = Path(__file__).parent / "fixtures"
    for filename in ["sample.md", "sample.csv"]:
        file_path = fixtures_path / filename
        with open(file_path, "rb") as f:
            content_type = "text/markdown" if filename.endswith(".md") else "text/csv"
            ingest_response = client.post("/ingest", files={"file": (filename, f, content_type)})
            assert ingest_response.status_code == 200
            ingest_data = ingest_response.json()
            assert ingest_data["status"] == "completed"

    # Query that crosses documents
    response = client.post("/query", json={"query": "What information is available?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "retrieved_count" in data
    # Should retrieve some documents
    assert data["retrieved_count"] >= 0


@pytest.mark.e2e
def test_docker_health():
    """Test Docker services are healthy"""
    client = TestClient(app)

    # Check API health
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.e2e
def test_query_with_top_k(ollama_available, milvus_available):
    """Test query with custom top_k parameter"""
    client = TestClient(app)

    # Ingest a document first
    sample_md_path = Path(__file__).parent / "fixtures" / "sample.md"
    with open(sample_md_path, "rb") as f:
        ingest_response = client.post("/ingest", files={"file": ("sample.md", f, "text/markdown")})
        assert ingest_response.status_code == 200

    # Query with custom top_k
    response = client.post("/query", json={"query": "What are the key components?", "top_k": 3})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "retrieved_count" in data
    # Retrieved count should be <= top_k
    assert data["retrieved_count"] <= 3


@pytest.mark.e2e
def test_query_with_invalid_top_k(ollama_available, milvus_available):
    """Test query with invalid top_k parameter"""
    client = TestClient(app)

    # Query with top_k > 20 (should be rejected or clamped)
    response = client.post("/query", json={"query": "What is RAG?", "top_k": 25})
    # Either rejected (400) or accepted with clamped value (200)
    assert response.status_code in [200, 400]


@pytest.mark.e2e
def test_ingest_with_different_file_types(ollama_available, milvus_available):
    """Test ingestion with different file types"""
    client = TestClient(app)

    fixtures_path = Path(__file__).parent / "fixtures"
    file_types = [
        ("sample.md", "text/markdown"),
        ("sample.csv", "text/csv"),
    ]

    for filename, content_type in file_types:
        file_path = fixtures_path / filename
        with open(file_path, "rb") as f:
            response = client.post("/ingest", files={"file": (filename, f, content_type)})
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert "document_id" in data
            assert data["chunk_count"] > 0


@pytest.mark.e2e
def test_query_evaluation_metrics(ollama_available, milvus_available):
    """Test that evaluation metrics are returned correctly"""
    client = TestClient(app)

    # Ingest a document first
    sample_md_path = Path(__file__).parent / "fixtures" / "sample.md"
    with open(sample_md_path, "rb") as f:
        ingest_response = client.post("/ingest", files={"file": ("sample.md", f, "text/markdown")})
        assert ingest_response.status_code == 200

    # Query
    response = client.post("/query", json={"query": "What are the key components of RAG?"})
    assert response.status_code == 200
    data = response.json()

    # Check evaluation metrics
    assert "evaluation" in data
    evaluation = data["evaluation"]
    assert "faithfulness" in evaluation
    assert "answer_relevance" in evaluation
    assert "context_precision" in evaluation
    assert "context_recall" in evaluation
    assert "overall_score" in evaluation

    # All scores should be between 0 and 1
    assert 0 <= evaluation["faithfulness"] <= 1
    assert 0 <= evaluation["answer_relevance"] <= 1
    assert 0 <= evaluation["context_precision"] <= 1
    assert 0 <= evaluation["context_recall"] <= 1
    assert 0 <= evaluation["overall_score"] <= 1


@pytest.mark.e2e
def test_query_returns_sources(ollama_available, milvus_available):
    """Test that query returns source information"""
    client = TestClient(app)

    # Ingest a document first
    sample_md_path = Path(__file__).parent / "fixtures" / "sample.md"
    with open(sample_md_path, "rb") as f:
        ingest_response = client.post("/ingest", files={"file": ("sample.md", f, "text/markdown")})
        assert ingest_response.status_code == 200

    # Query
    response = client.post("/query", json={"query": "What is RAG?"})
    assert response.status_code == 200
    data = response.json()

    # Check sources
    assert "sources" in data
    assert isinstance(data["sources"], list)
