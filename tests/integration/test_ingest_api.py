"""Integration tests for /ingest API endpoint."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from io import BytesIO


def test_ingest_endpoint_accepts_file():
    """Test that /ingest endpoint accepts file upload."""
    # Import here to avoid import errors if module doesn't exist yet
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Create a test file
        file_content = b"This is a test document for ingestion."
        files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}

        # Make request to /ingest endpoint
        response = client.post("/ingest", files=files)

        # Verify response status
        assert response.status_code == 200

        # Verify response structure
        data = response.json()
        assert "document_id" in data
        assert "status" in data
        assert "chunk_count" in data
        assert "source" in data


def test_ingest_endpoint_returns_document_id():
    """Test that /ingest endpoint returns document_id."""
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Create a test file
        file_content = b"Test document for document_id check."
        files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}

        # Make request to /ingest endpoint
        response = client.post("/ingest", files=files)

        # Verify response status
        assert response.status_code == 200

        # Verify document_id is returned
        data = response.json()
        assert data["document_id"] is not None
        assert isinstance(data["document_id"], str)
        assert len(data["document_id"]) > 0


def test_ingest_endpoint_returns_status():
    """Test that /ingest endpoint returns status."""
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Create a test file
        file_content = b"Test document for status check."
        files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}

        # Make request to /ingest endpoint
        response = client.post("/ingest", files=files)

        # Verify response status
        assert response.status_code == 200

        # Verify status is returned
        data = response.json()
        assert data["status"] is not None
        assert isinstance(data["status"], str)
        assert data["status"] in ["completed", "failed"]


def test_ingest_endpoint_returns_chunk_count():
    """Test that /ingest endpoint returns chunk_count."""
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Create a test file
        file_content = b"Test document for chunk_count check."
        files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}

        # Make request to /ingest endpoint
        response = client.post("/ingest", files=files)

        # Verify response status
        assert response.status_code == 200

        # Verify chunk_count is returned
        data = response.json()
        assert "chunk_count" in data
        if data["status"] == "completed":
            assert data["chunk_count"] is not None
            assert isinstance(data["chunk_count"], int)
            assert data["chunk_count"] >= 0


def test_ingest_endpoint_returns_source():
    """Test that /ingest endpoint returns source filename."""
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Create a test file
        file_content = b"Test document for source check."
        files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}

        # Make request to /ingest endpoint
        response = client.post("/ingest", files=files)

        # Verify response status
        assert response.status_code == 200

        # Verify source is returned
        data = response.json()
        assert data["source"] is not None
        assert isinstance(data["source"], str)
        assert data["source"] == "test.txt"


def test_ingest_endpoint_handles_error():
    """Test that /ingest endpoint handles errors gracefully."""
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Create a test file
        file_content = b"Test document for error handling."
        files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}

        # Mock uuid.uuid4 to raise an exception
        with patch("src.rag_system.api.main.uuid.uuid4", side_effect=Exception("Ingestion failed")):
            # Make request to /ingest endpoint
            response = client.post("/ingest", files=files)

            # Verify response status is 500
            assert response.status_code == 500

            # Verify error message is returned
            data = response.json()
            assert "detail" in data
            assert "Ingestion failed" in data["detail"]


def test_ingest_endpoint_with_different_file_types():
    """Test that /ingest endpoint handles different file types."""
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Test with markdown file
        md_content = b"# Test Document\n\nThis is a test markdown file."
        files = {"file": ("test.md", BytesIO(md_content), "text/markdown")}

        response = client.post("/ingest", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "test.md"

        # Test with PDF file (mock content)
        pdf_content = b"%PDF-1.4\nmock pdf content"
        files = {"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")}

        response = client.post("/ingest", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "test.pdf"
