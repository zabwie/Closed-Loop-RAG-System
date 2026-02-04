"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch


def test_query_endpoint_with_evaluation():
    """Test that /query endpoint returns evaluation scores."""
    # Mock the RAG engine response
    mock_rag_response = {
        "answer": "RAG stands for Retrieval-Augmented Generation, a technique that combines retrieval and generation.",
        "sources": [
            {"text": "RAG is Retrieval-Augmented Generation", "score": 0.9},
            {"text": "It combines retrieval and generation", "score": 0.85},
        ],
        "retrieved_count": 2,
    }

    # Import here to avoid import errors if module doesn't exist yet
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Mock the rag_engine.query method
        with patch(
            "src.rag_system.api.main.rag_engine.query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_rag_response

            # Make request to /query endpoint
            response = client.post("/query", json={"query": "What is RAG?", "top_k": 5})

            # Verify response status
            assert response.status_code == 200

            # Verify response structure
            data = response.json()
            assert "query" in data
            assert "answer" in data
            assert "sources" in data
            assert "retrieved_count" in data
            assert "evaluation" in data

            # Verify evaluation metrics are present
            evaluation = data["evaluation"]
            assert "faithfulness" in evaluation
            assert "context_precision" in evaluation
            assert "context_recall" in evaluation
            assert "answer_relevance" in evaluation
            assert "overall_score" in evaluation

            # Verify all scores are between 0 and 1
            assert 0 <= evaluation["faithfulness"] <= 1
            assert 0 <= evaluation["context_precision"] <= 1
            assert 0 <= evaluation["context_recall"] <= 1
            assert 0 <= evaluation["answer_relevance"] <= 1
            assert 0 <= evaluation["overall_score"] <= 1

            # Verify rag_engine.query was called with correct arguments
            mock_query.assert_called_once_with("What is RAG?", 5)


def test_query_endpoint_error_handling():
    """Test that /query endpoint handles errors gracefully."""
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Mock the rag_engine.query method to raise an exception
        with patch(
            "src.rag_system.api.main.rag_engine.query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.side_effect = Exception("RAG engine error")

            # Make request to /query endpoint
            response = client.post("/query", json={"query": "What is RAG?", "top_k": 5})

            # Verify response status is 500
            assert response.status_code == 500

            # Verify error message is returned
            data = response.json()
            assert "detail" in data
            assert "RAG engine error" in data["detail"]
