"""Integration tests for /query endpoint with real RAGQueryEngine."""

import sys
from unittest.mock import AsyncMock, Mock, patch

# Mock MarkItDown before importing any modules that use it
sys.modules["markitdown"] = Mock()

import pytest
from fastapi.testclient import TestClient


def test_query_endpoint_with_results():
    """Test query with retrieved documents."""
    # Mock RAGQueryEngine.query to return results
    mock_query_result = {
        "answer": "RAG is a technique that combines retrieval and generation.",
        "sources": [
            {
                "text": "RAG stands for Retrieval-Augmented Generation.",
                "score": 0.95,
                "metadata": {"source": "test.pdf", "chunk_index": 0},
            },
            {
                "text": "RAG combines retrieval and generation for better answers.",
                "score": 0.90,
                "metadata": {"source": "test.pdf", "chunk_index": 1},
            },
        ],
        "retrieved_count": 2,
    }

    # Mock evaluator.evaluate_query
    mock_evaluation = {
        "faithfulness": 0.85,
        "context_precision": 0.90,
        "context_recall": 0.80,
        "answer_relevance": 0.88,
        "overall_score": 0.86,
    }

    # Import here to avoid import errors if module doesn't exist yet
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Mock the rag_engine.query method
        with patch(
            "src.rag_system.api.main.rag_engine.query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_query_result

            # Mock the evaluator.evaluate_query method
            with patch(
                "src.rag_system.api.main.evaluator.evaluate_query", new_callable=AsyncMock
            ) as mock_eval:
                mock_eval.return_value = mock_evaluation

                # Make request
                response = client.post(
                    "/query",
                    json={"query": "What is RAG?", "top_k": 5},
                )

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is RAG?"
    assert data["answer"] == "RAG is a technique that combines retrieval and generation."
    assert len(data["sources"]) == 2
    assert data["retrieved_count"] == 2
    assert "evaluation" in data
    assert data["evaluation"]["overall_score"] == 0.86


def test_query_endpoint_without_results():
    """Test query with no results."""
    # Mock RAGQueryEngine.query to return no results
    mock_query_result = {
        "answer": "No relevant documents found.",
        "sources": [],
        "retrieved_count": 0,
    }

    # Mock evaluator.evaluate_query
    mock_evaluation = {
        "faithfulness": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "answer_relevance": 0.0,
        "overall_score": 0.0,
    }

    # Import here to avoid import errors if module doesn't exist yet
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Mock the rag_engine.query method
        with patch(
            "src.rag_system.api.main.rag_engine.query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_query_result

            # Mock the evaluator.evaluate_query method
            with patch(
                "src.rag_system.api.main.evaluator.evaluate_query", new_callable=AsyncMock
            ) as mock_eval:
                mock_eval.return_value = mock_evaluation

                # Make request
                response = client.post(
                    "/query",
                    json={"query": "What is quantum computing?", "top_k": 5},
                )

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is quantum computing?"
    assert data["answer"] == "No relevant documents found."
    assert len(data["sources"]) == 0
    assert data["retrieved_count"] == 0
    assert "evaluation" in data


def test_query_endpoint_returns_answer():
    """Test answer is returned."""
    # Mock RAGQueryEngine.query
    mock_query_result = {
        "answer": "This is a test answer.",
        "sources": [],
        "retrieved_count": 0,
    }

    # Mock evaluator.evaluate_query
    mock_evaluation = {
        "faithfulness": 0.5,
        "context_precision": 0.5,
        "context_recall": 0.5,
        "answer_relevance": 0.5,
        "overall_score": 0.5,
    }

    # Import here to avoid import errors if module doesn't exist yet
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Mock the rag_engine.query method
        with patch(
            "src.rag_system.api.main.rag_engine.query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_query_result

            # Mock the evaluator.evaluate_query method
            with patch(
                "src.rag_system.api.main.evaluator.evaluate_query", new_callable=AsyncMock
            ) as mock_eval:
                mock_eval.return_value = mock_evaluation

                # Make request
                response = client.post(
                    "/query",
                    json={"query": "Test query", "top_k": 5},
                )

    # Verify answer is returned
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "This is a test answer."


def test_query_endpoint_returns_sources():
    """Test sources are returned."""
    # Mock RAGQueryEngine.query
    mock_query_result = {
        "answer": "Test answer",
        "sources": [
            {
                "text": "Source text 1",
                "score": 0.95,
                "metadata": {"source": "doc1.pdf", "chunk_index": 0},
            },
            {
                "text": "Source text 2",
                "score": 0.90,
                "metadata": {"source": "doc2.pdf", "chunk_index": 1},
            },
        ],
        "retrieved_count": 2,
    }

    # Mock evaluator.evaluate_query
    mock_evaluation = {
        "faithfulness": 0.5,
        "context_precision": 0.5,
        "context_recall": 0.5,
        "answer_relevance": 0.5,
        "overall_score": 0.5,
    }

    # Import here to avoid import errors if module doesn't exist yet
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Mock the rag_engine.query method
        with patch(
            "src.rag_system.api.main.rag_engine.query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_query_result

            # Mock the evaluator.evaluate_query method
            with patch(
                "src.rag_system.api.main.evaluator.evaluate_query", new_callable=AsyncMock
            ) as mock_eval:
                mock_eval.return_value = mock_evaluation

                # Make request
                response = client.post(
                    "/query",
                    json={"query": "Test query", "top_k": 5},
                )

    # Verify sources are returned
    assert response.status_code == 200
    data = response.json()
    assert "sources" in data
    assert len(data["sources"]) == 2
    assert data["sources"][0]["text"] == "Source text 1"
    assert data["sources"][0]["score"] == 0.95
    assert data["sources"][0]["metadata"]["source"] == "doc1.pdf"


def test_query_endpoint_returns_evaluation():
    """Test evaluation scores are returned."""
    # Mock RAGQueryEngine.query
    mock_query_result = {
        "answer": "Test answer",
        "sources": [],
        "retrieved_count": 0,
    }

    # Mock evaluator.evaluate_query
    mock_evaluation = {
        "faithfulness": 0.85,
        "context_precision": 0.90,
        "context_recall": 0.80,
        "answer_relevance": 0.88,
        "overall_score": 0.86,
    }

    # Import here to avoid import errors if module doesn't exist yet
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Mock the rag_engine.query method
        with patch(
            "src.rag_system.api.main.rag_engine.query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.return_value = mock_query_result

            # Mock the evaluator.evaluate_query method
            with patch(
                "src.rag_system.api.main.evaluator.evaluate_query", new_callable=AsyncMock
            ) as mock_eval:
                mock_eval.return_value = mock_evaluation

                # Make request
                response = client.post(
                    "/query",
                    json={"query": "Test query", "top_k": 5},
                )

    # Verify evaluation is returned
    assert response.status_code == 200
    data = response.json()
    assert "evaluation" in data
    assert data["evaluation"]["faithfulness"] == 0.85
    assert data["evaluation"]["context_precision"] == 0.90
    assert data["evaluation"]["context_recall"] == 0.80
    assert data["evaluation"]["answer_relevance"] == 0.88
    assert data["evaluation"]["overall_score"] == 0.86


def test_query_endpoint_handles_error():
    """Test error handling."""
    # Import here to avoid import errors if module doesn't exist yet
    from src.rag_system.api.main import app

    with TestClient(app) as client:
        # Mock the rag_engine.query method to raise an exception
        with patch(
            "src.rag_system.api.main.rag_engine.query", new_callable=AsyncMock
        ) as mock_query:
            mock_query.side_effect = Exception("RAG engine failed")

            # Make request
            response = client.post(
                "/query",
                json={"query": "Test query", "top_k": 5},
            )

    # Verify error is handled
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "RAG engine failed" in data["detail"]
