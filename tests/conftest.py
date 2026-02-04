"""
Shared pytest fixtures for Closed-Loop RAG System tests.

This file provides common fixtures used across unit, integration, and e2e tests.
Fixtures are automatically discovered by pytest and can be used in any test file.
"""

import pytest
import asyncio
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass


# ============================================================================
# Sample Data Fixtures
# ============================================================================


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """
    Sample documents for testing RAG functionality.

    Returns:
        List of document dictionaries with text and metadata.
    """
    return [
        {
            "id": "doc_001",
            "text": "Retrieval-Augmented Generation (RAG) combines retrieval systems with generative AI models.",
            "metadata": {"source": "wikipedia", "category": "ai", "author": "test_user"},
        },
        {
            "id": "doc_002",
            "text": "Milvus is a vector database designed for scalable similarity search and AI applications.",
            "metadata": {"source": "documentation", "category": "database", "author": "test_user"},
        },
        {
            "id": "doc_003",
            "text": "Ollama provides a simple way to run large language models locally.",
            "metadata": {"source": "blog", "category": "llm", "author": "test_user"},
        },
    ]


@pytest.fixture
def sample_query() -> str:
    """
    Sample query for testing retrieval.

    Returns:
        A test query string.
    """
    return "What is Retrieval-Augmented Generation?"


@pytest.fixture
def sample_embeddings() -> List[List[float]]:
    """
    Sample embeddings for testing vector operations.

    Returns:
        List of embedding vectors (768-dimensional).
    """
    return [
        [0.1, 0.2, 0.3] + [0.0] * 765,  # doc_001 embedding
        [0.4, 0.5, 0.6] + [0.0] * 765,  # doc_002 embedding
        [0.7, 0.8, 0.9] + [0.0] * 765,  # doc_003 embedding
    ]


# ============================================================================
# Mock Ollama Fixtures
# ============================================================================


@pytest.fixture
def mock_ollama_client():
    """
    Mock Ollama client for testing LLM interactions.

    Returns:
        MagicMock configured to simulate Ollama client behavior.
    """
    client = MagicMock()

    # Mock chat completion
    async def mock_chat(*args, **kwargs):
        return {
            "message": {"role": "assistant", "content": "This is a mock response from Ollama."},
            "model": "llama2",
            "done": True,
        }

    client.chat = AsyncMock(side_effect=mock_chat)

    # Mock embedding generation
    async def mock_embeddings(*args, **kwargs):
        return {"embedding": [0.1, 0.2, 0.3] + [0.0] * 765}

    client.embeddings = AsyncMock(side_effect=mock_embeddings)

    # Mock model list
    client.list.return_value = {
        "models": [
            {"name": "llama2", "modified_at": "2024-01-01"},
            {"name": "mistral", "modified_at": "2024-01-01"},
        ]
    }

    return client


@pytest.fixture
def mock_ollama_response():
    """
    Mock Ollama API response for testing.

    Returns:
        Dictionary simulating Ollama chat response.
    """
    return {
        "message": {
            "role": "assistant",
            "content": "This is a test response from the mocked Ollama service.",
        },
        "model": "llama2",
        "done": True,
        "eval_count": 42,
        "eval_duration": 1234567890,
    }


# ============================================================================
# Mock Milvus Fixtures
# ============================================================================


@pytest.fixture
def mock_milvus_client():
    """
    Mock Milvus client for testing vector database operations.

    Returns:
        MagicMock configured to simulate Milvus client behavior.
    """
    client = MagicMock()

    # Mock collection operations
    client.create_collection.return_value = None
    client.has_collection.return_value = True
    client.drop_collection.return_value = None

    # Mock insert operations
    client.insert.return_value = {"insert_count": 3, "ids": ["doc_001", "doc_002", "doc_003"]}

    # Mock search operations
    async def mock_search(*args, **kwargs):
        return [
            {
                "id": "doc_001",
                "distance": 0.1,
                "entity": {
                    "text": "Retrieval-Augmented Generation (RAG) combines retrieval systems with generative AI models.",
                    "metadata": {"source": "wikipedia", "category": "ai"},
                },
            },
            {
                "id": "doc_002",
                "distance": 0.3,
                "entity": {
                    "text": "Milvus is a vector database designed for scalable similarity search and AI applications.",
                    "metadata": {"source": "documentation", "category": "database"},
                },
            },
        ]

    client.search = AsyncMock(side_effect=mock_search)

    # Mock query operations
    async def mock_query(*args, **kwargs):
        return [
            {
                "id": "doc_001",
                "entity": {
                    "text": "Retrieval-Augmented Generation (RAG) combines retrieval systems with generative AI models.",
                    "metadata": {"source": "wikipedia", "category": "ai"},
                },
            }
        ]

    client.query = AsyncMock(side_effect=mock_query)

    # Mock delete operations
    client.delete.return_value = {"delete_count": 1}

    return client


@pytest.fixture
def mock_milvus_collection():
    """
    Mock Milvus collection for testing collection-level operations.

    Returns:
        MagicMock configured to simulate Milvus collection behavior.
    """
    collection = MagicMock()

    # Mock collection properties
    collection.name = "test_collection"
    collection.description = "Test collection for RAG system"
    collection.num_entities = 3

    # Mock collection operations
    collection.load.return_value = None
    collection.release.return_value = None
    collection.flush.return_value = None

    # Mock index operations
    collection.create_index.return_value = None
    collection.has_index.return_value = True
    collection.drop_index.return_value = None

    return collection


# ============================================================================
# Mock TruLens Fixtures
# ============================================================================


@pytest.fixture
def mock_trulens_feedback():
    """
    Mock TruLens feedback functions for testing RAG evaluation.

    Returns:
        MagicMock configured to simulate TruLens feedback behavior.
    """
    feedback = MagicMock()

    # Mock feedback scores
    feedback.relevance_score.return_value = 0.85
    feedback.faithfulness_score.return_value = 0.90
    feedback.answer_relevance_score.return_value = 0.88

    return feedback


# ============================================================================
# Test Configuration Fixtures
# ============================================================================


@pytest.fixture
def test_config():
    """
    Test configuration for RAG system components.

    Returns:
        Dictionary with test configuration values.
    """
    return {
        "ollama": {"base_url": "http://localhost:11434", "model": "llama2", "timeout": 30},
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "collection_name": "test_collection",
            "dimension": 768,
        },
        "retrieval": {"top_k": 3, "similarity_threshold": 0.7},
        "generation": {"temperature": 0.7, "max_tokens": 512},
    }


# ============================================================================
# Async Event Loop Fixture
# ============================================================================


@pytest.fixture
def event_loop():
    """
    Create an instance of the default event loop for each test case.

    This fixture is automatically used by pytest-asyncio when
    asyncio_mode is set to 'auto' in pytest.ini.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_configure(config):
    """
    Pytest configuration hook.

    Called after command line options have been parsed and all plugins
    and initial conftest files have been loaded.
    """
    # Register custom markers
    config.addinivalue_line("markers", "unit: Unit tests (fast, isolated)")
    config.addinivalue_line("markers", "integration: Integration tests (require external services)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (full system tests)")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU resources")
    config.addinivalue_line("markers", "async: Async tests")


def pytest_collection_modifyitems(config, items):
    """
    Pytest hook to modify test items after collection.

    Automatically adds markers based on test location.
    """
    for item in items:
        # Add markers based on file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)

        # Add async marker for async tests
        if asyncio.iscoroutinefunction(item.obj):
            item.add_marker(pytest.mark.asyncio)
