"""Unit tests for RAGQueryEngine."""

import sys
from unittest.mock import AsyncMock, Mock

# Mock MarkItDown to avoid ffmpeg dependency
sys.modules["markitdown"] = Mock()
sys.modules["markitdown._markitdown"] = Mock()

import pytest
from rag_system.generation.rag_engine import RAGQueryEngine


class TestRAGQueryEngineInitialization:
    """Tests for RAGQueryEngine initialization."""

    def test_rag_query_engine_initialization(self, mocker):
        """Test that RAGQueryEngine initializes with required dependencies."""
        # Create mock dependencies
        mock_ollama = Mock()
        mock_vector_store = Mock()
        mock_embeddings = Mock()

        # Initialize engine
        engine = RAGQueryEngine(
            ollama=mock_ollama, vector_store=mock_vector_store, embeddings=mock_embeddings
        )

        # Verify dependencies are stored
        assert engine.ollama is mock_ollama
        assert engine.vector_store is mock_vector_store
        assert engine.embeddings is mock_embeddings


class TestRAGQueryEngineQuery:
    """Tests for RAGQueryEngine query method."""

    @pytest.mark.asyncio
    async def test_query_with_results(self, mocker):
        """Test query with retrieved documents."""
        # Create mock dependencies
        mock_ollama = Mock()
        mock_vector_store = Mock()
        mock_embeddings = Mock()

        # Setup mocks
        mock_embeddings.embed_single = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_vector_store.search = AsyncMock(
            return_value=[
                {"text": "Document 1", "score": 0.9, "metadata": {"source": "doc1.pdf"}},
                {"text": "Document 2", "score": 0.8, "metadata": {"source": "doc2.pdf"}},
            ]
        )
        mock_ollama.chat = AsyncMock(return_value="This is the answer.")

        # Initialize engine
        engine = RAGQueryEngine(
            ollama=mock_ollama, vector_store=mock_vector_store, embeddings=mock_embeddings
        )

        # Execute query
        result = await engine.query("What is RAG?", top_k=5)

        # Verify result structure
        assert "answer" in result
        assert "sources" in result
        assert "retrieved_count" in result
        assert result["answer"] == "This is the answer."
        assert result["retrieved_count"] == 2
        assert len(result["sources"]) == 2

        # Verify mocks were called correctly
        mock_embeddings.embed_single.assert_called_once_with("What is RAG?")
        mock_vector_store.search.assert_called_once_with([0.1, 0.2, 0.3], top_k=5)
        mock_ollama.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_without_results(self, mocker):
        """Test query with no results."""
        # Create mock dependencies
        mock_ollama = Mock()
        mock_vector_store = Mock()
        mock_embeddings = Mock()

        # Setup mocks
        mock_embeddings.embed_single = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_vector_store.search = AsyncMock(return_value=[])

        # Initialize engine
        engine = RAGQueryEngine(
            ollama=mock_ollama, vector_store=mock_vector_store, embeddings=mock_embeddings
        )

        # Execute query
        result = await engine.query("What is RAG?", top_k=5)

        # Verify result structure
        assert result["answer"] == "No relevant documents found."
        assert result["sources"] == []
        assert result["retrieved_count"] == 0

        # Verify ollama.chat was NOT called
        mock_ollama.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_returns_answer(self, mocker):
        """Test that query returns answer from Ollama."""
        # Create mock dependencies
        mock_ollama = Mock()
        mock_vector_store = Mock()
        mock_embeddings = Mock()

        # Setup mocks
        mock_embeddings.embed_single = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_vector_store.search = AsyncMock(
            return_value=[{"text": "Document 1", "score": 0.9, "metadata": {}}]
        )
        mock_ollama.chat = AsyncMock(return_value="The answer is 42.")

        # Initialize engine
        engine = RAGQueryEngine(
            ollama=mock_ollama, vector_store=mock_vector_store, embeddings=mock_embeddings
        )

        # Execute query
        result = await engine.query("What is the answer?")

        # Verify answer is returned
        assert result["answer"] == "The answer is 42."

    @pytest.mark.asyncio
    async def test_query_returns_sources(self, mocker):
        """Test that query returns sources with text, score, and metadata."""
        # Create mock dependencies
        mock_ollama = Mock()
        mock_vector_store = Mock()
        mock_embeddings = Mock()

        # Setup mocks
        mock_embeddings.embed_single = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_vector_store.search = AsyncMock(
            return_value=[
                {"text": "Document 1", "score": 0.9, "metadata": {"source": "doc1.pdf", "page": 1}},
                {"text": "Document 2", "score": 0.8, "metadata": {"source": "doc2.pdf", "page": 2}},
            ]
        )
        mock_ollama.chat = AsyncMock(return_value="Answer.")

        # Initialize engine
        engine = RAGQueryEngine(
            ollama=mock_ollama, vector_store=mock_vector_store, embeddings=mock_embeddings
        )

        # Execute query
        result = await engine.query("Question?")

        # Verify sources are returned correctly
        assert len(result["sources"]) == 2
        assert result["sources"][0]["text"] == "Document 1"
        assert result["sources"][0]["score"] == 0.9
        assert result["sources"][0]["metadata"] == {"source": "doc1.pdf", "page": 1}
        assert result["sources"][1]["text"] == "Document 2"
        assert result["sources"][1]["score"] == 0.8
        assert result["sources"][1]["metadata"] == {"source": "doc2.pdf", "page": 2}

    @pytest.mark.asyncio
    async def test_query_returns_retrieved_count(self, mocker):
        """Test that query returns correct retrieved_count."""
        # Create mock dependencies
        mock_ollama = Mock()
        mock_vector_store = Mock()
        mock_embeddings = Mock()

        # Setup mocks
        mock_embeddings.embed_single = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_vector_store.search = AsyncMock(
            return_value=[
                {"text": "Doc 1", "score": 0.9, "metadata": {}},
                {"text": "Doc 2", "score": 0.8, "metadata": {}},
                {"text": "Doc 3", "score": 0.7, "metadata": {}},
            ]
        )
        mock_ollama.chat = AsyncMock(return_value="Answer.")

        # Initialize engine
        engine = RAGQueryEngine(
            ollama=mock_ollama, vector_store=mock_vector_store, embeddings=mock_embeddings
        )

        # Execute query
        result = await engine.query("Question?")

        # Verify retrieved_count
        assert result["retrieved_count"] == 3

    @pytest.mark.asyncio
    async def test_query_uses_top_k(self, mocker):
        """Test that query uses top_k parameter correctly."""
        # Create mock dependencies
        mock_ollama = Mock()
        mock_vector_store = Mock()
        mock_embeddings = Mock()

        # Setup mocks
        mock_embeddings.embed_single = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_vector_store.search = AsyncMock(
            return_value=[{"text": "Doc 1", "score": 0.9, "metadata": {}}]
        )
        mock_ollama.chat = AsyncMock(return_value="Answer.")

        # Initialize engine
        engine = RAGQueryEngine(
            ollama=mock_ollama, vector_store=mock_vector_store, embeddings=mock_embeddings
        )

        # Execute query with custom top_k
        await engine.query("Question?", top_k=10)

        # Verify top_k was passed to search
        mock_vector_store.search.assert_called_once_with([0.1, 0.2, 0.3], top_k=10)

    @pytest.mark.asyncio
    async def test_query_assembles_context(self, mocker):
        """Test that query assembles context correctly from retrieved documents."""
        # Create mock dependencies
        mock_ollama = Mock()
        mock_vector_store = Mock()
        mock_embeddings = Mock()

        # Setup mocks
        mock_embeddings.embed_single = AsyncMock(return_value=[0.1, 0.2, 0.3])
        mock_vector_store.search = AsyncMock(
            return_value=[
                {"text": "First document", "score": 0.9, "metadata": {}},
                {"text": "Second document", "score": 0.8, "metadata": {}},
            ]
        )
        mock_ollama.chat = AsyncMock(return_value="Answer.")

        # Initialize engine
        engine = RAGQueryEngine(
            ollama=mock_ollama, vector_store=mock_vector_store, embeddings=mock_embeddings
        )

        # Execute query
        await engine.query("Question?")

        # Verify context was assembled correctly
        expected_context = "First document\n\nSecond document"
        mock_ollama.chat.assert_called_once_with("Question?", expected_context)
