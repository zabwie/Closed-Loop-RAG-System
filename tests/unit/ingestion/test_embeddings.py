"""Unit tests for EmbeddingService."""

import sys
from unittest.mock import MagicMock, AsyncMock
import pytest

# Mock MarkItDown before importing to avoid RuntimeWarning during test collection
sys.modules["markitdown"] = MagicMock()
sys.modules["markitdown._markitdown"] = MagicMock()

from rag_system.ingestion.embeddings import EmbeddingService


class TestEmbeddingServiceInitialization:
    """Test EmbeddingService initialization."""

    def test_embedding_service_initialization_defaults(self):
        """Test service initialization with default parameters."""
        service = EmbeddingService(ollama_url="http://localhost:11434")
        assert service.model == "nomic-embed-text"
        assert service.client is not None

    def test_embedding_service_initialization_custom(self):
        """Test service initialization with custom parameters."""
        service = EmbeddingService(ollama_url="http://localhost:11434", model="custom-model")
        assert service.model == "custom-model"
        assert service.client is not None


class TestEmbeddingServiceEmbed:
    """Test EmbeddingService embed functionality."""

    @pytest.mark.asyncio
    async def test_embed_single_text(self, mocker):
        """Test embedding a single text."""
        # Mock the Ollama client
        mock_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_client.embed = AsyncMock(return_value=mock_response)

        service = EmbeddingService(ollama_url="http://localhost:11434")
        service.client = mock_client

        result = await service.embed(["test text"])
        assert result == [[0.1, 0.2, 0.3, 0.4, 0.5]]
        mock_client.embed.assert_called_once_with(input=["test text"], model="nomic-embed-text")

    @pytest.mark.asyncio
    async def test_embed_multiple_texts(self, mocker):
        """Test embedding multiple texts (batch)."""
        # Mock the Ollama client
        mock_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        mock_client.embed = AsyncMock(return_value=mock_response)

        service = EmbeddingService(ollama_url="http://localhost:11434")
        service.client = mock_client

        result = await service.embed(["text1", "text2", "text3"])
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        mock_client.embed.assert_called_once_with(
            input=["text1", "text2", "text3"], model="nomic-embed-text"
        )

    @pytest.mark.asyncio
    async def test_embed_returns_correct_dimensions(self, mocker):
        """Test embedding dimensions are correct."""
        # Mock the Ollama client
        mock_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        # nomic-embed-text produces 768-dimensional embeddings
        mock_response.embeddings = [[0.0] * 768]
        mock_client.embed = AsyncMock(return_value=mock_response)

        service = EmbeddingService(ollama_url="http://localhost:11434")
        service.client = mock_client

        result = await service.embed(["test"])
        assert len(result) == 1
        assert len(result[0]) == 768
        assert all(isinstance(x, float) for x in result[0])

    @pytest.mark.asyncio
    async def test_embed_handles_empty_list(self, mocker):
        """Test handling of empty text list."""
        # Mock the Ollama client
        mock_client = mocker.MagicMock()
        mock_response = mocker.MagicMock()
        mock_response.embeddings = []
        mock_client.embed = AsyncMock(return_value=mock_response)

        service = EmbeddingService(ollama_url="http://localhost:11434")
        service.client = mock_client

        result = await service.embed([])
        assert result == []
        mock_client.embed.assert_called_once_with(input=[], model="nomic-embed-text")

    @pytest.mark.asyncio
    async def test_embed_handles_connection_error(self, mocker):
        """Test error handling for connection failures."""
        # Mock the Ollama client to raise connection error
        mock_client = mocker.MagicMock()
        mock_client.embed = AsyncMock(side_effect=ConnectionError("Failed to connect to Ollama"))

        service = EmbeddingService(ollama_url="http://localhost:11434")
        service.client = mock_client

        with pytest.raises(ConnectionError):
            await service.embed(["test text"])

    @pytest.mark.asyncio
    async def test_embed_handles_timeout(self, mocker):
        """Test error handling for timeouts."""
        # Mock the Ollama client to raise timeout error
        mock_client = mocker.MagicMock()
        mock_client.embed = AsyncMock(side_effect=TimeoutError("Embedding generation timed out"))

        service = EmbeddingService(ollama_url="http://localhost:11434")
        service.client = mock_client

        with pytest.raises(TimeoutError):
            await service.embed(["test text"])

    @pytest.mark.asyncio
    async def test_embed_handles_generic_error(self, mocker):
        """Test error handling for generic errors."""
        # Mock the Ollama client to raise generic error
        mock_client = mocker.MagicMock()
        mock_client.embed = AsyncMock(side_effect=Exception("Generic embedding error"))

        service = EmbeddingService(ollama_url="http://localhost:11434")
        service.client = mock_client

        with pytest.raises(Exception, match="Generic embedding error"):
            await service.embed(["test text"])


class TestEmbeddingServiceEmbedSingle:
    """Test EmbeddingService embed_single functionality."""

    @pytest.mark.asyncio
    async def test_embed_single_calls_embed(self, mocker):
        """Test embed_single calls embed internally."""
        # Mock the embed method
        mock_embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        service = EmbeddingService(ollama_url="http://localhost:11434")
        service.embed = mock_embed

        result = await service.embed_single("test text")
        assert result == [0.1, 0.2, 0.3]
        mock_embed.assert_called_once_with(["test text"])

    @pytest.mark.asyncio
    async def test_embed_single_returns_first_embedding(self, mocker):
        """Test embed_single returns the first embedding from the list."""
        # Mock the embed method
        mock_embed = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

        service = EmbeddingService(ollama_url="http://localhost:11434")
        service.embed = mock_embed

        result = await service.embed_single("test text")
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_single_propagates_errors(self, mocker):
        """Test embed_single propagates errors from embed."""
        # Mock the embed method to raise error
        mock_embed = AsyncMock(side_effect=ConnectionError("Failed to connect"))

        service = EmbeddingService(ollama_url="http://localhost:11434")
        service.embed = mock_embed

        with pytest.raises(ConnectionError):
            await service.embed_single("test text")
