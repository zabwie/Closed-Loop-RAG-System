"""Unit tests for OllamaClient."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from httpx import HTTPStatusError, RequestError, TimeoutException

from rag_system.generation.ollama_client import OllamaClient


class TestOllamaClientInitialization:
    """Tests for OllamaClient initialization."""

    def test_ollama_client_initialization_with_defaults(self):
        """Test client initialization with default parameters."""
        client = OllamaClient(base_url="http://localhost:11434")
        assert client.base_url == "http://localhost:11434"
        assert client.model == "llama3:8b"

    def test_ollama_client_initialization_with_custom_model(self):
        """Test client initialization with custom model."""
        client = OllamaClient(base_url="http://localhost:11434", model="llama3:70b")
        assert client.base_url == "http://localhost:11434"
        assert client.model == "llama3:70b"


class TestOllamaClientChat:
    """Tests for OllamaClient.chat method."""

    @pytest.mark.asyncio
    async def test_chat_with_context(self, mocker):
        """Test chat with context."""
        # Mock httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Test response"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)

        mocker.patch("rag_system.generation.ollama_client.AsyncClient", return_value=mock_client)

        client = OllamaClient(base_url="http://localhost:11434")
        response = await client.chat(
            "What is RAG?", context="RAG stands for Retrieval-Augmented Generation."
        )

        assert response == "Test response"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/chat"
        assert call_args[1]["json"]["model"] == "llama3:8b"
        assert (
            "RAG stands for Retrieval-Augmented Generation."
            in call_args[1]["json"]["messages"][0]["content"]
        )
        assert "What is RAG?" in call_args[1]["json"]["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_chat_without_context(self, mocker):
        """Test chat without context."""
        # Mock httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Test response"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)

        mocker.patch("rag_system.generation.ollama_client.AsyncClient", return_value=mock_client)

        client = OllamaClient(base_url="http://localhost:11434")
        response = await client.chat("What is RAG?")

        assert response == "Test response"
        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://localhost:11434/api/chat"
        assert call_args[1]["json"]["model"] == "llama3:8b"
        assert "What is RAG?" in call_args[1]["json"]["messages"][0]["content"]

    @pytest.mark.asyncio
    async def test_chat_returns_response(self, mocker):
        """Test response is returned correctly."""
        # Mock httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "message": {"content": "This is the generated response."}
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)

        mocker.patch("rag_system.generation.ollama_client.AsyncClient", return_value=mock_client)

        client = OllamaClient(base_url="http://localhost:11434")
        response = await client.chat("Test prompt")

        assert response == "This is the generated response."
        assert isinstance(response, str)

    @pytest.mark.asyncio
    async def test_chat_handles_connection_error(self, mocker):
        """Test error handling for connection failures."""
        # Mock httpx.AsyncClient to raise RequestError
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(side_effect=RequestError("Connection failed"))

        mocker.patch("rag_system.generation.ollama_client.AsyncClient", return_value=mock_client)

        client = OllamaClient(base_url="http://localhost:11434")
        with pytest.raises(RequestError):
            await client.chat("Test prompt")

    @pytest.mark.asyncio
    async def test_chat_handles_timeout(self, mocker):
        """Test error handling for timeouts."""
        # Mock httpx.AsyncClient to raise TimeoutException
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(side_effect=TimeoutException("Request timed out"))

        mocker.patch("rag_system.generation.ollama_client.AsyncClient", return_value=mock_client)

        client = OllamaClient(base_url="http://localhost:11434")
        with pytest.raises(TimeoutException):
            await client.chat("Test prompt")

    @pytest.mark.asyncio
    async def test_chat_handles_generic_error(self, mocker):
        """Test error handling for generic errors."""
        # Mock httpx.AsyncClient to raise generic Exception
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(side_effect=Exception("Generic error"))

        mocker.patch("rag_system.generation.ollama_client.AsyncClient", return_value=mock_client)

        client = OllamaClient(base_url="http://localhost:11434")
        with pytest.raises(Exception, match="Generic error"):
            await client.chat("Test prompt")

    @pytest.mark.asyncio
    async def test_chat_uses_correct_model(self, mocker):
        """Test correct model is used in request."""
        # Mock httpx.AsyncClient
        mock_response = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Test response"}}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None
        mock_client.post = AsyncMock(return_value=mock_response)

        mocker.patch("rag_system.generation.ollama_client.AsyncClient", return_value=mock_client)

        client = OllamaClient(base_url="http://localhost:11434", model="llama3:70b")
        await client.chat("Test prompt")

        call_args = mock_client.post.call_args
        assert call_args[1]["json"]["model"] == "llama3:70b"
