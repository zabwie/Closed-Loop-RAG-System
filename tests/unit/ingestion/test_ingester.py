"""Unit tests for DocumentIngester class."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Mock MarkItDown at module level to avoid RuntimeWarning
sys.modules["markitdown"] = MagicMock()

from rag_system.ingestion.ingester import DocumentIngester
from rag_system.ingestion.chunker import Chunk


class TestDocumentIngesterInitialization:
    """Test DocumentIngester initialization."""

    def test_ingester_initialization(self, mocker):
        """Test that DocumentIngester initializes with all dependencies."""
        # Create mock dependencies
        mock_converter = mocker.MagicMock()
        mock_chunker = mocker.MagicMock()
        mock_embedding_service = mocker.MagicMock()
        mock_milvus_client = mocker.MagicMock()

        # Initialize ingester
        ingester = DocumentIngester(
            converter=mock_converter,
            chunker=mock_chunker,
            embedding_service=mock_embedding_service,
            milvus_client=mock_milvus_client,
        )

        # Verify all dependencies are stored
        assert ingester.converter == mock_converter
        assert ingester.chunker == mock_chunker
        assert ingester.embedding_service == mock_embedding_service
        assert ingester.milvus_client == mock_milvus_client


class TestDocumentIngesterIngest:
    """Test DocumentIngester.ingest method."""

    @pytest.mark.asyncio
    async def test_ingest_successful(self, mocker):
        """Test successful ingestion flow."""
        # Create mock dependencies
        mock_converter = mocker.MagicMock()
        mock_chunker = mocker.MagicMock()
        mock_embedding_service = mocker.MagicMock()
        mock_milvus_client = mocker.MagicMock()

        # Setup mocks
        mock_converter.convert.return_value = {
            "markdown": "# Test Document\n\nThis is a test.",
            "metadata": {"source": "test.pdf", "format": "pdf", "char_count": 30, "word_count": 6},
        }
        mock_chunker.chunk.return_value = [
            Chunk(
                text="# Test Document",
                metadata={"source": "test.pdf", "chunk_index": 0, "char_count": 15},
            ),
            Chunk(
                text="This is a test.",
                metadata={"source": "test.pdf", "chunk_index": 1, "char_count": 15},
            ),
        ]
        mock_embedding_service.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_milvus_client.insert = AsyncMock()

        # Initialize ingester
        ingester = DocumentIngester(
            converter=mock_converter,
            chunker=mock_chunker,
            embedding_service=mock_embedding_service,
            milvus_client=mock_milvus_client,
        )

        # Ingest document
        result = await ingester.ingest(Path("test.pdf"))

        # Verify result
        assert result["status"] == "completed"
        assert result["chunk_count"] == 2
        assert result["source"] == "test.pdf"
        assert "document_id" in result

        # Verify converter was called
        mock_converter.convert.assert_called_once_with(Path("test.pdf"))

        # Verify chunker was called
        mock_chunker.chunk.assert_called_once_with("# Test Document\n\nThis is a test.", "test.pdf")

        # Verify embedding service was called
        mock_embedding_service.embed.assert_called_once_with(["# Test Document", "This is a test."])

        # Verify milvus insert was called
        mock_milvus_client.insert.assert_called_once()
        call_args = mock_milvus_client.insert.call_args
        assert call_args[1]["embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert call_args[1]["texts"] == ["# Test Document", "This is a test."]
        assert len(call_args[1]["metadatas"]) == 2
        assert "document_id" in call_args[1]["metadatas"][0]

    @pytest.mark.asyncio
    async def test_ingest_converter_fails(self, mocker):
        """Test handling of converter failures."""
        # Create mock dependencies
        mock_converter = mocker.MagicMock()
        mock_chunker = mocker.MagicMock()
        mock_embedding_service = mocker.MagicMock()
        mock_milvus_client = mocker.MagicMock()

        # Setup converter to raise exception
        mock_converter.convert.side_effect = Exception("Conversion failed")

        # Initialize ingester
        ingester = DocumentIngester(
            converter=mock_converter,
            chunker=mock_chunker,
            embedding_service=mock_embedding_service,
            milvus_client=mock_milvus_client,
        )

        # Ingest document
        result = await ingester.ingest(Path("test.pdf"))

        # Verify result
        assert result["status"] == "failed"
        assert "error" in result
        assert "Conversion failed" in result["error"]
        assert "document_id" in result

        # Verify chunker and embedding service were not called
        mock_chunker.chunk.assert_not_called()
        mock_embedding_service.embed.assert_not_called()
        mock_milvus_client.insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_chunker_fails(self, mocker):
        """Test handling of chunker failures."""
        # Create mock dependencies
        mock_converter = mocker.MagicMock()
        mock_chunker = mocker.MagicMock()
        mock_embedding_service = mocker.MagicMock()
        mock_milvus_client = mocker.MagicMock()

        # Setup mocks
        mock_converter.convert.return_value = {
            "markdown": "# Test Document",
            "metadata": {"source": "test.pdf", "format": "pdf", "char_count": 15, "word_count": 2},
        }
        mock_chunker.chunk.side_effect = Exception("Chunking failed")

        # Initialize ingester
        ingester = DocumentIngester(
            converter=mock_converter,
            chunker=mock_chunker,
            embedding_service=mock_embedding_service,
            milvus_client=mock_milvus_client,
        )

        # Ingest document
        result = await ingester.ingest(Path("test.pdf"))

        # Verify result
        assert result["status"] == "failed"
        assert "error" in result
        assert "Chunking failed" in result["error"]
        assert "document_id" in result

        # Verify embedding service and milvus were not called
        mock_embedding_service.embed.assert_not_called()
        mock_milvus_client.insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_embedding_fails(self, mocker):
        """Test handling of embedding failures."""
        # Create mock dependencies
        mock_converter = mocker.MagicMock()
        mock_chunker = mocker.MagicMock()
        mock_embedding_service = mocker.MagicMock()
        mock_milvus_client = mocker.MagicMock()

        # Setup mocks
        mock_converter.convert.return_value = {
            "markdown": "# Test Document",
            "metadata": {"source": "test.pdf", "format": "pdf", "char_count": 15, "word_count": 2},
        }
        mock_chunker.chunk.return_value = [
            Chunk(
                text="# Test Document",
                metadata={"source": "test.pdf", "chunk_index": 0, "char_count": 15},
            ),
        ]
        mock_embedding_service.embed = AsyncMock(side_effect=Exception("Embedding failed"))

        # Initialize ingester
        ingester = DocumentIngester(
            converter=mock_converter,
            chunker=mock_chunker,
            embedding_service=mock_embedding_service,
            milvus_client=mock_milvus_client,
        )

        # Ingest document
        result = await ingester.ingest(Path("test.pdf"))

        # Verify result
        assert result["status"] == "failed"
        assert "error" in result
        assert "Embedding failed" in result["error"]
        assert "document_id" in result

        # Verify milvus was not called
        mock_milvus_client.insert.assert_not_called()

    @pytest.mark.asyncio
    async def test_ingest_milvus_fails(self, mocker):
        """Test handling of Milvus failures."""
        # Create mock dependencies
        mock_converter = mocker.MagicMock()
        mock_chunker = mocker.MagicMock()
        mock_embedding_service = mocker.MagicMock()
        mock_milvus_client = mocker.MagicMock()

        # Setup mocks
        mock_converter.convert.return_value = {
            "markdown": "# Test Document",
            "metadata": {"source": "test.pdf", "format": "pdf", "char_count": 15, "word_count": 2},
        }
        mock_chunker.chunk.return_value = [
            Chunk(
                text="# Test Document",
                metadata={"source": "test.pdf", "chunk_index": 0, "char_count": 15},
            ),
        ]
        mock_embedding_service.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock_milvus_client.insert = AsyncMock(side_effect=Exception("Milvus insert failed"))

        # Initialize ingester
        ingester = DocumentIngester(
            converter=mock_converter,
            chunker=mock_chunker,
            embedding_service=mock_embedding_service,
            milvus_client=mock_milvus_client,
        )

        # Ingest document
        result = await ingester.ingest(Path("test.pdf"))

        # Verify result
        assert result["status"] == "failed"
        assert "error" in result
        assert "Milvus insert failed" in result["error"]
        assert "document_id" in result

    @pytest.mark.asyncio
    async def test_ingest_returns_document_id(self, mocker):
        """Test that document_id is returned."""
        # Create mock dependencies
        mock_converter = mocker.MagicMock()
        mock_chunker = mocker.MagicMock()
        mock_embedding_service = mocker.MagicMock()
        mock_milvus_client = mocker.MagicMock()

        # Setup mocks
        mock_converter.convert.return_value = {
            "markdown": "# Test Document",
            "metadata": {"source": "test.pdf", "format": "pdf", "char_count": 15, "word_count": 2},
        }
        mock_chunker.chunk.return_value = [
            Chunk(
                text="# Test Document",
                metadata={"source": "test.pdf", "chunk_index": 0, "char_count": 15},
            ),
        ]
        mock_embedding_service.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock_milvus_client.insert = AsyncMock()

        # Initialize ingester
        ingester = DocumentIngester(
            converter=mock_converter,
            chunker=mock_chunker,
            embedding_service=mock_embedding_service,
            milvus_client=mock_milvus_client,
        )

        # Ingest document
        result = await ingester.ingest(Path("test.pdf"))

        # Verify document_id is present and is a string
        assert "document_id" in result
        assert isinstance(result["document_id"], str)
        assert len(result["document_id"]) > 0

    @pytest.mark.asyncio
    async def test_ingest_returns_chunk_count(self, mocker):
        """Test that chunk_count is returned correctly."""
        # Create mock dependencies
        mock_converter = mocker.MagicMock()
        mock_chunker = mocker.MagicMock()
        mock_embedding_service = mocker.MagicMock()
        mock_milvus_client = mocker.MagicMock()

        # Setup mocks
        mock_converter.convert.return_value = {
            "markdown": "# Test Document\n\nThis is a test.",
            "metadata": {"source": "test.pdf", "format": "pdf", "char_count": 30, "word_count": 6},
        }
        mock_chunker.chunk.return_value = [
            Chunk(
                text="# Test Document",
                metadata={"source": "test.pdf", "chunk_index": 0, "char_count": 15},
            ),
            Chunk(
                text="This is a test.",
                metadata={"source": "test.pdf", "chunk_index": 1, "char_count": 15},
            ),
            Chunk(
                text="Another chunk.",
                metadata={"source": "test.pdf", "chunk_index": 2, "char_count": 14},
            ),
        ]
        mock_embedding_service.embed = AsyncMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        )
        mock_milvus_client.insert = AsyncMock()

        # Initialize ingester
        ingester = DocumentIngester(
            converter=mock_converter,
            chunker=mock_chunker,
            embedding_service=mock_embedding_service,
            milvus_client=mock_milvus_client,
        )

        # Ingest document
        result = await ingester.ingest(Path("test.pdf"))

        # Verify chunk_count is correct
        assert result["chunk_count"] == 3

    @pytest.mark.asyncio
    async def test_ingest_returns_status(self, mocker):
        """Test that status is returned correctly (completed/failed)."""
        # Create mock dependencies
        mock_converter = mocker.MagicMock()
        mock_chunker = mocker.MagicMock()
        mock_embedding_service = mocker.MagicMock()
        mock_milvus_client = mocker.MagicMock()

        # Setup mocks for successful ingestion
        mock_converter.convert.return_value = {
            "markdown": "# Test Document",
            "metadata": {"source": "test.pdf", "format": "pdf", "char_count": 15, "word_count": 2},
        }
        mock_chunker.chunk.return_value = [
            Chunk(
                text="# Test Document",
                metadata={"source": "test.pdf", "chunk_index": 0, "char_count": 15},
            ),
        ]
        mock_embedding_service.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock_milvus_client.insert = AsyncMock()

        # Initialize ingester
        ingester = DocumentIngester(
            converter=mock_converter,
            chunker=mock_chunker,
            embedding_service=mock_embedding_service,
            milvus_client=mock_milvus_client,
        )

        # Test successful ingestion
        result = await ingester.ingest(Path("test.pdf"))
        assert result["status"] == "completed"

        # Test failed ingestion
        mock_converter.convert.side_effect = Exception("Conversion failed")
        result = await ingester.ingest(Path("test.pdf"))
        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_ingest_merges_metadata(self, mocker):
        """Test that metadata is merged correctly."""
        # Create mock dependencies
        mock_converter = mocker.MagicMock()
        mock_chunker = mocker.MagicMock()
        mock_embedding_service = mocker.MagicMock()
        mock_milvus_client = mocker.MagicMock()

        # Setup mocks
        mock_converter.convert.return_value = {
            "markdown": "# Test Document",
            "metadata": {"source": "test.pdf", "format": "pdf", "char_count": 15, "word_count": 2},
        }
        mock_chunker.chunk.return_value = [
            Chunk(
                text="# Test Document",
                metadata={
                    "source": "test.pdf",
                    "chunk_index": 0,
                    "char_count": 15,
                    "custom_field": "custom_value",
                },
            ),
        ]
        mock_embedding_service.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock_milvus_client.insert = AsyncMock()

        # Initialize ingester
        ingester = DocumentIngester(
            converter=mock_converter,
            chunker=mock_chunker,
            embedding_service=mock_embedding_service,
            milvus_client=mock_milvus_client,
        )

        # Ingest document
        result = await ingester.ingest(Path("test.pdf"))

        # Verify metadata was merged
        call_args = mock_milvus_client.insert.call_args
        metadata = call_args[1]["metadatas"][0]
        assert metadata["source"] == "test.pdf"
        assert metadata["format"] == "pdf"
        assert metadata["char_count"] == 15
        assert metadata["word_count"] == 2
        assert metadata["chunk_index"] == 0
        assert metadata["custom_field"] == "custom_value"
        assert "document_id" in metadata
