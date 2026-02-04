"""Unit tests for TextChunker."""

import sys
from unittest.mock import MagicMock

# Mock MarkItDown before importing to avoid RuntimeWarning during test collection
sys.modules["markitdown"] = MagicMock()
sys.modules["markitdown._markitdown"] = MagicMock()

import pytest
from rag_system.ingestion.chunker import TextChunker, Chunk


class TestTextChunkerInitialization:
    """Test TextChunker initialization."""

    def test_chunker_initialization_defaults(self):
        """Test chunker initialization with default parameters."""
        chunker = TextChunker()
        assert chunker.chunk_size == 512
        assert chunker.overlap == 50

    def test_chunker_initialization_custom(self):
        """Test chunker initialization with custom parameters."""
        chunker = TextChunker(chunk_size=256, overlap=25)
        assert chunker.chunk_size == 256
        assert chunker.overlap == 25


class TestTextChunkerChunking:
    """Test TextChunker chunking functionality."""

    def test_chunker_splits_text(self):
        """Test basic text splitting."""
        chunker = TextChunker(chunk_size=10, overlap=2)
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12"
        chunks = chunker.chunk(text, source="test.txt")
        assert len(chunks) == 2
        assert chunks[0].text == "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        assert chunks[1].text == "word9 word10 word11 word12"

    def test_chunker_creates_overlapping_chunks(self):
        """Test overlap between chunks."""
        chunker = TextChunker(chunk_size=10, overlap=2)
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12"
        chunks = chunker.chunk(text, source="test.txt")
        # First chunk ends with "word9 word10"
        # Second chunk starts with "word9 word10" (overlap)
        assert chunks[0].text.endswith("word9 word10")
        assert chunks[1].text.startswith("word9 word10")

    def test_chunker_adds_metadata(self):
        """Test metadata is added to chunks."""
        chunker = TextChunker(chunk_size=10, overlap=2)
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = chunker.chunk(text, source="test.txt")
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[0].metadata["chunk_index"] == 0
        assert "char_count" in chunks[0].metadata

    def test_chunker_empty_text(self):
        """Test handling of empty text."""
        chunker = TextChunker(chunk_size=10, overlap=2)
        chunks = chunker.chunk("", source="test.txt")
        assert len(chunks) == 0

    def test_chunker_whitespace_only(self):
        """Test handling of whitespace-only text."""
        chunker = TextChunker(chunk_size=10, overlap=2)
        chunks = chunker.chunk("   \n\t  ", source="test.txt")
        assert len(chunks) == 0

    def test_chunker_single_chunk(self):
        """Test when text fits in single chunk."""
        chunker = TextChunker(chunk_size=10, overlap=2)
        text = "word1 word2 word3 word4 word5"
        chunks = chunker.chunk(text, source="test.txt")
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].metadata["chunk_index"] == 0

    def test_chunker_custom_chunk_size(self):
        """Test custom chunk size."""
        chunker = TextChunker(chunk_size=5, overlap=1)
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = chunker.chunk(text, source="test.txt")
        assert len(chunks) == 2
        assert len(chunks[0].text.split()) == 5
        assert len(chunks[1].text.split()) == 4

    def test_chunker_custom_overlap(self):
        """Test custom overlap."""
        chunker = TextChunker(chunk_size=10, overlap=5)
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        chunks = chunker.chunk(text, source="test.txt")
        assert len(chunks) == 2
        # First chunk: word1-word10 (10 words)
        # Second chunk: word6-word10 (5 words, overlap of 5 words: word6-word10)
        assert chunks[0].text.endswith("word6 word7 word8 word9 word10")
        assert chunks[1].text.startswith("word6 word7 word8 word9 word10")

    def test_chunker_preserves_word_boundaries(self):
        """Test word boundaries are preserved."""
        chunker = TextChunker(chunk_size=10, overlap=2)
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = chunker.chunk(text, source="test.txt")
        # All chunks should have complete words (no partial words)
        for chunk in chunks:
            words = chunk.text.split()
            # Check that words are alphanumeric (no special characters that would indicate partial words)
            assert all(word.isalnum() for word in words)

    def test_chunker_multiple_chunks_metadata(self):
        """Test metadata for multiple chunks."""
        chunker = TextChunker(chunk_size=5, overlap=1)
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = chunker.chunk(text, source="test.txt")
        assert len(chunks) == 2
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[1].metadata["chunk_index"] == 1
        assert chunks[0].metadata["source"] == "test.txt"
        assert chunks[1].metadata["source"] == "test.txt"

    def test_chunker_char_count_metadata(self):
        """Test char_count in metadata."""
        chunker = TextChunker(chunk_size=5, overlap=1)
        text = "word1 word2 word3 word4"
        chunks = chunker.chunk(text, source="test.txt")
        assert len(chunks) == 1
        assert chunks[0].metadata["char_count"] == len(text)

    def test_chunker_large_text(self):
        """Test chunking of large text."""
        chunker = TextChunker(chunk_size=100, overlap=10)
        # Create text with 500 words
        words = [f"word{i}" for i in range(1, 501)]
        text = " ".join(words)
        chunks = chunker.chunk(text, source="large.txt")
        # Expected: chunks at positions 0, 90, 180, 270, 360, 450 = 6 chunks
        assert len(chunks) == 6
        assert all(chunk.metadata["source"] == "large.txt" for chunk in chunks)
