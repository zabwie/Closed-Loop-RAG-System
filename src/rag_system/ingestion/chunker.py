"""Text chunker for splitting documents into overlapping chunks."""

from typing import List, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A chunk of text with metadata."""

    text: str
    metadata: Dict[str, Any]


class TextChunker:
    """Chunker for splitting text into overlapping chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """Initialize the TextChunker.

        Args:
            chunk_size: Maximum number of words per chunk.
            overlap: Number of words to overlap between chunks.
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, source: str) -> List[Chunk]:
        """Split text into overlapping chunks.

        Args:
            text: The text to chunk.
            source: The source identifier for metadata.

        Returns:
            List of Chunk objects with text and metadata.
        """
        chunks = []
        words = text.split()

        if not words:
            return chunks

        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i : i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk_idx = i // (self.chunk_size - self.overlap)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata={
                        "source": source,
                        "chunk_index": chunk_idx,
                        "char_count": len(chunk_text),
                    },
                )
            )

        return chunks
