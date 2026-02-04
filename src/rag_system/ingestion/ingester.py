"""DocumentIngester for processing documents and storing in Milvus."""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict

from .chunker import Chunk, TextChunker
from .embeddings import EmbeddingService
from .markitdown_converter import MarkItDownConverter

logger = logging.getLogger(__name__)


class DocumentIngester:
    """Ingester for processing documents and storing in Milvus."""

    def __init__(
        self,
        converter: MarkItDownConverter,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        milvus_client: Any,
    ):
        """Initialize the DocumentIngester.

        Args:
            converter: MarkItDownConverter instance.
            chunker: TextChunker instance.
            embedding_service: EmbeddingService instance.
            milvus_client: MilvusVectorStore instance.
        """
        self.converter = converter
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.milvus_client = milvus_client

    async def ingest(self, file_path: Path) -> Dict[str, Any]:
        """Process document and store in Milvus.

        Args:
            file_path: Path to the document file.

        Returns:
            Dictionary with document_id, status, chunk_count, source.

        Raises:
            Exception: If ingestion fails.
        """
        document_id = str(uuid.uuid4())

        try:
            # Convert to markdown
            converted = self.converter.convert(file_path)
            markdown = converted["markdown"]
            metadata = converted["metadata"]

            # Chunk
            chunks = self.chunker.chunk(markdown, metadata["source"])

            # Embed
            texts = [chunk.text for chunk in chunks]
            embeddings = await self.embedding_service.embed(texts)

            # Store in Milvus
            await self.milvus_client.insert(
                embeddings=embeddings,
                texts=texts,
                metadatas=[
                    {
                        **chunk.metadata,
                        **metadata,
                        "document_id": document_id,
                    }
                    for chunk in chunks
                ],
            )

            return {
                "document_id": document_id,
                "status": "completed",
                "chunk_count": len(chunks),
                "source": metadata["source"],
            }
        except Exception as e:
            logger.error(f"Ingestion failed for {file_path}: {e}")
            return {
                "document_id": document_id,
                "status": "failed",
                "error": str(e),
            }
