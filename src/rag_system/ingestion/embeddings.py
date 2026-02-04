"""Embedding service for generating embeddings using Ollama."""

import logging
from typing import List

from ollama import Client

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using Ollama."""

    def __init__(self, ollama_url: str, model: str = "nomic-embed-text"):
        """Initialize the EmbeddingService.

        Args:
            ollama_url: URL of the Ollama service.
            model: Name of the embedding model to use.
        """
        self.client = Client(host=ollama_url)
        self.model = model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors (each is a list of floats).

        Raises:
            Exception: If embedding generation fails.
        """
        try:
            response = await self.client.embed(input=texts, model=self.model)
            return response.embeddings
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    async def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector (list of floats).

        Raises:
            Exception: If embedding generation fails.
        """
        embeddings = await self.embed([text])
        return embeddings[0]
