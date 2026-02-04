"""RAG query engine for retrieval-augmented generation."""

import logging
from typing import Dict, Any

from .ollama_client import OllamaClient
from ..vector_store.milvus_client import MilvusVectorStore
from ..ingestion.embeddings import EmbeddingService


logger = logging.getLogger(__name__)


class RAGQueryEngine:
    """RAG query engine for retrieval-augmented generation."""

    def __init__(
        self, ollama: OllamaClient, vector_store: MilvusVectorStore, embeddings: EmbeddingService
    ):
        """Initialize the RAGQueryEngine.

        Args:
            ollama: OllamaClient instance.
            vector_store: MilvusVectorStore instance.
            embeddings: EmbeddingService instance.
        """
        self.ollama = ollama
        self.vector_store = vector_store
        self.embeddings = embeddings

    async def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Execute RAG query.

        Args:
            question: The user's question.
            top_k: Number of documents to retrieve.

        Returns:
            Dictionary with answer, sources, retrieved_count.
        """
        # Embed query
        query_vector = await self.embeddings.embed_single(question)

        # Retrieve relevant chunks
        results = await self.vector_store.search(query_vector, top_k=top_k)

        if not results:
            return {"answer": "No relevant documents found.", "sources": [], "retrieved_count": 0}

        # Assemble context
        context = "\n\n".join([r["text"] for r in results])

        # Generate answer
        answer = await self.ollama.chat(question, context)

        return {
            "answer": answer,
            "sources": [
                {"text": r["text"], "score": r["score"], "metadata": r["metadata"]} for r in results
            ],
            "retrieved_count": len(results),
        }
