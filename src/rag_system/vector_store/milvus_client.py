"""Milvus vector store implementation."""

from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility
from typing import List, Dict, Any
import logging
import uuid

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Vector store using Milvus for similarity search."""

    def __init__(self, host: str, port: int, collection_name: str, dimension: int = 768):
        """Initialize the MilvusVectorStore.

        Args:
            host: Milvus host address.
            port: Milvus port.
            collection_name: Name of the collection.
            dimension: Dimension of the embedding vectors.
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.collection = None

    def connect(self):
        """Connect to Milvus."""
        connections.connect(host=self.host, port=self.port)

    def create_collection(self):
        """Create collection with HNSW index."""
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=36),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="metadata", dtype=DataType.JSON),
            ]
            schema = CollectionSchema(fields, f"RAG documents {self.collection_name}")
            self.collection = Collection(self.collection_name, schema)
            self.collection.create_index("vector", {"index_type": "HNSW", "metric_type": "IP"})
        else:
            self.collection = Collection(self.collection_name)

    async def insert(
        self, embeddings: List[List[float]], texts: List[str], metadatas: List[Dict[str, Any]]
    ):
        """Insert documents into Milvus.

        Args:
            embeddings: List of embedding vectors.
            texts: List of text chunks.
            metadatas: List of metadata dictionaries.
        """
        data = [[str(uuid.uuid4()) for _ in embeddings], texts, embeddings, metadatas]
        self.collection.insert(data)
        self.collection.flush()

    async def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of search results with text, metadata, and score.
        """
        self.collection.load()
        results = self.collection.search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=top_k,
        )
        # results[0] is a list of hits, each hit is iterable and contains results
        search_results = []
        for hit in results[0]:
            for result in hit:
                search_results.append(
                    {
                        "text": result.entity.get("text"),
                        "metadata": result.entity.get("metadata"),
                        "score": result.score,
                    }
                )
        return search_results
