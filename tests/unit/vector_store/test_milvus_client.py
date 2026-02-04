"""Unit tests for MilvusVectorStore."""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from rag_system.vector_store.milvus_client import MilvusVectorStore


class TestMilvusVectorStoreInitialization:
    """Tests for MilvusVectorStore initialization."""

    def test_milvus_vector_store_initialization(self):
        """Test store initialization with default parameters."""
        store = MilvusVectorStore(host="localhost", port=19530, collection_name="documents")
        assert store.host == "localhost"
        assert store.port == 19530
        assert store.collection_name == "documents"
        assert store.dimension == 768
        assert store.collection is None

    def test_milvus_vector_store_initialization_with_custom_dimension(self):
        """Test store initialization with custom dimension."""
        store = MilvusVectorStore(
            host="localhost", port=19530, collection_name="documents", dimension=1536
        )
        assert store.dimension == 1536


class TestMilvusVectorStoreConnect:
    """Tests for MilvusVectorStore connect method."""

    def test_connect(self, mocker):
        """Test connection to Milvus."""
        mock_connections = mocker.patch("rag_system.vector_store.milvus_client.connections")
        store = MilvusVectorStore(host="localhost", port=19530, collection_name="documents")
        store.connect()
        mock_connections.connect.assert_called_once_with(host="localhost", port=19530)


class TestMilvusVectorStoreCreateCollection:
    """Tests for MilvusVectorStore create_collection method."""

    def test_create_collection_new(self, mocker):
        """Test collection creation when it doesn't exist."""
        mock_utility = mocker.patch("rag_system.vector_store.milvus_client.utility")
        mock_utility.has_collection.return_value = False

        mock_collection = MagicMock()
        mock_collection_class = mocker.patch("rag_system.vector_store.milvus_client.Collection")
        mock_collection_class.return_value = mock_collection

        store = MilvusVectorStore(host="localhost", port=19530, collection_name="documents")
        store.create_collection()

        mock_utility.has_collection.assert_called_once_with("documents")
        mock_collection_class.assert_called_once()
        mock_collection.create_index.assert_called_once_with(
            "vector", {"index_type": "HNSW", "metric_type": "IP"}
        )
        assert store.collection == mock_collection

    def test_create_collection_existing(self, mocker):
        """Test collection creation when it already exists."""
        mock_utility = mocker.patch("rag_system.vector_store.milvus_client.utility")
        mock_utility.has_collection.return_value = True

        mock_collection = MagicMock()
        mock_collection_class = mocker.patch("rag_system.vector_store.milvus_client.Collection")
        mock_collection_class.return_value = mock_collection

        store = MilvusVectorStore(host="localhost", port=19530, collection_name="documents")
        store.create_collection()

        mock_utility.has_collection.assert_called_once_with("documents")
        mock_collection_class.assert_called_once_with("documents")
        mock_collection.create_index.assert_not_called()
        assert store.collection == mock_collection


class TestMilvusVectorStoreInsert:
    """Tests for MilvusVectorStore insert method."""

    @pytest.mark.asyncio
    async def test_insert(self, mocker):
        """Test inserting embeddings with metadata."""
        mock_uuid = mocker.patch("rag_system.vector_store.milvus_client.uuid")
        mock_uuid.uuid4.return_value = "test-uuid-1"

        mock_collection = MagicMock()
        store = MilvusVectorStore(host="localhost", port=19530, collection_name="documents")
        store.collection = mock_collection

        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        texts = ["text1", "text2"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}]

        await store.insert(embeddings, texts, metadatas)

        mock_collection.insert.assert_called_once()
        call_args = mock_collection.insert.call_args[0][0]
        assert len(call_args) == 4  # ids, texts, embeddings, metadatas
        mock_collection.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_insert_generates_uuids(self, mocker):
        """Test UUID generation for IDs."""
        mock_uuid = mocker.patch("rag_system.vector_store.milvus_client.uuid")
        mock_uuid.uuid4.side_effect = ["uuid-1", "uuid-2", "uuid-3"]

        mock_collection = MagicMock()
        store = MilvusVectorStore(host="localhost", port=19530, collection_name="documents")
        store.collection = mock_collection

        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        texts = ["text1", "text2", "text3"]
        metadatas = [{"source": "doc1"}, {"source": "doc2"}, {"source": "doc3"}]

        await store.insert(embeddings, texts, metadatas)

        call_args = mock_collection.insert.call_args[0][0]
        ids = call_args[0]
        assert ids == ["uuid-1", "uuid-2", "uuid-3"]


class TestMilvusVectorStoreSearch:
    """Tests for MilvusVectorStore search method."""

    @pytest.mark.asyncio
    async def test_search(self, mocker):
        """Test searching by similarity."""
        # Mock search results
        mock_entity = MagicMock()
        mock_entity.get.side_effect = lambda key: {
            "text": "result text",
            "metadata": {"source": "doc1"},
        }.get(key)

        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.entity = mock_entity

        # Create a proper mock structure for Milvus search results
        # Milvus returns: [[hit1, hit2, ...]] where each hit is iterable
        mock_hit = mocker.Mock()
        mock_hit.__iter__ = Mock(return_value=iter([mock_result]))

        mock_search_results = [[mock_hit]]

        mock_collection = MagicMock()
        mock_collection.search.return_value = mock_search_results
        mock_collection.load = MagicMock()

        store = MilvusVectorStore(host="localhost", port=19530, collection_name="documents")
        store.collection = mock_collection

        query_vector = [0.1, 0.2, 0.3]
        results = await store.search(query_vector, top_k=5)

        mock_collection.load.assert_called_once()
        mock_collection.search.assert_called_once_with(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=5,
        )
        assert len(results) == 1
        assert results[0]["text"] == "result text"
        assert results[0]["metadata"] == {"source": "doc1"}
        assert results[0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_search_returns_correct_results(self, mocker):
        """Test search results format."""
        # Mock search results with multiple hits
        mock_entity1 = MagicMock()
        mock_entity1.get.side_effect = lambda key: {
            "text": "result1",
            "metadata": {"source": "doc1"},
        }.get(key)

        mock_entity2 = MagicMock()
        mock_entity2.get.side_effect = lambda key: {
            "text": "result2",
            "metadata": {"source": "doc2"},
        }.get(key)

        mock_result1 = MagicMock()
        mock_result1.score = 0.95
        mock_result1.entity = mock_entity1

        mock_result2 = MagicMock()
        mock_result2.score = 0.85
        mock_result2.entity = mock_entity2

        # Create a proper mock structure for Milvus search results
        mock_hit = mocker.Mock()
        mock_hit.__iter__ = Mock(return_value=iter([mock_result1, mock_result2]))

        mock_search_results = [[mock_hit]]

        mock_collection = MagicMock()
        mock_collection.search.return_value = mock_search_results
        mock_collection.load = MagicMock()

        store = MilvusVectorStore(host="localhost", port=19530, collection_name="documents")
        store.collection = mock_collection

        query_vector = [0.1, 0.2, 0.3]
        results = await store.search(query_vector, top_k=5)

        assert len(results) == 2
        assert results[0]["text"] == "result1"
        assert results[0]["metadata"] == {"source": "doc1"}
        assert results[0]["score"] == 0.95
        assert results[1]["text"] == "result2"
        assert results[1]["metadata"] == {"source": "doc2"}
        assert results[1]["score"] == 0.85

    @pytest.mark.asyncio
    async def test_search_with_top_k(self, mocker):
        """Test top_k parameter."""
        mock_entity = MagicMock()
        mock_entity.get.side_effect = lambda key: {
            "text": "result",
            "metadata": {"source": "doc1"},
        }.get(key)

        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.entity = mock_entity

        mock_hit = MagicMock()
        mock_hit.__iter__ = Mock(return_value=iter([mock_result]))

        mock_search_results = [mock_hit]

        mock_collection = MagicMock()
        mock_collection.search.return_value = mock_search_results
        mock_collection.load = MagicMock()

        store = MilvusVectorStore(host="localhost", port=19530, collection_name="documents")
        store.collection = mock_collection

        query_vector = [0.1, 0.2, 0.3]
        results = await store.search(query_vector, top_k=10)

        mock_collection.search.assert_called_once_with(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"ef": 64}},
            limit=10,
        )
