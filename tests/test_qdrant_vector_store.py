"""
Unit tests for QdrantVectorStore.

Tests vector storage operations including indexing, searching,
filtering, and batch operations with Qdrant vector database.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.retrieval.vectordb.qdrant_vector_store import QdrantVectorStore
from src.models import ContentChunk, ContentType, RetrievalResult, SourceLocation
from src.config import StorageConfig, EmbeddingConfig


class TestQdrantVectorStore:
    """Test suite for QdrantVectorStore."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def storage_config(self, temp_storage_dir):
        """Create test storage configuration."""
        return StorageConfig(
            vector_db_path=temp_storage_dir,
            collection_name="test_collection"
        )
    
    @pytest.fixture
    def embedding_config(self):
        """Create test embedding configuration."""
        return EmbeddingConfig(
            embedding_dimension=384,
            normalize_embeddings=True
        )
    
    @pytest.fixture
    def vector_store(self, storage_config, embedding_config):
        """Create QdrantVectorStore instance."""
        return QdrantVectorStore(storage_config, embedding_config)
    
    @pytest.fixture
    def sample_chunks(self):
        """Create sample content chunks for testing."""
        return [
            ContentChunk(
                chunk_id="chunk_1",
                content="This is a test document about machine learning.",
                document_id="doc_1",
                chunk_index=0,
                content_type=ContentType.TEXT,
                source_location=SourceLocation(
                    file_path="/path/to/doc1.pdf",
                    page_number=1,
                    section="introduction"
                ),
                metadata={"topic": "AI", "language": "en"}
            ),
            ContentChunk(
                chunk_id="chunk_2",
                content="Deep learning is a subset of machine learning.",
                document_id="doc_1",
                chunk_index=1,
                content_type=ContentType.TEXT,
                source_location=SourceLocation(
                    file_path="/path/to/doc1.pdf",
                    page_number=2,
                    section="methods"
                ),
                metadata={"topic": "AI", "language": "en"}
            ),
            ContentChunk(
                chunk_id="chunk_3",
                content="Natural language processing uses neural networks.",
                document_id="doc_2",
                chunk_index=0,
                content_type=ContentType.TEXT,
                source_location=SourceLocation(
                    file_path="/path/to/doc2.pdf",
                    page_number=1,
                    section="overview"
                ),
                metadata={"topic": "NLP", "language": "en"}
            )
        ]
    
    def test_initialization(self, storage_config, embedding_config):
        """Test QdrantVectorStore initialization."""
        store = QdrantVectorStore(storage_config, embedding_config)
        
        assert store.storage_config == storage_config
        assert store.embedding_config == embedding_config
        assert store.embedding_dimension == 384
        assert store.client is None
        assert not store.is_initialized
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_initialize_success(self, mock_qdrant_client, vector_store):
        """Test successful initialization."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.create_collection.return_value = True
        
        vector_store.initialize()
        
        assert vector_store.is_initialized
        assert vector_store.client == mock_client
        mock_client.create_collection.assert_called_once()
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_initialize_collection_exists(self, mock_qdrant_client, vector_store):
        """Test initialization when collection already exists."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        
        # Mock existing collection
        existing_collection = Mock()
        existing_collection.name = "test_collection"
        mock_client.get_collections.return_value = Mock(collections=[existing_collection])
        
        vector_store.initialize()
        
        assert vector_store.is_initialized
        assert vector_store.client == mock_client
        # Should not create collection if it exists
        mock_client.create_collection.assert_not_called()
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_initialize_failure(self, mock_qdrant_client, vector_store):
        """Test initialization failure."""
        mock_qdrant_client.side_effect = Exception("Connection failed")
        
        with pytest.raises(RuntimeError, match="Failed to initialize Qdrant"):
            vector_store.initialize()
        
        assert not vector_store.is_initialized
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_index_chunks_success(self, mock_qdrant_client, vector_store, sample_chunks):
        """Test successful chunk indexing."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.create_collection.return_value = True
        mock_client.upsert.return_value = True
        
        # Mock embedding generation
        mock_embeddings = [np.random.rand(384).astype(np.float32) for _ in sample_chunks]
        
        with patch.object(vector_store, '_generate_embeddings') as mock_generate:
            mock_generate.return_value = mock_embeddings
            
            vector_store.initialize()
            result = vector_store.index_chunks(sample_chunks)
            
            assert result == len(sample_chunks)
            mock_client.upsert.assert_called_once()
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_index_chunks_not_initialized(self, mock_qdrant_client, vector_store, sample_chunks):
        """Test indexing chunks without initialization."""
        with pytest.raises(RuntimeError, match="Vector store not initialized"):
            vector_store.index_chunks(sample_chunks)
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_search_similar_success(self, mock_qdrant_client, vector_store):
        """Test successful similarity search."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.create_collection.return_value = True
        
        # Mock search results
        mock_point1 = Mock()
        mock_point1.id = "chunk_1"
        mock_point1.score = 0.95
        mock_point1.payload = {
            "content": "Test content 1",
            "document_id": "doc_1",
            "content_type": "text",
            "metadata": {"topic": "AI"}
        }
        
        mock_point2 = Mock()
        mock_point2.id = "chunk_2"
        mock_point2.score = 0.87
        mock_point2.payload = {
            "content": "Test content 2",
            "document_id": "doc_2",
            "content_type": "text",
            "metadata": {"topic": "ML"}
        }
        
        mock_client.search.return_value = [mock_point1, mock_point2]
        
        # Mock embedding generation
        query_embedding = np.random.rand(384).astype(np.float32)
        
        with patch.object(vector_store, '_generate_query_embedding') as mock_generate:
            mock_generate.return_value = query_embedding
            
            vector_store.initialize()
            results = vector_store.search_similar(
                query="machine learning",
                top_k=2,
                content_type=ContentType.TEXT
            )
            
            assert len(results) == 2
            assert isinstance(results[0], RetrievalResult)
            assert results[0].chunk_id == "chunk_1"
            assert results[0].similarity_score == 0.95
            assert results[1].chunk_id == "chunk_2"
            assert results[1].similarity_score == 0.87
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_search_similar_with_filter(self, mock_qdrant_client, vector_store):
        """Test similarity search with metadata filter."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.create_collection.return_value = True
        
        # Mock search results
        mock_point = Mock()
        mock_point.id = "chunk_1"
        mock_point.score = 0.92
        mock_point.payload = {
            "content": "AI content",
            "document_id": "doc_1",
            "content_type": "text",
            "metadata": {"topic": "AI", "language": "en"}
        }
        
        mock_client.search.return_value = [mock_point]
        
        # Mock embedding generation
        query_embedding = np.random.rand(384).astype(np.float32)
        
        with patch.object(vector_store, '_generate_query_embedding') as mock_generate:
            mock_generate.return_value = query_embedding
            
            vector_store.initialize()
            results = vector_store.search_similar(
                query="artificial intelligence",
                top_k=5,
                content_type=ContentType.TEXT,
                metadata_filter={"topic": "AI"}
            )
            
            assert len(results) == 1
            assert results[0].chunk_id == "chunk_1"
            # Verify filter was applied
            mock_client.search.assert_called_once()
            call_args = mock_client.search.call_args
            assert 'query_filter' in call_args[1]
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_search_similar_not_initialized(self, mock_qdrant_client, vector_store):
        """Test similarity search without initialization."""
        with pytest.raises(RuntimeError, match="Vector store not initialized"):
            vector_store.search_similar("test query")
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_delete_document(self, mock_qdrant_client, vector_store):
        """Test document deletion."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.create_collection.return_value = True
        mock_client.delete.return_value = True
        
        vector_store.initialize()
        result = vector_store.delete_document("doc_1")
        
        assert result is True
        mock_client.delete.assert_called_once()
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_delete_document_not_initialized(self, mock_qdrant_client, vector_store):
        """Test document deletion without initialization."""
        with pytest.raises(RuntimeError, match="Vector store not initialized"):
            vector_store.delete_document("doc_1")
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_get_collection_stats(self, mock_qdrant_client, vector_store):
        """Test getting collection statistics."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.create_collection.return_value = True
        
        # Mock collection info
        mock_collection_info = Mock()
        mock_collection_info.points_count = 100
        mock_collection_info.config.params.vectors.size = 384
        mock_client.get_collection.return_value = mock_collection_info
        
        vector_store.initialize()
        stats = vector_store.get_collection_stats()
        
        assert stats['total_chunks'] == 100
        assert stats['vector_dimension'] == 384
        assert stats['collection_name'] == "test_collection"
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_get_collection_stats_not_initialized(self, mock_qdrant_client, vector_store):
        """Test getting collection stats without initialization."""
        with pytest.raises(RuntimeError, match="Vector store not initialized"):
            vector_store.get_collection_stats()
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_clear_collection(self, mock_qdrant_client, vector_store):
        """Test clearing collection."""
        mock_client = Mock()
        mock_qdrant_client.return_value = mock_client
        mock_client.get_collections.return_value = Mock(collections=[])
        mock_client.create_collection.return_value = True
        mock_client.delete.return_value = True
        mock_client.create_collection.return_value = True
        
        vector_store.initialize()
        result = vector_store.clear_collection()
        
        assert result is True
        # Should delete and recreate collection
        assert mock_client.delete.call_count >= 1
        assert mock_client.create_collection.call_count >= 2
    
    @patch('src.retrieval.vectordb.qdrant_vector_store.QdrantClient')
    def test_clear_collection_not_initialized(self, mock_qdrant_client, vector_store):
        """Test clearing collection without initialization."""
        with pytest.raises(RuntimeError, match="Vector store not initialized"):
            vector_store.clear_collection()
    
    def test_generate_embeddings_text(self, vector_store, sample_chunks):
        """Test embedding generation for text chunks."""
        # Mock embedding generator
        mock_embedding = np.random.rand(384).astype(np.float32)
        
        with patch.object(vector_store, '_get_embedding_generator') as mock_generator_class:
            mock_generator = Mock()
            mock_generator.encode_text.return_value = mock_embedding
            mock_generator_class.return_value = mock_generator
            
            embeddings = vector_store._generate_embeddings(sample_chunks[:2])
            
            assert len(embeddings) == 2
            assert all(isinstance(emb, np.ndarray) for emb in embeddings)
            assert all(emb.shape == (384,) for emb in embeddings)
            assert mock_generator.encode_text.call_count == 2
    
    def test_generate_embeddings_image(self, vector_store):
        """Test embedding generation for image chunks."""
        image_chunk = ContentChunk(
            chunk_id="img_chunk_1",
            content="/path/to/image.jpg",
            document_id="img_doc_1",
            chunk_index=0,
            content_type=ContentType.IMAGE,
            source_location=SourceLocation(file_path="/path/to/image.jpg"),
            metadata={"format": "jpg"}
        )
        
        # Mock embedding generator
        mock_embedding = np.random.rand(384).astype(np.float32)
        
        with patch.object(vector_store, '_get_embedding_generator') as mock_generator_class:
            mock_generator = Mock()
            mock_generator.encode_image.return_value = mock_embedding
            mock_generator_class.return_value = mock_generator
            
            with patch('PIL.Image.open') as mock_image_open:
                mock_img = Mock()
                mock_image_open.return_value = mock_img
                
                embeddings = vector_store._generate_embeddings([image_chunk])
                
                assert len(embeddings) == 1
                assert isinstance(embeddings[0], np.ndarray)
                assert embeddings[0].shape == (384,)
                mock_generator.encode_image.assert_called_once()
    
    def test_chunk_to_point_conversion(self, vector_store, sample_chunks):
        """Test conversion of content chunks to Qdrant points."""
        mock_embedding = np.random.rand(384).astype(np.float32)
        chunk = sample_chunks[0]
        
        point = vector_store._chunk_to_point(chunk, mock_embedding)
        
        assert point.id == chunk.chunk_id
        assert np.array_equal(point.vector, mock_embedding)
        assert point.payload['content'] == chunk.content
        assert point.payload['document_id'] == chunk.document_id
        assert point.payload['content_type'] == chunk.content_type.value
        assert point.payload['metadata'] == chunk.metadata
    
    def test_point_to_retrieval_result_conversion(self, vector_store):
        """Test conversion of Qdrant points to retrieval results."""
        mock_point = Mock()
        mock_point.id = "chunk_1"
        mock_point.score = 0.95
        mock_point.payload = {
            "content": "Test content",
            "document_id": "doc_1",
            "chunk_index": 0,
            "content_type": "text",
            "source_location": {
                "file_path": "/path/to/doc.pdf",
                "page_number": 1,
                "section": "intro"
            },
            "metadata": {"topic": "AI"}
        }
        
        result = vector_store._point_to_retrieval_result(mock_point)
        
        assert isinstance(result, RetrievalResult)
        assert result.chunk_id == "chunk_1"
        assert result.similarity_score == 0.95
        assert result.content == "Test content"
        assert result.document_id == "doc_1"
        assert result.content_type == ContentType.TEXT
        assert result.metadata == {"topic": "AI"}
    
    def test_build_metadata_filter(self, vector_store):
        """Test building metadata filter for Qdrant."""
        metadata_filter = {"topic": "AI", "language": "en"}
        
        with patch('src.retrieval.vectordb.qdrant_vector_store.models') as mock_models:
            mock_filter = Mock()
            mock_models.Filter.return_value = mock_filter
            
            result = vector_store._build_metadata_filter(metadata_filter)
            
            assert result == mock_filter
            mock_models.Filter.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
