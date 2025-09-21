"""
Unit tests for MultimodalRetrievalSystem.

Tests the complete retrieval pipeline including document processing,
embedding generation, vector storage, and response generation.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.retrieval.retrieval_system import MultimodalRetrievalSystem
from src.models import (
    ContentChunk, ContentType, RetrievalResult, 
    DocumentContent, ProcessingResult, GroundedResponse, Citation
)
from src.config import SystemConfig, ProcessingConfig, EmbeddingConfig, StorageConfig


class TestMultimodalRetrievalSystem:
    """Test suite for MultimodalRetrievalSystem."""
    
    @pytest.fixture
    def temp_storage_dir(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def system_config(self, temp_storage_dir):
        """Create test system configuration."""
        processing_config = ProcessingConfig(
            max_file_size_mb=10,
            supported_formats=['pdf', 'docx', 'png', 'jpg', 'mp3', 'wav']
        )
        
        embedding_config = EmbeddingConfig(
            text_model_name="sentence-transformers/all-MiniLM-L6-v2",
            image_model_name="openai/clip-vit-base-patch32",
            embedding_dimension=512,
            device="cpu"
        )
        
        storage_config = StorageConfig(
            vector_db_path=temp_storage_dir,
            collection_name="test_collection"
        )
        
        return SystemConfig(
            processing=processing_config,
            embedding=embedding_config,
            storage=storage_config
        )
    
    @pytest.fixture
    def retrieval_system(self, system_config):
        """Create MultimodalRetrievalSystem instance."""
        return MultimodalRetrievalSystem(system_config)
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            {
                "file_path": "/path/to/doc1.pdf",
                "content": "This is a document about machine learning and artificial intelligence.",
                "metadata": {"author": "John Doe", "topic": "AI"}
            },
            {
                "file_path": "/path/to/doc2.pdf", 
                "content": "Deep learning is a subset of machine learning that uses neural networks.",
                "metadata": {"author": "Jane Smith", "topic": "ML"}
            }
        ]
    
    def test_initialization(self, system_config):
        """Test MultimodalRetrievalSystem initialization."""
        system = MultimodalRetrievalSystem(system_config)
        
        assert system.config == system_config
        assert system.document_router is not None
        assert system.embedding_generator is not None
        assert system.vector_store is not None
        assert not system.is_initialized
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_initialize_success(self, mock_init_components, retrieval_system):
        """Test successful system initialization."""
        mock_init_components.return_value = True
        
        retrieval_system.initialize()
        
        assert retrieval_system.is_initialized
        mock_init_components.assert_called_once()
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_initialize_failure(self, mock_init_components, retrieval_system):
        """Test system initialization failure."""
        mock_init_components.side_effect = Exception("Initialization failed")
        
        with pytest.raises(RuntimeError, match="Failed to initialize retrieval system"):
            retrieval_system.initialize()
        
        assert not retrieval_system.is_initialized
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_index_documents_success(self, mock_init_components, retrieval_system, sample_documents):
        """Test successful document indexing."""
        mock_init_components.return_value = True
        
        # Mock document processing
        mock_chunks = [
            ContentChunk(
                chunk_id="chunk_1",
                content="This is a document about machine learning and artificial intelligence.",
                document_id="doc_1",
                chunk_index=0,
                content_type=ContentType.TEXT,
                metadata={"author": "John Doe", "topic": "AI"}
            )
        ]
        
        with patch.object(retrieval_system.document_router, 'process_document') as mock_process:
            mock_process.return_value = ProcessingResult(
                document_id="doc_1",
                content_type=ContentType.TEXT,
                chunks=mock_chunks,
                metadata={"author": "John Doe", "topic": "AI"}
            )
            
            with patch.object(retrieval_system.vector_store, 'index_chunks') as mock_index:
                mock_index.return_value = 1
                
                retrieval_system.initialize()
                result = retrieval_system.index_documents([sample_documents[0]])
                
                assert result['successful'] == 1
                assert result['failed'] == 0
                assert result['total_chunks'] == 1
                mock_process.assert_called_once()
                mock_index.assert_called_once()
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_index_documents_not_initialized(self, mock_init_components, retrieval_system, sample_documents):
        """Test document indexing without initialization."""
        with pytest.raises(RuntimeError, match="Retrieval system not initialized"):
            retrieval_system.index_documents(sample_documents)
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_search_success(self, mock_init_components, retrieval_system):
        """Test successful search."""
        mock_init_components.return_value = True
        
        # Mock search results
        mock_results = [
            RetrievalResult(
                chunk_id="chunk_1",
                content="This is about machine learning",
                document_id="doc_1",
                similarity_score=0.95,
                content_type=ContentType.TEXT,
                metadata={"topic": "AI"}
            )
        ]
        
        with patch.object(retrieval_system.vector_store, 'search_similar') as mock_search:
            mock_search.return_value = mock_results
            
            retrieval_system.initialize()
            results = retrieval_system.search(
                query="machine learning",
                top_k=5,
                content_type=ContentType.TEXT
            )
            
            assert len(results) == 1
            assert results[0].chunk_id == "chunk_1"
            assert results[0].similarity_score == 0.95
            mock_search.assert_called_once()
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_search_with_filter(self, mock_init_components, retrieval_system):
        """Test search with metadata filter."""
        mock_init_components.return_value = True
        
        mock_results = [
            RetrievalResult(
                chunk_id="chunk_1",
                content="AI content",
                document_id="doc_1",
                similarity_score=0.92,
                content_type=ContentType.TEXT,
                metadata={"topic": "AI"}
            )
        ]
        
        with patch.object(retrieval_system.vector_store, 'search_similar') as mock_search:
            mock_search.return_value = mock_results
            
            retrieval_system.initialize()
            results = retrieval_system.search(
                query="artificial intelligence",
                top_k=5,
                content_type=ContentType.TEXT,
                metadata_filter={"topic": "AI"}
            )
            
            assert len(results) == 1
            # Verify filter was passed to vector store
            mock_search.assert_called_once()
            call_args = mock_search.call_args
            assert call_args[1]['metadata_filter'] == {"topic": "AI"}
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_search_not_initialized(self, mock_init_components, retrieval_system):
        """Test search without initialization."""
        with pytest.raises(RuntimeError, match="Retrieval system not initialized"):
            retrieval_system.search("test query")
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_generate_grounded_response(self, mock_init_components, retrieval_system):
        """Test grounded response generation."""
        mock_init_components.return_value = True
        
        # Mock search results
        mock_results = [
            RetrievalResult(
                chunk_id="chunk_1",
                content="Machine learning is a subset of artificial intelligence.",
                document_id="doc_1",
                similarity_score=0.95,
                content_type=ContentType.TEXT,
                metadata={"topic": "AI"}
            ),
            RetrievalResult(
                chunk_id="chunk_2",
                content="Deep learning uses neural networks with multiple layers.",
                document_id="doc_2",
                similarity_score=0.87,
                content_type=ContentType.TEXT,
                metadata={"topic": "ML"}
            )
        ]
        
        with patch.object(retrieval_system, 'search') as mock_search:
            mock_search.return_value = mock_results
            
            retrieval_system.initialize()
            response = retrieval_system.generate_grounded_response(
                query="What is machine learning?",
                max_sources=2
            )
            
            assert isinstance(response, GroundedResponse)
            assert response.query == "What is machine learning?"
            assert len(response.citations) == 2
            assert response.citations[0].chunk_id == "chunk_1"
            assert response.citations[0].similarity_score == 0.95
            assert response.citations[1].chunk_id == "chunk_2"
            assert response.citations[1].similarity_score == 0.87
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_generate_grounded_response_no_results(self, mock_init_components, retrieval_system):
        """Test grounded response generation with no search results."""
        mock_init_components.return_value = True
        
        with patch.object(retrieval_system, 'search') as mock_search:
            mock_search.return_value = []
            
            retrieval_system.initialize()
            response = retrieval_system.generate_grounded_response(
                query="What is quantum computing?",
                max_sources=5
            )
            
            assert isinstance(response, GroundedResponse)
            assert response.query == "What is quantum computing?"
            assert len(response.citations) == 0
            assert "No relevant information found" in response.response_text
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_delete_document(self, mock_init_components, retrieval_system):
        """Test document deletion."""
        mock_init_components.return_value = True
        
        with patch.object(retrieval_system.vector_store, 'delete_document') as mock_delete:
            mock_delete.return_value = True
            
            retrieval_system.initialize()
            result = retrieval_system.delete_document("doc_1")
            
            assert result is True
            mock_delete.assert_called_once_with("doc_1")
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_get_system_stats(self, mock_init_components, retrieval_system):
        """Test getting system statistics."""
        mock_init_components.return_value = True
        
        # Mock collection stats
        mock_collection_stats = {
            'total_chunks': 100,
            'vector_dimension': 512,
            'collection_name': 'test_collection'
        }
        
        with patch.object(retrieval_system.vector_store, 'get_collection_stats') as mock_stats:
            mock_stats.return_value = mock_collection_stats
            
            retrieval_system.initialize()
            stats = retrieval_system.get_system_stats()
            
            assert stats['total_chunks'] == 100
            assert stats['vector_dimension'] == 512
            assert stats['collection_name'] == 'test_collection'
            assert stats['is_initialized'] is True
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_clear_system(self, mock_init_components, retrieval_system):
        """Test clearing the system."""
        mock_init_components.return_value = True
        
        with patch.object(retrieval_system.vector_store, 'clear_collection') as mock_clear:
            mock_clear.return_value = True
            
            retrieval_system.initialize()
            result = retrieval_system.clear_system()
            
            assert result is True
            mock_clear.assert_called_once()
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_multimodal_search_text_query(self, mock_init_components, retrieval_system):
        """Test multimodal search with text query."""
        mock_init_components.return_value = True
        
        mock_results = [
            RetrievalResult(
                chunk_id="chunk_1",
                content="Text about cats",
                document_id="doc_1",
                similarity_score=0.90,
                content_type=ContentType.TEXT
            )
        ]
        
        with patch.object(retrieval_system.vector_store, 'search_similar') as mock_search:
            mock_search.return_value = mock_results
            
            retrieval_system.initialize()
            results = retrieval_system.multimodal_search(
                query="cats",
                content_types=[ContentType.TEXT, ContentType.IMAGE],
                top_k=5
            )
            
            assert len(results) == 1
            assert results[0].content_type == ContentType.TEXT
            mock_search.assert_called_once()
    
    @patch('src.retrieval.retrieval_system.MultimodalRetrievalSystem._initialize_components')
    def test_multimodal_search_image_query(self, mock_init_components, retrieval_system):
        """Test multimodal search with image query."""
        mock_init_components.return_value = True
        
        mock_results = [
            RetrievalResult(
                chunk_id="img_chunk_1",
                content="/path/to/cat_image.jpg",
                document_id="img_doc_1",
                similarity_score=0.88,
                content_type=ContentType.IMAGE
            )
        ]
        
        with patch.object(retrieval_system.vector_store, 'search_similar') as mock_search:
            mock_search.return_value = mock_results
            
            retrieval_system.initialize()
            results = retrieval_system.multimodal_search(
                query="/path/to/query_image.jpg",
                content_types=[ContentType.TEXT, ContentType.IMAGE],
                top_k=5
            )
            
            assert len(results) == 1
            assert results[0].content_type == ContentType.IMAGE
            mock_search.assert_called_once()
    
    def test_validate_query_input(self, retrieval_system):
        """Test query input validation."""
        # Valid text query
        assert retrieval_system._validate_query_input("What is AI?") is True
        
        # Valid file path query
        assert retrieval_system._validate_query_input("/path/to/image.jpg") is True
        
        # Invalid empty query
        with pytest.raises(ValueError, match="Query cannot be empty"):
            retrieval_system._validate_query_input("")
        
        # Invalid None query
        with pytest.raises(ValueError, match="Query cannot be None"):
            retrieval_system._validate_query_input(None)
    
    def test_build_response_text(self, retrieval_system):
        """Test building response text from search results."""
        mock_results = [
            RetrievalResult(
                chunk_id="chunk_1",
                content="Machine learning is a subset of AI.",
                document_id="doc_1",
                similarity_score=0.95,
                content_type=ContentType.TEXT
            ),
            RetrievalResult(
                chunk_id="chunk_2",
                content="Deep learning uses neural networks.",
                document_id="doc_2",
                similarity_score=0.87,
                content_type=ContentType.TEXT
            )
        ]
        
        response_text = retrieval_system._build_response_text(
            "What is machine learning?",
            mock_results
        )
        
        assert "Machine learning is a subset of AI" in response_text
        assert "Deep learning uses neural networks" in response_text
        assert len(response_text) > 0
    
    def test_build_response_text_no_results(self, retrieval_system):
        """Test building response text with no results."""
        response_text = retrieval_system._build_response_text(
            "What is quantum computing?",
            []
        )
        
        assert "No relevant information found" in response_text
        assert "quantum computing" in response_text.lower()


if __name__ == "__main__":
    pytest.main([__file__])
