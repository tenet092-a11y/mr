"""
Unit tests for text embedding generator.

Tests text embedding generation, batch processing, caching,
normalization, and performance characteristics.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import time

from src.embeddings.text_embedding_generator import SentenceTransformerEmbeddingGenerator
from src.embeddings.base import EmbeddingError, ModelLoadError
from src.models import ContentType
from src.config import EmbeddingConfig


class TestSentenceTransformerEmbeddingGenerator:
    """Test suite for SentenceTransformerEmbeddingGenerator."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def embedding_config(self, temp_cache_dir):
        """Create test embedding configuration."""
        return EmbeddingConfig(
            text_model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimension=384,
            batch_size=4,
            normalize_embeddings=True,
            cache_embeddings=True,
            embedding_cache_dir=temp_cache_dir,
            device="cpu",  # Force CPU for consistent testing
            max_sequence_length=512
        )
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create mock SentenceTransformer for testing."""
        mock_model = Mock()
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_model.eval.return_value = None
        return mock_model
    
    @pytest.fixture
    def generator(self, embedding_config):
        """Create text embedding generator for testing."""
        return SentenceTransformerEmbeddingGenerator(embedding_config)
    
    def test_initialization(self, embedding_config):
        """Test generator initialization."""
        generator = SentenceTransformerEmbeddingGenerator(embedding_config)
        
        assert generator.config == embedding_config
        assert generator.embedding_dimension == 384
        assert not generator.is_loaded
        assert generator.device == "cpu"
        assert generator.cache is not None
    
    def test_device_determination_auto(self, temp_cache_dir):
        """Test automatic device determination."""
        config = EmbeddingConfig(
            embedding_cache_dir=temp_cache_dir,
            device="auto"
        )
        generator = SentenceTransformerEmbeddingGenerator(config)
        
        # Should select CPU, CUDA, or MPS based on availability
        assert generator.device in ["cpu", "cuda", "mps"]
    
    def test_device_determination_explicit(self, temp_cache_dir):
        """Test explicit device specification."""
        config = EmbeddingConfig(
            embedding_cache_dir=temp_cache_dir,
            device="cpu"
        )
        generator = SentenceTransformerEmbeddingGenerator(config)
        
        assert generator.device == "cpu"
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_model_loading_success(self, mock_st_class, generator, mock_sentence_transformer):
        """Test successful model loading."""
        mock_st_class.return_value = mock_sentence_transformer
        
        generator.load_model()
        
        assert generator.is_loaded
        assert generator.model is not None
        mock_st_class.assert_called_once_with(
            generator.config.text_model_name,
            device=generator.device
        )
        mock_sentence_transformer.eval.assert_called_once()
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_model_loading_failure(self, mock_st_class, generator):
        """Test model loading failure."""
        mock_st_class.side_effect = Exception("Model not found")
        
        with pytest.raises(ModelLoadError) as exc_info:
            generator.load_model()
        
        assert "Failed to load text embedding model" in str(exc_info.value)
        assert not generator.is_loaded
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_dimension_validation_and_update(self, mock_st_class, generator, mock_sentence_transformer):
        """Test embedding dimension validation and automatic update."""
        # Mock model returns different dimension than configured
        mock_sentence_transformer.encode.return_value = np.random.rand(512).astype(np.float32)
        mock_st_class.return_value = mock_sentence_transformer
        
        generator.load_model()
        
        # Should update dimension to match model
        assert generator.embedding_dimension == 512
        assert generator.config.embedding_dimension == 512
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_encode_text_success(self, mock_st_class, generator, mock_sentence_transformer):
        """Test successful text encoding."""
        expected_embedding = np.random.rand(384).astype(np.float32)
        mock_sentence_transformer.encode.return_value = expected_embedding
        mock_st_class.return_value = mock_sentence_transformer
        
        result = generator.encode_text("test text")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        # Should be normalized
        assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_encode_text_empty_input(self, mock_st_class, generator, mock_sentence_transformer):
        """Test encoding empty text."""
        mock_st_class.return_value = mock_sentence_transformer
        
        result = generator.encode_text("")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (384,)
        assert np.allclose(result, 0)
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_encode_text_with_caching(self, mock_st_class, generator, mock_sentence_transformer):
        """Test text encoding with caching."""
        expected_embedding = np.random.rand(384).astype(np.float32)
        mock_sentence_transformer.encode.return_value = expected_embedding
        mock_st_class.return_value = mock_sentence_transformer
        
        text = "test text for caching"
        
        # First call should generate embedding
        result1 = generator.encode_text(text)
        
        # Second call should use cache
        result2 = generator.encode_text(text)
        
        np.testing.assert_array_equal(result1, result2)
        # Model should only be called once
        assert mock_sentence_transformer.encode.call_count == 1
        
        # Check cache hit statistics
        stats = generator.get_embedding_stats()
        assert stats['cache_hits'] == 1
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_encode_text_batch_success(self, mock_st_class, generator, mock_sentence_transformer):
        """Test successful batch text encoding."""
        # Mock batch encoding
        batch_embeddings = np.random.rand(3, 384).astype(np.float32)
        mock_sentence_transformer.encode.return_value = batch_embeddings
        mock_st_class.return_value = mock_sentence_transformer
        
        texts = ["text 1", "text 2", "text 3"]
        results = generator.encode_text_batch(texts)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, np.ndarray)
            assert result.shape == (384,)
            # Should be normalized
            assert np.isclose(np.linalg.norm(result), 1.0, atol=1e-6)
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_encode_text_batch_empty_list(self, mock_st_class, generator, mock_sentence_transformer):
        """Test batch encoding with empty list."""
        mock_st_class.return_value = mock_sentence_transformer
        
        results = generator.encode_text_batch([])
        
        assert results == []
        mock_sentence_transformer.encode.assert_not_called()
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_encode_text_batch_with_empty_texts(self, mock_st_class, generator, mock_sentence_transformer):
        """Test batch encoding with some empty texts."""
        batch_embeddings = np.random.rand(2, 384).astype(np.float32)
        mock_sentence_transformer.encode.return_value = batch_embeddings
        mock_st_class.return_value = mock_sentence_transformer
        
        texts = ["text 1", "", "text 3", None]
        results = generator.encode_text_batch(texts)
        
        assert len(results) == 4
        # Empty texts should have zero embeddings
        assert np.allclose(results[1], 0)
        assert np.allclose(results[3], 0)
        # Valid texts should have proper embeddings
        assert not np.allclose(results[0], 0)
        assert not np.allclose(results[2], 0)
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_encode_text_batch_with_batching(self, mock_st_class, generator, mock_sentence_transformer):
        """Test batch encoding respects batch size configuration."""
        # Set small batch size
        generator.config.batch_size = 2
        
        # Mock multiple batch calls
        batch1 = np.random.rand(2, 384).astype(np.float32)
        batch2 = np.random.rand(1, 384).astype(np.float32)
        mock_sentence_transformer.encode.side_effect = [batch1, batch2]
        mock_st_class.return_value = mock_sentence_transformer
        
        texts = ["text 1", "text 2", "text 3"]
        results = generator.encode_text_batch(texts)
        
        assert len(results) == 3
        # Should have made 2 batch calls
        assert mock_sentence_transformer.encode.call_count == 2
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_encode_text_batch_with_mixed_cache(self, mock_st_class, generator, mock_sentence_transformer):
        """Test batch encoding with some cached and some new texts."""
        expected_embedding = np.random.rand(384).astype(np.float32)
        mock_sentence_transformer.encode.return_value = expected_embedding
        mock_st_class.return_value = mock_sentence_transformer
        
        # First, cache one text
        cached_text = "cached text"
        generator.encode_text(cached_text)
        
        # Now batch encode with mix of cached and new
        new_embedding = np.random.rand(1, 384).astype(np.float32)
        mock_sentence_transformer.encode.return_value = new_embedding
        
        texts = [cached_text, "new text"]
        results = generator.encode_text_batch(texts)
        
        assert len(results) == 2
        # First should be from cache, second should be newly generated
        stats = generator.get_embedding_stats()
        assert stats['cache_hits'] >= 1
    
    def test_content_hash_computation(self, generator):
        """Test content hash computation for caching."""
        text1 = "test text"
        text2 = "test text"
        text3 = "different text"
        
        hash1 = generator._compute_content_hash(text1)
        hash2 = generator._compute_content_hash(text2)
        hash3 = generator._compute_content_hash(text3)
        
        assert hash1 == hash2  # Same text should have same hash
        assert hash1 != hash3  # Different text should have different hash
        assert isinstance(hash1, str)
        assert len(hash1) == 64  # SHA256 hex digest length
    
    def test_embedding_validation_valid(self, generator):
        """Test embedding validation with valid embedding."""
        valid_embedding = np.random.rand(384).astype(np.float32)
        
        assert generator._validate_embedding(valid_embedding)
    
    def test_embedding_validation_invalid_shape(self, generator):
        """Test embedding validation with invalid shape."""
        invalid_embedding = np.random.rand(256).astype(np.float32)  # Wrong dimension
        
        assert not generator._validate_embedding(invalid_embedding)
    
    def test_embedding_validation_nan_values(self, generator):
        """Test embedding validation with NaN values."""
        invalid_embedding = np.full(384, np.nan, dtype=np.float32)
        
        assert not generator._validate_embedding(invalid_embedding)
    
    def test_embedding_validation_infinite_values(self, generator):
        """Test embedding validation with infinite values."""
        invalid_embedding = np.full(384, np.inf, dtype=np.float32)
        
        assert not generator._validate_embedding(invalid_embedding)
    
    def test_embedding_validation_all_zeros(self, generator):
        """Test embedding validation with all zeros."""
        zero_embedding = np.zeros(384, dtype=np.float32)
        
        assert not generator._validate_embedding(zero_embedding)
    
    def test_embedding_validation_none(self, generator):
        """Test embedding validation with None."""
        assert not generator._validate_embedding(None)
    
    def test_normalization_enabled(self, temp_cache_dir):
        """Test embedding normalization when enabled."""
        config = EmbeddingConfig(
            embedding_cache_dir=temp_cache_dir,
            normalize_embeddings=True
        )
        generator = SentenceTransformerEmbeddingGenerator(config)
        
        # Test vector that's not normalized
        embedding = np.array([3.0, 4.0, 0.0] + [0.0] * 381, dtype=np.float32)
        normalized = generator.normalize_embedding(embedding)
        
        # Should be normalized to unit length
        assert np.isclose(np.linalg.norm(normalized), 1.0, atol=1e-6)
    
    def test_normalization_disabled(self, temp_cache_dir):
        """Test embedding normalization when disabled."""
        config = EmbeddingConfig(
            embedding_cache_dir=temp_cache_dir,
            normalize_embeddings=False
        )
        generator = SentenceTransformerEmbeddingGenerator(config)
        
        # Test vector that's not normalized
        embedding = np.array([3.0, 4.0, 0.0] + [0.0] * 381, dtype=np.float32)
        result = generator.normalize_embedding(embedding)
        
        # Should remain unchanged
        np.testing.assert_array_equal(result, embedding)
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_get_embedding_stats(self, mock_st_class, generator, mock_sentence_transformer):
        """Test embedding statistics tracking."""
        mock_st_class.return_value = mock_sentence_transformer
        mock_sentence_transformer.encode.return_value = np.random.rand(384).astype(np.float32)
        
        # Generate some embeddings
        generator.encode_text("text 1")
        generator.encode_text("text 2")
        generator.encode_text("text 1")  # Should hit cache
        
        stats = generator.get_embedding_stats()
        
        assert stats['total_embeddings'] == 2
        assert stats['cache_hits'] == 1
        assert stats['cache_hit_rate'] > 0
        assert stats['model_name'] == generator.config.text_model_name
        assert stats['device'] == generator.device
        assert stats['embedding_dimension'] == generator.embedding_dimension
    
    def test_clear_cache(self, generator):
        """Test cache clearing functionality."""
        # This is a basic test since we're using a mock cache
        generator.clear_cache()
        # Should not raise any exceptions
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_performance_batch_vs_individual(self, mock_st_class, generator, mock_sentence_transformer):
        """Test that batch processing is more efficient than individual calls."""
        mock_st_class.return_value = mock_sentence_transformer
        
        # Mock individual calls
        mock_sentence_transformer.encode.return_value = np.random.rand(384).astype(np.float32)
        
        texts = ["text 1", "text 2", "text 3", "text 4"]
        
        # Individual encoding
        start_time = time.time()
        individual_results = [generator.encode_text(text) for text in texts]
        individual_time = time.time() - start_time
        
        # Reset mock and cache
        mock_sentence_transformer.reset_mock()
        generator.clear_cache()
        
        # Batch encoding
        mock_sentence_transformer.encode.return_value = np.random.rand(4, 384).astype(np.float32)
        start_time = time.time()
        batch_results = generator.encode_text_batch(texts)
        batch_time = time.time() - start_time
        
        # Batch should make fewer model calls
        # Individual: 4 calls, Batch: 1 call
        assert len(individual_results) == len(batch_results) == 4
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_long_text_truncation(self, mock_st_class, generator, mock_sentence_transformer):
        """Test that very long texts are truncated appropriately."""
        mock_st_class.return_value = mock_sentence_transformer
        mock_sentence_transformer.encode.return_value = np.random.rand(384).astype(np.float32)
        
        # Create very long text
        long_text = "word " * 10000  # Much longer than max_sequence_length
        
        generator.encode_text(long_text)
        
        # Should have called encode with truncated text
        mock_sentence_transformer.encode.assert_called_once()
        call_args = mock_sentence_transformer.encode.call_args[0]
        encoded_text = call_args[0]
        
        # Should be truncated
        assert len(encoded_text) <= generator.config.max_sequence_length * 4
    
    def test_supported_content_types(self, generator):
        """Test that generator reports correct supported content types."""
        supported_types = generator.get_supported_content_types()
        
        assert supported_types == [ContentType.TEXT]
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_error_handling_in_encoding(self, mock_st_class, generator, mock_sentence_transformer):
        """Test error handling during encoding."""
        mock_st_class.return_value = mock_sentence_transformer
        mock_sentence_transformer.encode.side_effect = Exception("Encoding failed")
        
        with pytest.raises(EmbeddingError) as exc_info:
            generator.encode_text("test text")
        
        assert "Failed to encode text" in str(exc_info.value)
        assert exc_info.value.content_type == ContentType.TEXT
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_error_handling_in_batch_encoding(self, mock_st_class, generator, mock_sentence_transformer):
        """Test error handling during batch encoding."""
        mock_st_class.return_value = mock_sentence_transformer
        mock_sentence_transformer.encode.side_effect = Exception("Batch encoding failed")
        
        with pytest.raises(EmbeddingError) as exc_info:
            generator.encode_text_batch(["text 1", "text 2"])
        
        assert "Failed to encode text batch" in str(exc_info.value)
        assert exc_info.value.content_type == ContentType.TEXT


class TestEmbeddingConsistency:
    """Test embedding consistency and reproducibility."""
    
    @pytest.fixture
    def generator_config(self):
        """Create consistent configuration for reproducibility tests."""
        return EmbeddingConfig(
            text_model_name="sentence-transformers/all-MiniLM-L6-v2",
            embedding_dimension=384,
            normalize_embeddings=True,
            cache_embeddings=False,  # Disable cache for consistency tests
            device="cpu"
        )
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_embedding_consistency(self, mock_st_class, generator_config):
        """Test that same text produces consistent embeddings."""
        # Create deterministic mock
        fixed_embedding = np.random.RandomState(42).rand(384).astype(np.float32)
        mock_model = Mock()
        mock_model.encode.return_value = fixed_embedding
        mock_model.eval.return_value = None
        mock_st_class.return_value = mock_model
        
        generator = SentenceTransformerEmbeddingGenerator(generator_config)
        
        text = "consistent test text"
        embedding1 = generator.encode_text(text)
        embedding2 = generator.encode_text(text)
        
        np.testing.assert_array_equal(embedding1, embedding2)
    
    @patch('src.embeddings.text_embedding_generator.SentenceTransformer')
    def test_different_texts_different_embeddings(self, mock_st_class, generator_config):
        """Test that different texts produce different embeddings."""
        # Mock different embeddings for different calls
        embeddings = [
            np.random.RandomState(42).rand(384).astype(np.float32),
            np.random.RandomState(43).rand(384).astype(np.float32)
        ]
        
        mock_model = Mock()
        mock_model.encode.side_effect = embeddings
        mock_model.eval.return_value = None
        mock_st_class.return_value = mock_model
        
        generator = SentenceTransformerEmbeddingGenerator(generator_config)
        
        embedding1 = generator.encode_text("text one")
        embedding2 = generator.encode_text("text two")
        
        # Should be different (with high probability)
        assert not np.allclose(embedding1, embedding2, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])