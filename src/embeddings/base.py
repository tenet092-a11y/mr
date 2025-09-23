"""
Base interfaces for embedding generation components.

Defines abstract classes for text and multimodal embedding generation
to ensure consistent interfaces across different embedding models.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
import numpy as np

from ..models import ContentChunk, ContentType
from ..config import EmbeddingConfig


class EmbeddingGenerator(ABC):
    """
    Abstract base class for embedding generators.
    
    Provides a common interface for generating embeddings from different
    content types (text, images, audio) into a unified vector space.
    """
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.embedding_dimension: int = config.embedding_dimension
        self.is_loaded: bool = False
    
    @abstractmethod
    def load_model(self) -> None:
        """
        Load the embedding model into memory.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        pass
    
    @abstractmethod
    def generate_embedding(self, content: str, content_type: ContentType) -> np.ndarray:
        """
        Generate embedding for a single piece of content.
        
        Args:
            content: Content to embed (text, image path, etc.)
            content_type: Type of content being embedded
            
        Returns:
            Normalized embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        pass
    
    @abstractmethod
    def generate_batch_embeddings(
        self, 
        contents: List[str], 
        content_types: List[ContentType]
    ) -> List[np.ndarray]:
        """
        Generate embeddings for multiple pieces of content efficiently.
        
        Args:
            contents: List of content to embed
            content_types: List of content types corresponding to contents
            
        Returns:
            List of normalized embedding vectors
            
        Raises:
            EmbeddingError: If batch embedding generation fails
        """
        pass
    
    def embed_chunks(self, chunks: List[ContentChunk]) -> List[ContentChunk]:
        """
        Generate embeddings for content chunks and update them in place.
        
        Args:
            chunks: List of content chunks to embed
            
        Returns:
            List of chunks with embeddings added
        """
        if not self.is_loaded:
            self.load_model()
        
        # Prepare batch data
        contents = [chunk.content for chunk in chunks]
        content_types = [chunk.content_type for chunk in chunks]
        
        # Generate embeddings in batch
        embeddings = self.generate_batch_embeddings(contents, content_types)
        
        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this generator.
        
        Returns:
            Embedding vector dimension
        """
        return self.embedding_dimension
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize embedding vector to unit length.
        
        Args:
            embedding: Raw embedding vector
            
        Returns:
            Normalized embedding vector
        """
        if self.config.normalize_embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                return embedding / norm
        return embedding
    
    @abstractmethod
    def get_supported_content_types(self) -> List[ContentType]:
        """
        Get list of content types supported by this embedding generator.
        
        Returns:
            List of supported ContentType values
        """
        pass


class TextEmbeddingGenerator(EmbeddingGenerator):
    """
    Abstract base class for text-only embedding generators.
    
    Specialized for generating embeddings from textual content using
    transformer-based models like sentence-transformers.
    """
    
    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text into embedding vector.
        
        Args:
            text: Text content to encode
            
        Returns:
            Text embedding vector
        """
        pass
    
    @abstractmethod
    def encode_text_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode multiple texts into embedding vectors efficiently.
        
        Args:
            texts: List of text content to encode
            
        Returns:
            List of text embedding vectors
        """
        pass
    
    def generate_embedding(self, content: str, content_type: ContentType) -> np.ndarray:
        """Generate embedding for text content."""
        if content_type != ContentType.TEXT:
            raise EmbeddingError(f"TextEmbeddingGenerator only supports TEXT content, got {content_type}")
        return self.encode_text(content)
    
    def generate_batch_embeddings(
        self, 
        contents: List[str], 
        content_types: List[ContentType]
    ) -> List[np.ndarray]:
        """Generate embeddings for batch of text content."""
        # Validate all content types are TEXT
        for content_type in content_types:
            if content_type != ContentType.TEXT:
                raise EmbeddingError(f"TextEmbeddingGenerator only supports TEXT content, got {content_type}")
        
        return self.encode_text_batch(contents)
    
    def get_supported_content_types(self) -> List[ContentType]:
        """Text embedding generator only supports text content."""
        return [ContentType.TEXT]


class MultimodalEmbeddingGenerator(EmbeddingGenerator):
    """
    Abstract base class for multimodal embedding generators.
    
    Handles multiple content types (text, images, audio) and maps them
    into a unified vector space for cross-modal retrieval.
    """
    
    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text content into embedding vector."""
        pass
    
    @abstractmethod
    def encode_image(self, image_path: str) -> np.ndarray:
        """Encode image content into embedding vector."""
        pass
    
    @abstractmethod
    def encode_audio(self, audio_content: str) -> np.ndarray:
        """Encode audio content (transcription) into embedding vector."""
        pass
    
    def generate_embedding(self, content: str, content_type: ContentType) -> np.ndarray:
        """Generate embedding based on content type."""
        if content_type == ContentType.TEXT:
            return self.encode_text(content)
        elif content_type == ContentType.IMAGE:
            return self.encode_image(content)  # content is image path
        elif content_type == ContentType.AUDIO:
            return self.encode_audio(content)  # content is transcription
        else:
            raise EmbeddingError(f"Unsupported content type: {content_type}")
    
    def get_supported_content_types(self) -> List[ContentType]:
        """Multimodal generator supports all content types."""
        return [ContentType.TEXT, ContentType.PDF, ContentType.DOCX, ContentType.IMAGE, ContentType.AUDIO]


class EmbeddingCache:
    """
    Cache for storing and retrieving generated embeddings.
    
    Helps avoid recomputing embeddings for the same content,
    improving performance for repeated processing.
    """
    
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        self._memory_cache: Dict[str, np.ndarray] = {}
    
    def get_embedding(self, content_hash: str) -> Optional[np.ndarray]:
        """
        Retrieve cached embedding by content hash.
        
        Args:
            content_hash: Hash of the content
            
        Returns:
            Cached embedding or None if not found
        """
        if not self.enabled:
            return None
        
        # Check memory cache first
        if content_hash in self._memory_cache:
            return self._memory_cache[content_hash]
        
        # Check disk cache
        cache_file = f"{self.cache_dir}/{content_hash}.npy"
        try:
            if Path(cache_file).exists():
                embedding = np.load(cache_file)
                self._memory_cache[content_hash] = embedding
                return embedding
        except Exception:
            pass  # Cache miss or corruption
        
        return None
    
    def store_embedding(self, content_hash: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.
        
        Args:
            content_hash: Hash of the content
            embedding: Embedding vector to cache
        """
        if not self.enabled:
            return
        
        # Store in memory cache
        self._memory_cache[content_hash] = embedding
        
        # Store in disk cache
        try:
            from pathlib import Path
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            cache_file = f"{self.cache_dir}/{content_hash}.npy"
            np.save(cache_file, embedding)
        except Exception:
            pass  # Cache write failure is not critical
    
    def clear_cache(self) -> None:
        """Clear all cached embeddings."""
        self._memory_cache.clear()
        # Note: Disk cache clearing would be implemented here


class EmbeddingError(Exception):
    """Exception raised during embedding generation."""
    
    def __init__(self, message: str, content_type: Optional[ContentType] = None, cause: Optional[Exception] = None):
        self.message = message
        self.content_type = content_type
        self.cause = cause
        super().__init__(self.message)


class ModelLoadError(EmbeddingError):
    """Exception raised when embedding model loading fails."""
    pass