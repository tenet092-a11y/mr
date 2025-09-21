"""
Unified embedding generator for cross-modal content.

Implements a unified vector space mapping for text, image, and audio content
using sentence-transformers for text, CLIP for images, and text embeddings
for transcribed audio. Includes caching to avoid recomputation.
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    import torch
    import clip
    from PIL import Image
except ImportError as e:
    raise ImportError(
        "Required dependencies missing. Install with: "
        "pip install sentence-transformers torch clip-by-openai pillow"
    ) from e

from .base import MultimodalEmbeddingGenerator, EmbeddingCache, ModelLoadError, EmbeddingError
from .text_embedding_generator import SentenceTransformerEmbeddingGenerator
from ..models import ContentType, ContentChunk
from ..config import EmbeddingConfig


logger = logging.getLogger(__name__)


class UnifiedEmbeddingGenerator(MultimodalEmbeddingGenerator):
    """
    Unified embedding generator that handles all content types.
    
    Maps text, image, and audio content into a unified vector space for
    cross-modal retrieval. Uses sentence-transformers for text, CLIP for images,
    and text embeddings for transcribed audio content.
    
    Features:
    - Cross-modal embedding generation
    - Unified vector space mapping
    - Embedding caching for performance
    - Batch processing support
    - Device optimization (CPU/GPU)
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        Initialize unified embedding generator.
        
        Args:
            config: Embedding configuration
        """
        super().__init__(config)
        
        # Initialize component models
        self.text_model: Optional[SentenceTransformer] = None
        self.clip_model: Optional[Any] = None
        self.clip_preprocess: Optional[Any] = None
        
        # Device configuration
        self.device = self._determine_device()
        
        # Initialize cache
        self.cache = EmbeddingCache(
            cache_dir=config.embedding_cache_dir,
            enabled=config.cache_embeddings
        )
        
        # Performance tracking
        self._embedding_stats = {
            'total_embeddings': 0,
            'cache_hits': 0,
            'text_embeddings': 0,
            'image_embeddings': 0,
            'audio_embeddings': 0,
            'batch_operations': 0
        }
        
        # Model loading flags
        self._text_model_loaded = False
        self._clip_model_loaded = False
        
        logger.info(f"UnifiedEmbeddingGenerator initialized with device: {self.device}")
    
    def _determine_device(self) -> str:
        """
        Determine the best available device for embedding generation.
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if self.config.device != "auto":
            return self.config.device
        
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def load_model(self) -> None:
        """
        Load all embedding models.
        
        Raises:
            ModelLoadError: If any model loading fails
        """
        try:
            self._load_text_model()
            self._load_clip_model()
            self.is_loaded = True
            logger.info("All embedding models loaded successfully")
            
        except Exception as e:
            error_msg = f"Failed to load unified embedding models: {str(e)}"
            logger.error(error_msg)
            raise ModelLoadError(error_msg, cause=e)
    
    def _load_text_model(self) -> None:
        """Load sentence transformer model for text embeddings."""
        if self._text_model_loaded:
            return
            
        try:
            logger.info(f"Loading text embedding model: {self.config.text_model_name}")
            self.text_model = SentenceTransformer(
                self.config.text_model_name,
                device=self.device
            )
            
            # Validate embedding dimension
            test_embedding = self.text_model.encode("test", convert_to_numpy=True)
            actual_dimension = test_embedding.shape[0]
            
            if actual_dimension != self.config.embedding_dimension:
                logger.warning(
                    f"Text model embedding dimension ({actual_dimension}) differs from "
                    f"configured dimension ({self.config.embedding_dimension}). "
                    f"Updating configuration."
                )
                self.embedding_dimension = actual_dimension
                self.config.embedding_dimension = actual_dimension
            
            self.text_model.eval()
            self._text_model_loaded = True
            logger.info(f"Text model loaded successfully. Dimension: {self.embedding_dimension}")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load text model: {str(e)}", cause=e)
    
    def _load_clip_model(self) -> None:
        """Load CLIP model for image embeddings."""
        if self._clip_model_loaded:
            return
            
        try:
            logger.info(f"Loading CLIP model: {self.config.image_model_name}")
            self.clip_model, self.clip_preprocess = clip.load(
                self.config.image_model_name, 
                device=self.device
            )
            
            # Test CLIP embedding dimension
            test_image = Image.new('RGB', (224, 224), color='white')
            test_tensor = self.clip_preprocess(test_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                test_embedding = self.clip_model.encode_image(test_tensor)
                clip_dimension = test_embedding.shape[1]
            
            logger.info(f"CLIP model loaded. Image embedding dimension: {clip_dimension}")
            
            # Store CLIP dimension for mapping
            self._clip_dimension = clip_dimension
            self._clip_model_loaded = True
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load CLIP model: {str(e)}", cause=e)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text content into embedding vector.
        
        Args:
            text: Text content to encode
            
        Returns:
            Normalized embedding vector
            
        Raises:
            EmbeddingError: If encoding fails
        """
        if not self._text_model_loaded:
            self._load_text_model()
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self.embedding_dimension, dtype=np.float32)
        
        # Check cache first
        content_hash = self._compute_content_hash(text, ContentType.TEXT)
        cached_embedding = self.cache.get_embedding(content_hash)
        if cached_embedding is not None:
            self._embedding_stats['cache_hits'] += 1
            return cached_embedding
        
        try:
            # Truncate text if too long
            if len(text) > self.config.max_sequence_length * 4:
                text = text[:self.config.max_sequence_length * 4]
                logger.debug(f"Truncated long text to {len(text)} characters")
            
            # Generate embedding
            embedding = self.text_model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False
            )
            
            # Validate and normalize
            if not self._validate_embedding(embedding):
                raise EmbeddingError(f"Invalid text embedding generated for: {text[:100]}...")
            
            embedding = self.normalize_embedding(embedding)
            
            # Cache the embedding
            self.cache.store_embedding(content_hash, embedding)
            
            self._embedding_stats['total_embeddings'] += 1
            self._embedding_stats['text_embeddings'] += 1
            
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to encode text: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, ContentType.TEXT, cause=e)
    
    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Encode image content into embedding vector.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Normalized embedding vector mapped to text space
            
        Raises:
            EmbeddingError: If encoding fails
        """
        if not self._clip_model_loaded:
            self._load_clip_model()
        
        # Check cache first
        content_hash = self._compute_content_hash(image_path, ContentType.IMAGE)
        cached_embedding = self.cache.get_embedding(content_hash)
        if cached_embedding is not None:
            self._embedding_stats['cache_hits'] += 1
            return cached_embedding
        
        try:
            # Load and preprocess image
            with Image.open(image_path) as image:
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Preprocess for CLIP
                image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
                
                # Generate CLIP embedding
                with torch.no_grad():
                    clip_embedding = self.clip_model.encode_image(image_tensor)
                    # Normalize CLIP embedding
                    clip_embedding = clip_embedding / clip_embedding.norm(dim=-1, keepdim=True)
                    clip_embedding = clip_embedding.cpu().numpy().flatten()
                
                # Map CLIP embedding to text embedding space
                unified_embedding = self._map_clip_to_text_space(clip_embedding)
                
                # Validate and normalize
                if not self._validate_embedding(unified_embedding):
                    raise EmbeddingError(f"Invalid image embedding generated for: {image_path}")
                
                unified_embedding = self.normalize_embedding(unified_embedding)
                
                # Cache the embedding
                self.cache.store_embedding(content_hash, unified_embedding)
                
                self._embedding_stats['total_embeddings'] += 1
                self._embedding_stats['image_embeddings'] += 1
                
                return unified_embedding
                
        except Exception as e:
            error_msg = f"Failed to encode image {image_path}: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, ContentType.IMAGE, cause=e)
    
    def encode_audio(self, audio_content: str) -> np.ndarray:
        """
        Encode audio content (transcription) into embedding vector.
        
        Args:
            audio_content: Transcribed text from audio
            
        Returns:
            Normalized embedding vector
            
        Raises:
            EmbeddingError: If encoding fails
        """
        # Audio content is transcribed text, so we use text encoding
        # but with special handling for audio-specific characteristics
        
        if not audio_content or not audio_content.strip():
            logger.warning("Empty audio content provided for embedding")
            return np.zeros(self.embedding_dimension, dtype=np.float32)
        
        # Check cache first
        content_hash = self._compute_content_hash(audio_content, ContentType.AUDIO)
        cached_embedding = self.cache.get_embedding(content_hash)
        if cached_embedding is not None:
            self._embedding_stats['cache_hits'] += 1
            return cached_embedding
        
        try:
            # Preprocess audio transcription text
            processed_text = self._preprocess_audio_text(audio_content)
            
            # Generate text embedding for transcribed content
            embedding = self.encode_text(processed_text)
            
            # Apply audio-specific transformation to distinguish from regular text
            audio_embedding = self._apply_audio_transformation(embedding)
            
            # Cache the embedding
            self.cache.store_embedding(content_hash, audio_embedding)
            
            self._embedding_stats['total_embeddings'] += 1
            self._embedding_stats['audio_embeddings'] += 1
            
            return audio_embedding
            
        except Exception as e:
            error_msg = f"Failed to encode audio content: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, ContentType.AUDIO, cause=e)
    
    def generate_embedding(self, content: str, content_type: ContentType) -> np.ndarray:
        """
        Generate embedding for content based on its type.
        
        Args:
            content: Content to embed (text, image path, or transcribed audio)
            content_type: Type of content being embedded
            
        Returns:
            Normalized embedding vector in unified space
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if content_type == ContentType.TEXT:
            return self.encode_text(content)
        elif content_type == ContentType.IMAGE:
            return self.encode_image(content)
        elif content_type == ContentType.AUDIO:
            return self.encode_audio(content)
        else:
            raise EmbeddingError(f"Unsupported content type: {content_type}")
    
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
        if len(contents) != len(content_types):
            raise EmbeddingError("Contents and content_types lists must have same length")
        
        if not contents:
            return []
        
        embeddings = []
        
        # Group by content type for efficient batch processing
        text_indices = []
        image_indices = []
        audio_indices = []
        
        for i, content_type in enumerate(content_types):
            if content_type == ContentType.TEXT:
                text_indices.append(i)
            elif content_type == ContentType.IMAGE:
                image_indices.append(i)
            elif content_type == ContentType.AUDIO:
                audio_indices.append(i)
            else:
                raise EmbeddingError(f"Unsupported content type: {content_type}")
        
        # Initialize results array
        embeddings = [None] * len(contents)
        
        try:
            # Process text content in batch
            if text_indices:
                text_contents = [contents[i] for i in text_indices]
                text_embeddings = self._encode_text_batch(text_contents)
                for idx, embedding in zip(text_indices, text_embeddings):
                    embeddings[idx] = embedding
            
            # Process images individually (CLIP doesn't batch well with file paths)
            for idx in image_indices:
                embeddings[idx] = self.encode_image(contents[idx])
            
            # Process audio content (transcribed text) in batch
            if audio_indices:
                audio_contents = [contents[i] for i in audio_indices]
                audio_embeddings = self._encode_audio_batch(audio_contents)
                for idx, embedding in zip(audio_indices, audio_embeddings):
                    embeddings[idx] = embedding
            
            self._embedding_stats['batch_operations'] += 1
            
            return embeddings
            
        except Exception as e:
            error_msg = f"Failed to generate batch embeddings: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg, cause=e)
    
    def _encode_text_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode multiple texts efficiently."""
        if not self._text_model_loaded:
            self._load_text_model()
        
        # Filter and prepare texts
        valid_texts = []
        valid_indices = []
        text_hashes = []
        
        for i, text in enumerate(texts):
            if text and text.strip():
                content_hash = self._compute_content_hash(text, ContentType.TEXT)
                cached_embedding = self.cache.get_embedding(content_hash)
                
                if cached_embedding is None:
                    # Truncate if needed
                    if len(text) > self.config.max_sequence_length * 4:
                        text = text[:self.config.max_sequence_length * 4]
                    valid_texts.append(text)
                    valid_indices.append(i)
                    text_hashes.append(content_hash)
                else:
                    self._embedding_stats['cache_hits'] += 1
        
        # Initialize result array
        embeddings = [np.zeros(self.embedding_dimension, dtype=np.float32) for _ in texts]
        
        # Fill cached embeddings
        for i, text in enumerate(texts):
            if text and text.strip():
                content_hash = self._compute_content_hash(text, ContentType.TEXT)
                cached_embedding = self.cache.get_embedding(content_hash)
                if cached_embedding is not None:
                    embeddings[i] = cached_embedding
        
        # Process non-cached texts
        if valid_texts:
            batch_embeddings = self.text_model.encode(
                valid_texts,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
                batch_size=self.config.batch_size
            )
            
            for embedding, original_idx, content_hash in zip(batch_embeddings, valid_indices, text_hashes):
                if self._validate_embedding(embedding):
                    embedding = self.normalize_embedding(embedding)
                    self.cache.store_embedding(content_hash, embedding)
                    embeddings[original_idx] = embedding
                else:
                    logger.warning(f"Invalid embedding generated for text at index {original_idx}")
            
            self._embedding_stats['total_embeddings'] += len(valid_texts)
            self._embedding_stats['text_embeddings'] += len(valid_texts)
        
        return embeddings
    
    def _encode_audio_batch(self, audio_contents: List[str]) -> List[np.ndarray]:
        """Encode multiple audio transcriptions efficiently."""
        # Preprocess all audio texts
        processed_texts = [self._preprocess_audio_text(content) for content in audio_contents]
        
        # Generate text embeddings
        text_embeddings = self._encode_text_batch(processed_texts)
        
        # Apply audio transformations
        audio_embeddings = []
        for i, embedding in enumerate(text_embeddings):
            audio_embedding = self._apply_audio_transformation(embedding)
            
            # Cache with audio content hash
            content_hash = self._compute_content_hash(audio_contents[i], ContentType.AUDIO)
            self.cache.store_embedding(content_hash, audio_embedding)
            
            audio_embeddings.append(audio_embedding)
            
            self._embedding_stats['audio_embeddings'] += 1
        
        return audio_embeddings
    
    def _map_clip_to_text_space(self, clip_embedding: np.ndarray) -> np.ndarray:
        """
        Map CLIP embedding to text embedding space.
        
        This is a simplified mapping - in practice, you might want to train
        a more sophisticated mapping function.
        
        Args:
            clip_embedding: CLIP embedding vector
            
        Returns:
            Embedding mapped to text space
        """
        # Simple linear projection to match text embedding dimension
        if len(clip_embedding) == self.embedding_dimension:
            return clip_embedding
        elif len(clip_embedding) > self.embedding_dimension:
            # Truncate or use PCA-like reduction
            return clip_embedding[:self.embedding_dimension]
        else:
            # Pad with zeros or repeat
            padded = np.zeros(self.embedding_dimension, dtype=np.float32)
            padded[:len(clip_embedding)] = clip_embedding
            return padded
    
    def _preprocess_audio_text(self, audio_text: str) -> str:
        """
        Preprocess transcribed audio text for embedding.
        
        Args:
            audio_text: Transcribed text from audio
            
        Returns:
            Preprocessed text
        """
        # Basic cleaning for transcribed text
        processed = audio_text.strip()
        
        # Remove common transcription artifacts
        artifacts = ['[MUSIC]', '[NOISE]', '[INAUDIBLE]', '[SILENCE]', '[UNK]']
        for artifact in artifacts:
            processed = processed.replace(artifact, '')
        
        # Normalize whitespace
        processed = ' '.join(processed.split())
        
        # Add audio context marker for embedding distinction
        if processed:
            processed = f"[AUDIO] {processed}"
        
        return processed
    
    def _apply_audio_transformation(self, text_embedding: np.ndarray) -> np.ndarray:
        """
        Apply transformation to distinguish audio embeddings from text.
        
        Args:
            text_embedding: Text embedding of transcribed content
            
        Returns:
            Transformed embedding for audio content
        """
        # Simple transformation: slight rotation in embedding space
        # This helps distinguish audio content from regular text
        
        # Apply small rotation matrix (this is a simplified approach)
        rotation_factor = 0.1
        transformed = text_embedding.copy()
        
        # Apply small perturbation to create audio-specific embedding space
        if len(transformed) >= 2:
            # Simple 2D rotation on first two dimensions
            cos_theta = np.cos(rotation_factor)
            sin_theta = np.sin(rotation_factor)
            
            x, y = transformed[0], transformed[1]
            transformed[0] = cos_theta * x - sin_theta * y
            transformed[1] = sin_theta * x + cos_theta * y
        
        return transformed
    
    def _compute_content_hash(self, content: str, content_type: ContentType) -> str:
        """
        Compute hash for content caching.
        
        Args:
            content: Content to hash
            content_type: Type of content
            
        Returns:
            Content hash string
        """
        # Include content type and model info in hash
        cache_key = (
            f"{content_type.value}:"
            f"{self.config.text_model_name}:"
            f"{self.config.image_model_name}:"
            f"{content}:"
            f"{self.config.normalize_embeddings}"
        )
        return hashlib.sha256(cache_key.encode()).hexdigest()
    
    def _validate_embedding(self, embedding: np.ndarray) -> bool:
        """
        Validate that embedding is well-formed.
        
        Args:
            embedding: Embedding vector to validate
            
        Returns:
            True if embedding is valid, False otherwise
        """
        if embedding is None or not isinstance(embedding, np.ndarray):
            return False
        
        if embedding.shape != (self.embedding_dimension,):
            logger.warning(
                f"Embedding dimension mismatch: expected {self.embedding_dimension}, "
                f"got {embedding.shape}"
            )
            return False
        
        if np.isnan(embedding).any() or np.isinf(embedding).any():
            logger.warning("Embedding contains NaN or infinite values")
            return False
        
        if np.allclose(embedding, 0):
            logger.warning("Embedding is all zeros")
            return False
        
        return True
    
    def get_supported_content_types(self) -> List[ContentType]:
        """Get list of supported content types."""
        return [ContentType.TEXT, ContentType.IMAGE, ContentType.AUDIO]
    
    def get_embedding_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the embedding generator.
        
        Returns:
            Dictionary containing performance metrics
        """
        total_generated = (
            self._embedding_stats['text_embeddings'] + 
            self._embedding_stats['image_embeddings'] + 
            self._embedding_stats['audio_embeddings']
        )
        
        cache_hit_rate = 0.0
        if total_generated > 0:
            cache_hit_rate = self._embedding_stats['cache_hits'] / (total_generated + self._embedding_stats['cache_hits'])
        
        return {
            **self._embedding_stats,
            'cache_hit_rate': cache_hit_rate,
            'text_model_name': self.config.text_model_name,
            'image_model_name': self.config.image_model_name,
            'device': self.device,
            'embedding_dimension': self.embedding_dimension,
            'is_loaded': self.is_loaded,
            'text_model_loaded': self._text_model_loaded,
            'clip_model_loaded': self._clip_model_loaded
        }
    
    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self.cache.clear_cache()
        self._embedding_stats['cache_hits'] = 0
        logger.info("Unified embedding cache cleared")
    
    def __del__(self):
        """Cleanup resources when generator is destroyed."""
        try:
            if hasattr(self, 'text_model') and self.text_model is not None:
                del self.text_model
            
            if hasattr(self, 'clip_model') and self.clip_model is not None:
                del self.clip_model
            
            # Clear GPU memory if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass  # Ignore cleanup errors