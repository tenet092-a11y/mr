"""
Semantic retrieval engine for multimodal content.

Implements core semantic search functionality with vector similarity search,
query embedding generation, result filtering, and cross-modal search capabilities.
Provides the main interface for retrieving relevant content from the vector store.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import time
import numpy as np
from dataclasses import dataclass
import math
import re

from ..models import (
    ContentType, RetrievalResult, SourceLocation, 
    ContentChunk, SearchResult
)
from ..config import SystemConfig, EmbeddingConfig
from ..embeddings.unified_embedding_generator import UnifiedEmbeddingGenerator
from .vectordb.qdrant_vector_store import QdrantVectorStore


logger = logging.getLogger(__name__)


@dataclass
class SearchFilter:
    """Filter conditions for semantic search."""
    content_types: Optional[List[ContentType]] = None
    document_ids: Optional[List[str]] = None
    file_paths: Optional[List[str]] = None
    date_range: Optional[Tuple[str, str]] = None
    confidence_threshold: float = 0.0
    metadata_filters: Optional[Dict[str, Any]] = None


@dataclass
class SearchConfig:
    """Configuration for semantic search operations."""
    similarity_threshold: float = 0.5
    max_results: int = 100
    enable_cross_modal: bool = True
    enable_reranking: bool = True
    diversity_threshold: float = 0.8
    query_expansion: bool = False
    reranking_model: str = "cross_encoder"  # Options: "cross_encoder", "bm25", "hybrid"
    relevance_weights: Dict[str, float] = None  # Weights for different relevance factors
    
    def __post_init__(self):
        """Set default relevance weights if not provided."""
        if self.relevance_weights is None:
            self.relevance_weights = {
                "semantic_similarity": 0.6,
                "content_type_boost": 0.1,
                "recency_boost": 0.1,
                "length_penalty": 0.1,
                "keyword_match": 0.1
            }


class SemanticRetriever:
    """
    Core semantic search functionality for multimodal content.
    
    Features:
    - Vector similarity search with configurable thresholds
    - Query embedding generation for text and image queries
    - Cross-modal search across text, image, and audio content
    - Result filtering by content type, metadata, and confidence
    - Performance optimization with caching and batch operations
    """
    
    def __init__(
        self, 
        vector_store: QdrantVectorStore,
        embedding_generator: UnifiedEmbeddingGenerator,
        config: Optional[SearchConfig] = None
    ):
        """
        Initialize semantic retriever.
        
        Args:
            vector_store: Vector database for similarity search
            embedding_generator: Unified embedding generator
            config: Search configuration (uses defaults if None)
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.config = config or SearchConfig()
        
        # Performance tracking
        self._search_stats = {
            'total_searches': 0,
            'text_searches': 0,
            'image_searches': 0,
            'cross_modal_searches': 0,
            'average_search_time': 0.0,
            'cache_hits': 0,
            'total_results_returned': 0,
            'reranking_operations': 0,
            'average_reranking_time': 0.0,
            'relevance_score_improvements': 0
        }
        
        # Query cache for performance optimization
        self._query_cache: Dict[str, List[RetrievalResult]] = {}
        self._cache_max_size = 1000
        
        logger.info("SemanticRetriever initialized successfully")
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        search_filter: Optional[SearchFilter] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Perform semantic search using text query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            search_filter: Optional filter conditions
            similarity_threshold: Minimum similarity score (overrides config)
            
        Returns:
            List of retrieval results sorted by relevance
            
        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If search operation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(query, top_k, search_filter, similarity_threshold)
            if cache_key in self._query_cache:
                self._search_stats['cache_hits'] += 1
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return self._query_cache[cache_key][:top_k]
            
            # Use provided threshold or config default
            threshold = similarity_threshold or self.config.similarity_threshold
            
            # Prepare filter conditions for vector store
            content_types = search_filter.content_types if search_filter else None
            filter_conditions = self._build_filter_conditions(search_filter)
            
            # Perform vector search
            results = self.vector_store.search(
                query=query,
                top_k=min(top_k, self.config.max_results),
                content_types=content_types,
                similarity_threshold=threshold,
                filter_conditions=filter_conditions
            )
            
            # Apply additional filtering
            if search_filter:
                results = self._apply_search_filter(results, search_filter)
            
            # Apply result deduplication and diversity
            results = self._ensure_result_diversity(results)
            
            # Apply reranking if enabled
            if self.config.enable_reranking and results:
                results = self.rerank_results(results, query)
            
            # Cache results
            self._cache_results(cache_key, results)
            
            # Update statistics
            search_time = time.time() - start_time
            self._update_search_stats('text', search_time, len(results))
            
            logger.info(f"Text search completed in {search_time:.3f}s, returned {len(results)} results")
            return results[:top_k]
            
        except Exception as e:
            error_msg = f"Error in semantic search: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def search_by_image(
        self, 
        image_path: str, 
        top_k: int = 10,
        search_filter: Optional[SearchFilter] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Perform semantic search using image query.
        
        Args:
            image_path: Path to query image
            top_k: Number of results to return
            search_filter: Optional filter conditions
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieval results sorted by relevance
            
        Raises:
            ValueError: If image path is invalid
            RuntimeError: If search operation fails
        """
        if not image_path or not image_path.strip():
            raise ValueError("Image path cannot be empty")
        
        start_time = time.time()
        
        try:
            # Use provided threshold or config default
            threshold = similarity_threshold or self.config.similarity_threshold
            
            # Prepare filter conditions
            content_types = search_filter.content_types if search_filter else None
            
            # Perform image-based vector search
            results = self.vector_store.search_by_image(
                image_path=image_path,
                top_k=min(top_k, self.config.max_results),
                content_types=content_types,
                similarity_threshold=threshold
            )
            
            # Apply additional filtering
            if search_filter:
                results = self._apply_search_filter(results, search_filter)
            
            # Apply result deduplication and diversity
            results = self._ensure_result_diversity(results)
            
            # Apply reranking if enabled
            if self.config.enable_reranking and results:
                results = self.rerank_results(results, f"image_query:{image_path}")
            
            # Update statistics
            search_time = time.time() - start_time
            self._update_search_stats('image', search_time, len(results))
            
            logger.info(f"Image search completed in {search_time:.3f}s, returned {len(results)} results")
            return results[:top_k]
            
        except Exception as e:
            error_msg = f"Error in image search: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def cross_modal_search(
        self, 
        query: str, 
        top_k: int = 10,
        include_images: bool = True,
        include_audio: bool = True,
        search_filter: Optional[SearchFilter] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Perform cross-modal semantic search across all content types.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            include_images: Whether to include image results
            include_audio: Whether to include audio results
            search_filter: Optional filter conditions
            similarity_threshold: Minimum similarity score
            
        Returns:
            Combined list of retrieval results from all modalities
            
        Raises:
            ValueError: If query is empty
            RuntimeError: If search operation fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = time.time()
        
        try:
            # Use provided threshold or config default
            threshold = similarity_threshold or self.config.similarity_threshold
            
            # Perform cross-modal search using vector store
            results = self.vector_store.search_cross_modal(
                query=query,
                top_k=min(top_k, self.config.max_results),
                include_images=include_images,
                similarity_threshold=threshold
            )
            
            # Filter by audio content if not included
            if not include_audio:
                results = [r for r in results if r.content_type != ContentType.AUDIO]
            
            # Apply additional filtering
            if search_filter:
                results = self._apply_search_filter(results, search_filter)
            
            # Apply result deduplication and diversity
            results = self._ensure_result_diversity(results)
            
            # Apply reranking if enabled
            if self.config.enable_reranking and results:
                results = self.rerank_results(results, query)
            
            # Update statistics
            search_time = time.time() - start_time
            self._update_search_stats('cross_modal', search_time, len(results))
            
            logger.info(f"Cross-modal search completed in {search_time:.3f}s, returned {len(results)} results")
            return results[:top_k]
            
        except Exception as e:
            error_msg = f"Error in cross-modal search: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def search_similar_content(
        self, 
        content_chunk: ContentChunk, 
        top_k: int = 10,
        exclude_same_document: bool = True,
        similarity_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """
        Find content similar to a given content chunk.
        
        Args:
            content_chunk: Reference content chunk
            top_k: Number of results to return
            exclude_same_document: Whether to exclude results from same document
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar content chunks
            
        Raises:
            ValueError: If content chunk is invalid
            RuntimeError: If search operation fails
        """
        if not content_chunk or not content_chunk.content:
            raise ValueError("Content chunk cannot be empty")
        
        try:
            # Generate embedding for the content chunk if not present
            if content_chunk.embedding is None:
                content_chunk.embedding = self.embedding_generator.generate_embedding(
                    content_chunk.content, 
                    content_chunk.content_type
                )
            
            # Use the content as query for similarity search
            results = self.search(
                query=content_chunk.content,
                top_k=top_k + (10 if exclude_same_document else 0),  # Get extra to account for filtering
                similarity_threshold=similarity_threshold
            )
            
            # Filter out the same chunk and optionally same document
            filtered_results = []
            for result in results:
                if result.chunk_id == content_chunk.chunk_id:
                    continue
                if exclude_same_document and result.source_location.file_path == content_chunk.source_location.file_path:
                    continue
                filtered_results.append(result)
            
            return filtered_results[:top_k]
            
        except Exception as e:
            error_msg = f"Error finding similar content: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def batch_search(
        self, 
        queries: List[str], 
        top_k: int = 10,
        search_filter: Optional[SearchFilter] = None
    ) -> List[List[RetrievalResult]]:
        """
        Perform batch semantic search for multiple queries efficiently.
        
        Args:
            queries: List of search queries
            top_k: Number of results per query
            search_filter: Optional filter conditions
            
        Returns:
            List of result lists, one for each query
            
        Raises:
            ValueError: If queries list is empty
            RuntimeError: If batch search fails
        """
        if not queries:
            raise ValueError("Queries list cannot be empty")
        
        start_time = time.time()
        
        try:
            results = []
            
            # Process queries individually (could be optimized with batch embedding generation)
            for query in queries:
                if query and query.strip():
                    query_results = self.search(
                        query=query,
                        top_k=top_k,
                        search_filter=search_filter
                    )
                    results.append(query_results)
                else:
                    results.append([])
            
            search_time = time.time() - start_time
            total_results = sum(len(r) for r in results)
            
            logger.info(f"Batch search completed in {search_time:.3f}s for {len(queries)} queries, returned {total_results} total results")
            return results
            
        except Exception as e:
            error_msg = f"Error in batch search: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def rerank_results(
        self, 
        results: List[RetrievalResult], 
        query: str,
        reranking_method: Optional[str] = None
    ) -> List[RetrievalResult]:
        """
        Rerank search results using advanced relevance scoring.
        
        Args:
            results: Initial search results to rerank
            query: Original search query
            reranking_method: Reranking method to use (overrides config)
            
        Returns:
            Reranked list of results with updated relevance scores
        """
        if not results:
            return results
        
        method = reranking_method or self.config.reranking_model
        
        try:
            start_time = time.time()
            
            # Store original relevance scores for comparison
            original_scores = [r.relevance_score for r in results]
            
            # Calculate enhanced relevance scores for each result
            for result in results:
                result.relevance_score = self._calculate_relevance_score(result, query)
            
            # Apply method-specific reranking
            if method == "cross_encoder":
                reranked_results = self._cross_encoder_rerank(results, query)
            elif method == "bm25":
                reranked_results = self._bm25_rerank(results, query)
            elif method == "hybrid":
                reranked_results = self._hybrid_rerank(results, query)
            else:
                # Default: sort by relevance score
                reranked_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
            
            # Update statistics
            reranking_time = time.time() - start_time
            self._update_reranking_stats(reranking_time, original_scores, reranked_results)
            
            logger.debug(f"Reranked {len(results)} results using {method} method in {reranking_time:.3f}s")
            return reranked_results
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}. Returning original results.")
            return results
    
    def _calculate_relevance_score(self, result: RetrievalResult, query: str) -> float:
        """
        Calculate comprehensive relevance score for a search result.
        
        Args:
            result: Search result to score
            query: Original search query
            
        Returns:
            Relevance score between 0 and 1
        """
        weights = self.config.relevance_weights
        
        # Base semantic similarity score
        semantic_score = result.similarity_score * weights["semantic_similarity"]
        
        # Content type boost (prefer certain content types)
        content_type_score = self._calculate_content_type_boost(result) * weights["content_type_boost"]
        
        # Recency boost (prefer newer content if metadata available)
        recency_score = self._calculate_recency_boost(result) * weights["recency_boost"]
        
        # Length penalty (penalize very short or very long content)
        length_score = self._calculate_length_penalty(result) * weights["length_penalty"]
        
        # Keyword match boost (exact keyword matches)
        keyword_score = self._calculate_keyword_match_score(result, query) * weights["keyword_match"]
        
        # Combine all scores
        total_score = (
            semantic_score + 
            content_type_score + 
            recency_score + 
            length_score + 
            keyword_score
        )
        
        # Normalize to [0, 1] range
        return min(1.0, max(0.0, total_score))
    
    def _calculate_content_type_boost(self, result: RetrievalResult) -> float:
        """Calculate content type preference boost."""
        # Boost scores based on content type preferences
        content_type_boosts = {
            ContentType.TEXT: 0.0,      # Baseline
            ContentType.IMAGE: 0.1,     # Slight boost for visual content
            ContentType.AUDIO: 0.05     # Small boost for audio content
        }
        
        return content_type_boosts.get(result.content_type, 0.0)
    
    def _calculate_recency_boost(self, result: RetrievalResult) -> float:
        """Calculate recency boost based on document metadata."""
        # Check if creation date is available in metadata
        creation_date = result.metadata.get('creation_date')
        if not creation_date:
            return 0.0
        
        try:
            # Simple recency calculation (would need proper date parsing in real implementation)
            # For now, return a small boost for documents with date metadata
            return 0.05
        except:
            return 0.0
    
    def _calculate_length_penalty(self, result: RetrievalResult) -> float:
        """Calculate penalty/boost based on content length."""
        content_length = len(result.content)
        
        # Optimal length range (adjust based on use case)
        optimal_min = 50
        optimal_max = 500
        
        if optimal_min <= content_length <= optimal_max:
            return 0.1  # Boost for optimal length
        elif content_length < optimal_min:
            return -0.05  # Small penalty for very short content
        elif content_length > optimal_max * 2:
            return -0.1  # Penalty for very long content
        else:
            return 0.0  # Neutral for moderate length
    
    def _calculate_keyword_match_score(self, result: RetrievalResult, query: str) -> float:
        """Calculate boost for exact keyword matches."""
        if not query or not result.content:
            return 0.0
        
        # Extract keywords from query (simple approach)
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        content_words = set(re.findall(r'\b\w+\b', result.content.lower()))
        
        if not query_words:
            return 0.0
        
        # Calculate exact match ratio
        exact_matches = len(query_words.intersection(content_words))
        match_ratio = exact_matches / len(query_words)
        
        # Return scaled boost
        return match_ratio * 0.2  # Max 0.2 boost for perfect keyword match
    
    def _cross_encoder_rerank(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """
        Rerank using cross-encoder approach (simulated).
        
        In a full implementation, this would use a trained cross-encoder model
        to score query-document pairs directly.
        """
        # For now, implement a sophisticated scoring based on multiple factors
        scored_results = []
        
        for result in results:
            # Simulate cross-encoder scoring with multiple factors
            cross_encoder_score = (
                result.similarity_score * 0.7 +  # Base similarity
                self._calculate_query_document_interaction(result, query) * 0.3  # Interaction features
            )
            
            # Update relevance score with cross-encoder result
            result.relevance_score = (result.relevance_score + cross_encoder_score) / 2
            scored_results.append(result)
        
        return sorted(scored_results, key=lambda x: x.relevance_score, reverse=True)
    
    def _calculate_query_document_interaction(self, result: RetrievalResult, query: str) -> float:
        """Calculate query-document interaction features."""
        # Position of query terms in document
        position_score = self._calculate_position_score(result.content, query)
        
        # Query term frequency in document
        frequency_score = self._calculate_term_frequency_score(result.content, query)
        
        # Document structure features (if available)
        structure_score = self._calculate_structure_score(result)
        
        return (position_score + frequency_score + structure_score) / 3
    
    def _calculate_position_score(self, content: str, query: str) -> float:
        """Calculate score based on position of query terms in content."""
        if not query or not content:
            return 0.0
        
        query_words = re.findall(r'\b\w+\b', query.lower())
        content_lower = content.lower()
        
        position_scores = []
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1:
                # Earlier positions get higher scores
                relative_pos = pos / len(content_lower)
                position_scores.append(1.0 - relative_pos)
        
        return sum(position_scores) / len(query_words) if query_words else 0.0
    
    def _calculate_term_frequency_score(self, content: str, query: str) -> float:
        """Calculate score based on term frequency."""
        if not query or not content:
            return 0.0
        
        query_words = re.findall(r'\b\w+\b', query.lower())
        content_words = re.findall(r'\b\w+\b', content.lower())
        
        if not content_words:
            return 0.0
        
        tf_scores = []
        for word in query_words:
            tf = content_words.count(word) / len(content_words)
            tf_scores.append(tf)
        
        return sum(tf_scores) / len(query_words) if query_words else 0.0
    
    def _calculate_structure_score(self, result: RetrievalResult) -> float:
        """Calculate score based on document structure features."""
        # Check if content appears to be from title, heading, or important section
        content = result.content.strip()
        
        # Simple heuristics for structure importance
        if len(content) < 100 and content.isupper():
            return 0.3  # Likely a title or heading
        elif result.source_location.paragraph_index == 0:
            return 0.2  # First paragraph often important
        elif any(marker in content.lower() for marker in ['summary', 'conclusion', 'abstract']):
            return 0.25  # Important sections
        else:
            return 0.0
    
    def _bm25_rerank(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """
        Rerank using BM25 algorithm.
        
        Implements a simplified version of BM25 scoring for reranking.
        """
        if not results or not query:
            return results
        
        # BM25 parameters
        k1 = 1.2
        b = 0.75
        
        # Calculate average document length
        doc_lengths = [len(result.content.split()) for result in results]
        avg_doc_length = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1
        
        # Calculate document frequencies
        query_terms = query.lower().split()
        doc_frequencies = {}
        
        for term in query_terms:
            df = sum(1 for result in results if term in result.content.lower())
            doc_frequencies[term] = df
        
        # Calculate BM25 scores
        scored_results = []
        for i, result in enumerate(results):
            doc_length = doc_lengths[i]
            bm25_score = 0.0
            
            for term in query_terms:
                # Term frequency in document
                tf = result.content.lower().count(term)
                
                # Document frequency
                df = doc_frequencies[term]
                
                # Inverse document frequency
                idf = math.log((len(results) - df + 0.5) / (df + 0.5)) if df > 0 else 0
                
                # BM25 component
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
                
                bm25_score += idf * (numerator / denominator) if denominator > 0 else 0
            
            # Combine with original relevance score
            result.relevance_score = (result.relevance_score + bm25_score / len(query_terms)) / 2
            scored_results.append(result)
        
        return sorted(scored_results, key=lambda x: x.relevance_score, reverse=True)
    
    def _hybrid_rerank(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """
        Hybrid reranking combining multiple approaches.
        
        Combines cross-encoder and BM25 approaches with weighted averaging.
        """
        # Apply cross-encoder reranking
        cross_encoder_results = self._cross_encoder_rerank(results.copy(), query)
        
        # Apply BM25 reranking
        bm25_results = self._bm25_rerank(results.copy(), query)
        
        # Create mapping for efficient lookup
        cross_encoder_scores = {r.chunk_id: r.relevance_score for r in cross_encoder_results}
        bm25_scores = {r.chunk_id: r.relevance_score for r in bm25_results}
        
        # Combine scores with weighted average
        hybrid_results = []
        for result in results:
            ce_score = cross_encoder_scores.get(result.chunk_id, result.relevance_score)
            bm25_score = bm25_scores.get(result.chunk_id, result.relevance_score)
            
            # Weighted combination (60% cross-encoder, 40% BM25)
            result.relevance_score = 0.6 * ce_score + 0.4 * bm25_score
            hybrid_results.append(result)
        
        return sorted(hybrid_results, key=lambda x: x.relevance_score, reverse=True)
    
    def optimize_result_diversity(
        self, 
        results: List[RetrievalResult], 
        diversity_lambda: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Optimize result diversity using Maximal Marginal Relevance (MMR).
        
        Args:
            results: Search results to diversify
            diversity_lambda: Balance between relevance and diversity (0-1)
            
        Returns:
            Diversified list of results
        """
        if not results or len(results) <= 1:
            return results
        
        # Start with the highest relevance result
        selected = [results[0]]
        remaining = results[1:]
        
        while remaining and len(selected) < len(results):
            mmr_scores = []
            
            for candidate in remaining:
                # Relevance score
                relevance = candidate.relevance_score
                
                # Maximum similarity to already selected results
                max_similarity = 0.0
                for selected_result in selected:
                    similarity = self._calculate_content_similarity(
                        candidate.content, 
                        selected_result.content
                    )
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score: balance relevance and diversity
                mmr_score = diversity_lambda * relevance - (1 - diversity_lambda) * max_similarity
                mmr_scores.append((mmr_score, candidate))
            
            # Select candidate with highest MMR score
            if mmr_scores:
                best_candidate = max(mmr_scores, key=lambda x: x[0])[1]
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return selected

    def _build_filter_conditions(self, search_filter: Optional[SearchFilter]) -> Optional[Dict[str, Any]]:
        """Build filter conditions for vector store query."""
        if not search_filter:
            return None
        
        conditions = {}
        
        if search_filter.document_ids:
            conditions['document_id'] = search_filter.document_ids
        
        if search_filter.file_paths:
            conditions['file_path'] = search_filter.file_paths
        
        if search_filter.metadata_filters:
            conditions.update(search_filter.metadata_filters)
        
        return conditions if conditions else None
    
    def _apply_search_filter(
        self, 
        results: List[RetrievalResult], 
        search_filter: SearchFilter
    ) -> List[RetrievalResult]:
        """Apply additional filtering to search results."""
        filtered_results = []
        
        for result in results:
            # Check confidence threshold
            if result.similarity_score < search_filter.confidence_threshold:
                continue
            
            # Check document ID filter
            if search_filter.document_ids and not any(
                doc_id in result.source_location.file_path for doc_id in search_filter.document_ids
            ):
                continue
            
            # Check file path filter
            if search_filter.file_paths and result.source_location.file_path not in search_filter.file_paths:
                continue
            
            # Check metadata filters
            if search_filter.metadata_filters:
                metadata_match = True
                for key, value in search_filter.metadata_filters.items():
                    if key not in result.metadata or result.metadata[key] != value:
                        metadata_match = False
                        break
                if not metadata_match:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _ensure_result_diversity(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Ensure diversity in search results by removing near-duplicates.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Filtered list with diverse results
        """
        if not results or len(results) <= 1:
            return results
        
        diverse_results = [results[0]]  # Always include the top result
        
        for result in results[1:]:
            # Check if this result is too similar to any already selected result
            is_diverse = True
            
            for selected_result in diverse_results:
                # Check content similarity (simple approach)
                if self._calculate_content_similarity(result.content, selected_result.content) > self.config.diversity_threshold:
                    is_diverse = False
                    break
                
                # Check if from same source location
                if (result.source_location.file_path == selected_result.source_location.file_path and
                    result.source_location.page_number == selected_result.source_location.page_number):
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_results.append(result)
        
        return diverse_results
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate simple content similarity between two text strings.
        
        Args:
            content1: First text content
            content2: Second text content
            
        Returns:
            Similarity score between 0 and 1
        """
        if not content1 or not content2:
            return 0.0
        
        # Simple word-based similarity (could be improved with more sophisticated methods)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_cache_key(
        self, 
        query: str, 
        top_k: int, 
        search_filter: Optional[SearchFilter],
        similarity_threshold: Optional[float]
    ) -> str:
        """Generate cache key for query results."""
        import hashlib
        
        # Create a string representation of the search parameters
        cache_data = f"{query}:{top_k}:{similarity_threshold or self.config.similarity_threshold}"
        
        if search_filter:
            cache_data += f":{search_filter.content_types}:{search_filter.confidence_threshold}"
        
        # Generate hash
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _cache_results(self, cache_key: str, results: List[RetrievalResult]) -> None:
        """Cache search results with size management."""
        # Implement simple LRU-like cache management
        if len(self._query_cache) >= self._cache_max_size:
            # Remove oldest entries (simple approach)
            keys_to_remove = list(self._query_cache.keys())[:self._cache_max_size // 4]
            for key in keys_to_remove:
                del self._query_cache[key]
        
        self._query_cache[cache_key] = results.copy()
    
    def _update_search_stats(self, search_type: str, search_time: float, result_count: int) -> None:
        """Update search performance statistics."""
        self._search_stats['total_searches'] += 1
        self._search_stats[f'{search_type}_searches'] += 1
        self._search_stats['total_results_returned'] += result_count
        
        # Update average search time
        total_time = self._search_stats['average_search_time'] * (self._search_stats['total_searches'] - 1)
        self._search_stats['average_search_time'] = (total_time + search_time) / self._search_stats['total_searches']
    
    def _update_reranking_stats(
        self, 
        reranking_time: float, 
        original_scores: List[float], 
        reranked_results: List[RetrievalResult]
    ) -> None:
        """Update reranking performance statistics."""
        self._search_stats['reranking_operations'] += 1
        
        # Update average reranking time
        total_ops = self._search_stats['reranking_operations']
        total_time = self._search_stats['average_reranking_time'] * (total_ops - 1)
        self._search_stats['average_reranking_time'] = (total_time + reranking_time) / total_ops
        
        # Calculate relevance score improvements
        new_scores = [r.relevance_score for r in reranked_results]
        if len(original_scores) == len(new_scores):
            improvements = sum(1 for old, new in zip(original_scores, new_scores) if new > old)
            self._search_stats['relevance_score_improvements'] += improvements
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive search performance statistics.
        
        Returns:
            Dictionary containing search metrics and performance data
        """
        return {
            **self._search_stats,
            'cache_size': len(self._query_cache),
            'cache_max_size': self._cache_max_size,
            'config': {
                'similarity_threshold': self.config.similarity_threshold,
                'max_results': self.config.max_results,
                'enable_cross_modal': self.config.enable_cross_modal,
                'enable_reranking': self.config.enable_reranking,
                'diversity_threshold': self.config.diversity_threshold,
                'reranking_model': self.config.reranking_model,
                'relevance_weights': self.config.relevance_weights
            }
        }
    
    def clear_cache(self) -> None:
        """Clear the query result cache."""
        self._query_cache.clear()
        logger.info("Search result cache cleared")
    
    def update_config(self, config: SearchConfig) -> None:
        """
        Update search configuration.
        
        Args:
            config: New search configuration
        """
        self.config = config
        # Clear cache since configuration changed
        self.clear_cache()
        logger.info("Search configuration updated")


def create_semantic_retriever_example():
    """
    Example usage of SemanticRetriever with different search modes.
    """
    from ..config import SystemConfig
    from ..embeddings.unified_embedding_generator import UnifiedEmbeddingGenerator
    from .vectordb.qdrant_vector_store import QdrantVectorStore
    
    # Initialize components
    config = SystemConfig()
    embedding_generator = UnifiedEmbeddingGenerator(config.embedding)
    vector_store = QdrantVectorStore(config.storage, config.embedding)
    
    # Create semantic retriever with reranking enabled
    search_config = SearchConfig(
        similarity_threshold=0.6,
        max_results=50,
        enable_cross_modal=True,
        enable_reranking=True,
        diversity_threshold=0.8,
        reranking_model="hybrid"
    )
    
    retriever = SemanticRetriever(
        vector_store=vector_store,
        embedding_generator=embedding_generator,
        config=search_config
    )
    
    # Example searches
    print("🔎 Semantic Search Examples:")
    
    # Text search
    print("\n1. Text Search:")
    results = retriever.search("machine learning algorithms", top_k=5)
    for i, result in enumerate(results):
        print(f"   {i+1}. {result.content[:100]}... (score: {result.similarity_score:.3f})")
    
    # Cross-modal search
    print("\n2. Cross-modal Search:")
    cross_results = retriever.cross_modal_search("data visualization", top_k=5)
    for i, result in enumerate(cross_results):
        print(f"   {i+1}. [{result.content_type.value}] {result.content[:80]}... (score: {result.similarity_score:.3f})")
    
    # Filtered search
    print("\n3. Filtered Search:")
    search_filter = SearchFilter(
        content_types=[ContentType.TEXT],
        confidence_threshold=0.7
    )
    filtered_results = retriever.search("artificial intelligence", top_k=3, search_filter=search_filter)
    for i, result in enumerate(filtered_results):
        print(f"   {i+1}. {result.content[:100]}... (score: {result.similarity_score:.3f})")
    
    # Batch search
    print("\n4. Batch Search:")
    queries = ["neural networks", "computer vision", "natural language processing"]
    batch_results = retriever.batch_search(queries, top_k=2)
    for i, (query, results) in enumerate(zip(queries, batch_results)):
        print(f"   Query {i+1}: {query}")
        for j, result in enumerate(results):
            print(f"      {j+1}. {result.content[:80]}... (score: {result.similarity_score:.3f})")
    
    # Test reranking functionality
    print("\n5. Reranking Example:")
    sample_results = [
        RetrievalResult(
            chunk_id="chunk_1",
            content="Machine learning algorithms for pattern recognition and data analysis",
            similarity_score=0.8,
            source_location=SourceLocation(file_path="ml_guide.pdf"),
            content_type=ContentType.TEXT,
            metadata={"creation_date": "2023-01-01"}
        ),
        RetrievalResult(
            chunk_id="chunk_2", 
            content="ML techniques",
            similarity_score=0.9,
            source_location=SourceLocation(file_path="short_doc.pdf"),
            content_type=ContentType.TEXT,
            metadata={}
        )
    ]
    
    reranked = retriever.rerank_results(sample_results, "machine learning algorithms")
    print("   Reranked results:")
    for i, result in enumerate(reranked):
        print(f"      {i+1}. {result.content[:50]}... (relevance: {result.relevance_score:.3f})")
    
    # Get statistics
    stats = retriever.get_search_statistics()
    print(f"\n📊 Search Statistics:")
    print(f"   - Total searches: {stats['total_searches']}")
    print(f"   - Average search time: {stats['average_search_time']:.3f}s")
    print(f"   - Cache hits: {stats['cache_hits']}")
    print(f"   - Total results returned: {stats['total_results_returned']}")
    print(f"   - Reranking operations: {stats['reranking_operations']}")
    print(f"   - Average reranking time: {stats['average_reranking_time']:.3f}s")
    print(f"   - Relevance improvements: {stats['relevance_score_improvements']}")


if __name__ == "__main__":
    create_semantic_retriever_example()