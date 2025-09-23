"""
Response generator integrating hybrid LLM models with citation generation.

Combines the hybrid LLM manager and citation generator to produce
comprehensive responses with accurate citations for multimodal RAG queries.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time

from .hybrid_llm_manager import HybridLLMManager, ModelResponse
from .citation_generator import CitationGenerator, CitationStyle
from ..models import RetrievalResult, GroundedResponse, Citation

logger = logging.getLogger(__name__)


@dataclass
class ResponseConfig:
    """Configuration for response generation."""
    max_length: int = 512
    temperature: float = 0.7
    enable_citations: bool = True
    citation_style: Optional[CitationStyle] = None
    min_citation_confidence: float = 0.3
    enable_fallback: bool = True
    validate_citations: bool = True
    include_confidence_scores: bool = True


class ResponseGenerator:
    """
    Comprehensive response generator for multimodal RAG.
    
    Features:
    - Hybrid LLM model selection and management
    - Automatic citation generation and validation
    - Response quality assessment
    - Multimodal content handling
    - Confidence scoring and validation
    """
    
    def __init__(
        self, 
        llm_manager: HybridLLMManager,
        citation_generator: Optional[CitationGenerator] = None
    ):
        """
        Initialize response generator.
        
        Args:
            llm_manager: Hybrid LLM manager instance
            citation_generator: Citation generator instance
        """
        self.llm_manager = llm_manager
        self.citation_generator = citation_generator or CitationGenerator()
        
        # Performance tracking
        self._generation_stats = {
            'total_responses': 0,
            'successful_responses': 0,
            'average_response_time': 0.0,
            'average_confidence': 0.0,
            'citation_stats': {
                'total_citations': 0,
                'average_citations_per_response': 0.0,
                'average_citation_confidence': 0.0
            }
        }
        
        logger.info("ResponseGenerator initialized")
    
    def generate_response(
        self, 
        query: str, 
        context_results: List[RetrievalResult],
        config: Optional[ResponseConfig] = None
    ) -> GroundedResponse:
        """
        Generate comprehensive response with citations.
        
        Args:
            query: User query
            context_results: Retrieved context results
            config: Response generation configuration
            
        Returns:
            GroundedResponse with text, citations, and metadata
            
        Raises:
            RuntimeError: If response generation fails
        """
        if config is None:
            config = ResponseConfig()
        
        start_time = time.time()
        
        try:
            logger.info(f"Generating response for query: {query[:100]}...")
            
            # Generate response using hybrid LLM
            model_response = self.llm_manager.generate_response(
                query=query,
                context_results=context_results,
                max_length=config.max_length,
                temperature=config.temperature,
                enable_fallback=config.enable_fallback
            )
            
            # Generate citations if enabled
            citations = []
            citation_validation = {}
            
            if config.enable_citations:
                citations = self.citation_generator.generate_citations(
                    response_text=model_response.text,
                    context_results=context_results,
                    min_confidence=config.min_citation_confidence
                )
                
                # Validate citations if requested
                if config.validate_citations:
                    citation_validation = self.citation_generator.validate_citations(
                        citations=citations,
                        context_results=context_results
                    )
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                model_response, citations, context_results
            )
            
            # Create grounded response
            grounded_response = GroundedResponse(
                response_text=model_response.text,
                citations=citations,
                confidence_score=overall_confidence,
                retrieval_results=context_results,
                query=query,
                generation_metadata={
                    'model_used': model_response.model_used,
                    'model_response_time': model_response.response_time,
                    'model_confidence': model_response.confidence_score,
                    'task_type': model_response.task_type.value,
                    'token_count': model_response.token_count,
                    'citation_count': len(citations),
                    'citation_validation': citation_validation,
                    'total_generation_time': time.time() - start_time,
                    'config': {
                        'max_length': config.max_length,
                        'temperature': config.temperature,
                        'min_citation_confidence': config.min_citation_confidence
                    }
                }
            )
            
            # Update statistics
            self._update_generation_stats(grounded_response, time.time() - start_time)
            
            logger.info(
                f"Response generated successfully: {len(model_response.text)} chars, "
                f"{len(citations)} citations, confidence: {overall_confidence:.3f}"
            )
            
            return grounded_response
            
        except Exception as e:
            self._generation_stats['total_responses'] += 1
            error_msg = f"Response generation failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def generate_batch_responses(
        self, 
        queries: List[str], 
        context_results_list: List[List[RetrievalResult]],
        config: Optional[ResponseConfig] = None
    ) -> List[GroundedResponse]:
        """
        Generate responses for multiple queries efficiently.
        
        Args:
            queries: List of user queries
            context_results_list: List of context results for each query
            config: Response generation configuration
            
        Returns:
            List of GroundedResponse objects
            
        Raises:
            ValueError: If input lists have different lengths
        """
        if len(queries) != len(context_results_list):
            raise ValueError("Queries and context results lists must have same length")
        
        responses = []
        
        for query, context_results in zip(queries, context_results_list):
            try:
                response = self.generate_response(query, context_results, config)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to generate response for query '{query[:50]}...': {e}")
                # Create error response
                error_response = GroundedResponse(
                    response_text=f"Error generating response: {str(e)}",
                    citations=[],
                    confidence_score=0.0,
                    retrieval_results=context_results,
                    query=query,
                    generation_metadata={'error': str(e)}
                )
                responses.append(error_response)
        
        logger.info(f"Generated {len(responses)} batch responses")
        return responses
    
    def _calculate_overall_confidence(
        self, 
        model_response: ModelResponse, 
        citations: List[Citation],
        context_results: List[RetrievalResult]
    ) -> float:
        """Calculate overall confidence score for the response."""
        # Base confidence from model
        confidence = model_response.confidence_score * 0.4
        
        # Citation quality factor
        if citations:
            avg_citation_confidence = sum(c.relevance_score for c in citations) / len(citations)
            confidence += avg_citation_confidence * 0.3
            
            # Citation coverage bonus
            cited_sources = {c.source_file for c in citations}
            available_sources = {r.source_location.file_path for r in context_results}
            
            if available_sources:
                coverage = len(cited_sources) / len(available_sources)
                confidence += coverage * 0.2
        else:
            # Penalty for no citations when sources are available
            if context_results:
                confidence -= 0.2
        
        # Context relevance factor
        if context_results:
            avg_context_relevance = sum(r.similarity_score for r in context_results) / len(context_results)
            confidence += avg_context_relevance * 0.1
        
        return min(1.0, max(0.0, confidence))
    
    def _update_generation_stats(self, response: GroundedResponse, generation_time: float) -> None:
        """Update generation performance statistics."""
        self._generation_stats['total_responses'] += 1
        self._generation_stats['successful_responses'] += 1
        
        # Update average response time
        total_responses = self._generation_stats['successful_responses']
        current_avg_time = self._generation_stats['average_response_time']
        self._generation_stats['average_response_time'] = (
            (current_avg_time * (total_responses - 1) + generation_time) / total_responses
        )
        
        # Update average confidence
        current_avg_conf = self._generation_stats['average_confidence']
        self._generation_stats['average_confidence'] = (
            (current_avg_conf * (total_responses - 1) + response.confidence_score) / total_responses
        )
        
        # Update citation statistics
        citation_count = len(response.citations)
        citation_stats = self._generation_stats['citation_stats']
        
        citation_stats['total_citations'] += citation_count
        citation_stats['average_citations_per_response'] = (
            citation_stats['total_citations'] / total_responses
        )
        
        if response.citations:
            avg_citation_conf = sum(c.relevance_score for c in response.citations) / len(response.citations)
            current_citation_avg = citation_stats['average_citation_confidence']
            
            # Update weighted average
            total_citations = citation_stats['total_citations']
            if total_citations > 0:
                citation_stats['average_citation_confidence'] = (
                    (current_citation_avg * (total_citations - citation_count) + 
                     avg_citation_conf * citation_count) / total_citations
                )
    
    def enhance_response_with_context(
        self, 
        response: GroundedResponse,
        additional_context: List[RetrievalResult]
    ) -> GroundedResponse:
        """
        Enhance existing response with additional context.
        
        Args:
            response: Original grounded response
            additional_context: Additional context results
            
        Returns:
            Enhanced GroundedResponse
        """
        try:
            # Combine original and additional context
            combined_context = response.retrieval_results + additional_context
            
            # Generate new response with enhanced context
            enhanced_response = self.generate_response(
                query=response.query,
                context_results=combined_context
            )
            
            # Preserve original metadata
            enhanced_response.generation_metadata['enhanced_from_original'] = True
            enhanced_response.generation_metadata['original_confidence'] = response.confidence_score
            enhanced_response.generation_metadata['additional_sources'] = len(additional_context)
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Failed to enhance response: {e}")
            return response  # Return original on failure
    
    def get_response_quality_metrics(self, response: GroundedResponse) -> Dict[str, Any]:
        """
        Analyze response quality and provide metrics.
        
        Args:
            response: Grounded response to analyze
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            'response_length': len(response.response_text),
            'word_count': len(response.response_text.split()),
            'citation_count': len(response.citations),
            'confidence_score': response.confidence_score,
            'source_coverage': 0.0,
            'citation_density': 0.0,
            'quality_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Calculate source coverage
        if response.retrieval_results:
            cited_sources = {c.source_file for c in response.citations}
            available_sources = {r.source_location.file_path for r in response.retrieval_results}
            metrics['source_coverage'] = len(cited_sources) / len(available_sources)
        
        # Calculate citation density
        if metrics['word_count'] > 0:
            metrics['citation_density'] = len(response.citations) / metrics['word_count']
        
        # Calculate overall quality score
        quality_factors = [
            response.confidence_score,
            min(1.0, metrics['response_length'] / 200),  # Length factor
            metrics['source_coverage'],
            min(1.0, metrics['citation_density'] * 20)  # Citation density factor
        ]
        metrics['quality_score'] = sum(quality_factors) / len(quality_factors)
        
        # Identify issues
        if metrics['response_length'] < 50:
            metrics['issues'].append("Response too short")
        elif metrics['response_length'] > 1000:
            metrics['issues'].append("Response may be too long")
        
        if metrics['citation_count'] == 0 and response.retrieval_results:
            metrics['issues'].append("No citations despite available sources")
        
        if metrics['source_coverage'] < 0.3:
            metrics['issues'].append("Low source coverage")
        
        if response.confidence_score < 0.5:
            metrics['issues'].append("Low confidence score")
        
        # Generate recommendations
        if metrics['citation_count'] < 2 and len(response.retrieval_results) > 2:
            metrics['recommendations'].append("Consider adding more citations")
        
        if metrics['source_coverage'] < 0.5:
            metrics['recommendations'].append("Utilize more diverse sources")
        
        if response.confidence_score < 0.7:
            metrics['recommendations'].append("Review source relevance and model selection")
        
        return metrics
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive generation statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        stats = self._generation_stats.copy()
        
        # Add success rate
        if stats['total_responses'] > 0:
            stats['success_rate'] = stats['successful_responses'] / stats['total_responses']
        else:
            stats['success_rate'] = 0.0
        
        # Add LLM manager stats
        stats['llm_manager_stats'] = self.llm_manager.get_model_status()
        
        return stats
    
    def clear_statistics(self) -> None:
        """Clear generation statistics."""
        self._generation_stats = {
            'total_responses': 0,
            'successful_responses': 0,
            'average_response_time': 0.0,
            'average_confidence': 0.0,
            'citation_stats': {
                'total_citations': 0,
                'average_citations_per_response': 0.0,
                'average_citation_confidence': 0.0
            }
        }
        logger.info("Generation statistics cleared")