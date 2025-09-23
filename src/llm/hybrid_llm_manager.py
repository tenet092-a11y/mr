"""
Hybrid LLM Manager for Phi3:mini and Llama3.2b models.

Manages multiple Ollama models with intelligent routing based on query type,
complexity, and performance requirements. Provides fallback mechanisms
and load balancing for optimal response generation.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import time
import re

from .ollama_client import OllamaClient, GenerationConfig
from ..models import RetrievalResult, ContentType

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels for model selection."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class TaskType(Enum):
    """Types of tasks for model specialization."""
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    CITATION_GENERATION = "citation_generation"
    ANALYSIS = "analysis"
    CREATIVE = "creative"
    FACTUAL = "factual"


@dataclass
class ModelCapabilities:
    """Capabilities and characteristics of a model."""
    name: str
    max_context_length: int
    strengths: List[TaskType]
    speed_score: float  # 1.0 = fastest, higher = slower
    quality_score: float  # 1.0 = highest quality
    memory_usage: str  # "low", "medium", "high"
    preferred_complexity: List[QueryComplexity]


@dataclass
class ModelResponse:
    """Response from a model with metadata."""
    text: str
    model_used: str
    response_time: float
    confidence_score: float
    token_count: int
    task_type: TaskType


class HybridLLMManager:
    """
    Manages hybrid Phi3:mini and Llama3.2b models for optimal performance.
    
    Features:
    - Intelligent model routing based on query characteristics
    - Fallback mechanisms for reliability
    - Performance optimization and load balancing
    - Task-specific model specialization
    - Context length management
    """
    
    def __init__(self, ollama_client: OllamaClient):
        """
        Initialize hybrid LLM manager.
        
        Args:
            ollama_client: Ollama client instance
        """
        self.ollama_client = ollama_client
        
        # Define model capabilities
        self.models = {
            "phi3:mini": ModelCapabilities(
                name="phi3:mini",
                max_context_length=4096,
                strengths=[TaskType.QUESTION_ANSWERING, TaskType.FACTUAL, TaskType.SUMMARIZATION],
                speed_score=1.0,  # Fastest
                quality_score=0.8,
                memory_usage="low",
                preferred_complexity=[QueryComplexity.SIMPLE, QueryComplexity.MODERATE]
            ),
            "llama3.2:3b": ModelCapabilities(
                name="llama3.2:3b", 
                max_context_length=8192,
                strengths=[TaskType.ANALYSIS, TaskType.CITATION_GENERATION, TaskType.CREATIVE],
                speed_score=1.5,  # Slower but higher quality
                quality_score=1.0,
                memory_usage="medium",
                preferred_complexity=[QueryComplexity.MODERATE, QueryComplexity.COMPLEX]
            )
        }
        
        # Performance tracking
        self._model_performance = {
            model_name: {
                'total_requests': 0,
                'successful_requests': 0,
                'average_response_time': 0.0,
                'average_quality_score': 0.0,
                'task_performance': {}
            }
            for model_name in self.models.keys()
        }
        
        # Model availability status
        self._model_status = {}
        self._check_model_availability()
        
        logger.info("HybridLLMManager initialized with models: " + ", ".join(self.models.keys()))
    
    def _check_model_availability(self) -> None:
        """Check which models are available."""
        try:
            available_models = self.ollama_client.list_models()
            available_names = {model.name for model in available_models}
            
            for model_name in self.models.keys():
                # Check for exact match or partial match (e.g., "phi3:mini" matches "phi3:mini")
                is_available = any(
                    model_name in available_name or available_name.startswith(model_name.split(':')[0])
                    for available_name in available_names
                )
                self._model_status[model_name] = is_available
                
                if not is_available:
                    logger.warning(f"Model {model_name} not available. Attempting to pull...")
                    if self.ollama_client.pull_model(model_name):
                        self._model_status[model_name] = True
                        logger.info(f"Successfully pulled model {model_name}")
                    else:
                        logger.error(f"Failed to pull model {model_name}")
            
            available_count = sum(self._model_status.values())
            logger.info(f"Available models: {available_count}/{len(self.models)}")
            
        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            # Assume all models are available as fallback
            for model_name in self.models.keys():
                self._model_status[model_name] = True
    
    def analyze_query(self, query: str, context_results: List[RetrievalResult]) -> Tuple[QueryComplexity, TaskType]:
        """
        Analyze query to determine complexity and task type.
        
        Args:
            query: User query
            context_results: Retrieved context for the query
            
        Returns:
            Tuple of (complexity, task_type)
        """
        # Analyze query complexity
        complexity = self._determine_query_complexity(query, context_results)
        
        # Determine task type
        task_type = self._determine_task_type(query)
        
        logger.debug(f"Query analysis: complexity={complexity.value}, task={task_type.value}")
        return complexity, task_type
    
    def _determine_query_complexity(self, query: str, context_results: List[RetrievalResult]) -> QueryComplexity:
        """Determine query complexity based on various factors."""
        complexity_score = 0
        
        # Query length factor
        if len(query) > 200:
            complexity_score += 2
        elif len(query) > 100:
            complexity_score += 1
        
        # Question complexity indicators
        complex_indicators = [
            'analyze', 'compare', 'contrast', 'evaluate', 'synthesize',
            'relationship', 'implications', 'consequences', 'why', 'how',
            'multiple', 'various', 'different', 'complex', 'detailed'
        ]
        
        simple_indicators = [
            'what is', 'who is', 'when', 'where', 'define', 'list',
            'name', 'identify', 'find', 'show', 'tell me'
        ]
        
        query_lower = query.lower()
        
        # Check for complex indicators
        complex_matches = sum(1 for indicator in complex_indicators if indicator in query_lower)
        complexity_score += complex_matches
        
        # Check for simple indicators (reduces complexity)
        simple_matches = sum(1 for indicator in simple_indicators if indicator in query_lower)
        complexity_score -= simple_matches
        
        # Context complexity factor
        if len(context_results) > 5:
            complexity_score += 1
        
        # Multi-modal content increases complexity
        content_types = {result.content_type for result in context_results}
        if len(content_types) > 1:
            complexity_score += 1
        
        # Determine final complexity
        if complexity_score <= 0:
            return QueryComplexity.SIMPLE
        elif complexity_score <= 3:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX
    
    def _determine_task_type(self, query: str) -> TaskType:
        """Determine the primary task type from the query."""
        query_lower = query.lower()
        
        # Task type indicators
        task_patterns = {
            TaskType.SUMMARIZATION: ['summarize', 'summary', 'overview', 'brief', 'main points'],
            TaskType.QUESTION_ANSWERING: ['what', 'who', 'when', 'where', 'which', 'how', 'why'],
            TaskType.CITATION_GENERATION: ['source', 'reference', 'cite', 'citation', 'evidence'],
            TaskType.ANALYSIS: ['analyze', 'analysis', 'examine', 'evaluate', 'assess', 'compare'],
            TaskType.CREATIVE: ['create', 'generate', 'write', 'compose', 'design', 'imagine'],
            TaskType.FACTUAL: ['fact', 'information', 'data', 'statistics', 'definition', 'explain']
        }
        
        # Score each task type
        task_scores = {}
        for task_type, patterns in task_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            if score > 0:
                task_scores[task_type] = score
        
        # Return highest scoring task type, default to QUESTION_ANSWERING
        if task_scores:
            return max(task_scores.items(), key=lambda x: x[1])[0]
        else:
            return TaskType.QUESTION_ANSWERING
    
    def select_optimal_model(
        self, 
        complexity: QueryComplexity, 
        task_type: TaskType,
        context_length: int = 0
    ) -> str:
        """
        Select the optimal model based on query characteristics.
        
        Args:
            complexity: Query complexity level
            task_type: Type of task to perform
            context_length: Estimated context length needed
            
        Returns:
            Selected model name
        """
        model_scores = {}
        
        for model_name, capabilities in self.models.items():
            # Skip unavailable models
            if not self._model_status.get(model_name, False):
                continue
            
            score = 0
            
            # Task specialization score
            if task_type in capabilities.strengths:
                score += 3
            
            # Complexity preference score
            if complexity in capabilities.preferred_complexity:
                score += 2
            
            # Context length compatibility
            if context_length <= capabilities.max_context_length:
                score += 1
            else:
                score -= 2  # Penalize if context too long
            
            # Performance factors
            # Prefer faster models for simple tasks
            if complexity == QueryComplexity.SIMPLE:
                score += (2.0 - capabilities.speed_score)
            else:
                # Prefer higher quality for complex tasks
                score += capabilities.quality_score
            
            # Historical performance
            model_perf = self._model_performance[model_name]
            if model_perf['total_requests'] > 0:
                success_rate = model_perf['successful_requests'] / model_perf['total_requests']
                score += success_rate
                
                # Task-specific performance
                if task_type.value in model_perf['task_performance']:
                    task_perf = model_perf['task_performance'][task_type.value]
                    score += task_perf.get('success_rate', 0) * 0.5
            
            model_scores[model_name] = score
        
        # Select model with highest score
        if model_scores:
            selected_model = max(model_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Selected model: {selected_model} (score: {model_scores[selected_model]:.2f})")
            return selected_model
        else:
            # Fallback to first available model
            for model_name in self.models.keys():
                if self._model_status.get(model_name, False):
                    logger.warning(f"Using fallback model: {model_name}")
                    return model_name
            
            raise RuntimeError("No models available")
    
    def generate_response(
        self, 
        query: str, 
        context_results: List[RetrievalResult],
        max_length: int = 512,
        temperature: float = 0.7,
        enable_fallback: bool = True
    ) -> ModelResponse:
        """
        Generate response using optimal model selection.
        
        Args:
            query: User query
            context_results: Retrieved context
            max_length: Maximum response length
            temperature: Generation temperature
            enable_fallback: Whether to try fallback models on failure
            
        Returns:
            ModelResponse with generated text and metadata
            
        Raises:
            RuntimeError: If all models fail
        """
        # Analyze query
        complexity, task_type = self.analyze_query(query, context_results)
        
        # Estimate context length
        context_text = "\n".join([result.content for result in context_results])
        context_length = len(query) + len(context_text) + max_length
        
        # Select optimal model
        primary_model = self.select_optimal_model(complexity, task_type, context_length)
        
        # Prepare models to try (primary + fallbacks)
        models_to_try = [primary_model]
        if enable_fallback:
            for model_name in self.models.keys():
                if model_name != primary_model and self._model_status.get(model_name, False):
                    models_to_try.append(model_name)
        
        # Try models in order
        last_error = None
        for model_name in models_to_try:
            try:
                logger.info(f"Attempting generation with model: {model_name}")
                
                # Generate response
                response = self._generate_with_model(
                    model_name, query, context_results, task_type, max_length, temperature
                )
                
                # Update performance tracking
                self._update_model_performance(model_name, task_type, response.response_time, True)
                
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model_name} failed: {e}")
                self._update_model_performance(model_name, task_type, 0, False)
                continue
        
        # All models failed
        error_msg = f"All models failed. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    def _generate_with_model(
        self,
        model_name: str,
        query: str,
        context_results: List[RetrievalResult],
        task_type: TaskType,
        max_length: int,
        temperature: float
    ) -> ModelResponse:
        """Generate response with specific model."""
        start_time = time.time()
        
        # Build prompt based on task type
        prompt = self._build_prompt(query, context_results, task_type)
        
        # Configure generation
        config = GenerationConfig(
            temperature=temperature,
            max_tokens=max_length,
            top_p=0.9,
            top_k=40,
            stop_sequences=["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"]
        )
        
        # Generate response
        generated_text = self.ollama_client.generate(model_name, prompt, config)
        
        response_time = time.time() - start_time
        
        # Calculate confidence score based on response quality
        confidence_score = self._calculate_confidence_score(
            generated_text, query, context_results
        )
        
        return ModelResponse(
            text=generated_text.strip(),
            model_used=model_name,
            response_time=response_time,
            confidence_score=confidence_score,
            token_count=len(generated_text.split()),
            task_type=task_type
        )
    
    def _build_prompt(
        self, 
        query: str, 
        context_results: List[RetrievalResult], 
        task_type: TaskType
    ) -> str:
        """Build task-specific prompt."""
        # Build context section
        context_sections = []
        for i, result in enumerate(context_results[:5], 1):  # Limit to top 5 results
            source_info = f"Source: {result.source_location.file_path}"
            if result.source_location.page_number:
                source_info += f", Page {result.source_location.page_number}"
            if result.source_location.timestamp_start:
                source_info += f", Time {result.source_location.timestamp_start:.1f}s"
            
            context_sections.append(f"[{i}] {source_info}\n{result.content}")
        
        context_text = "\n\n".join(context_sections)
        
        # Task-specific prompt templates
        if task_type == TaskType.CITATION_GENERATION:
            prompt_template = """You are an expert research assistant. Based on the provided context, answer the question and include precise citations using [1], [2], etc. format.

Context:
{context}

Question: {query}

Instructions:
- Provide a comprehensive answer based on the context
- Include numbered citations [1], [2], etc. for each source used
- Be precise about which information comes from which source
- If information is not in the context, clearly state that

Answer:"""
        
        elif task_type == TaskType.SUMMARIZATION:
            prompt_template = """You are an expert at creating concise, accurate summaries. Based on the provided context, create a summary that answers the question.

Context:
{context}

Question: {query}

Instructions:
- Create a clear, concise summary
- Include the most important information
- Maintain accuracy to the source material
- Use numbered citations [1], [2], etc. when referencing sources

Summary:"""
        
        elif task_type == TaskType.ANALYSIS:
            prompt_template = """You are an expert analyst. Based on the provided context, provide a detailed analysis that addresses the question.

Context:
{context}

Question: {query}

Instructions:
- Provide thorough analysis based on the context
- Consider multiple perspectives if present
- Support your analysis with evidence from the sources
- Use numbered citations [1], [2], etc. for references
- Be objective and balanced

Analysis:"""
        
        else:  # Default for QUESTION_ANSWERING and others
            prompt_template = """You are a helpful assistant that provides accurate answers based on the given context. Answer the question using only the information provided in the context.

Context:
{context}

Question: {query}

Instructions:
- Answer based only on the provided context
- Include numbered citations [1], [2], etc. for sources
- If the answer is not in the context, say so clearly
- Be accurate and concise

Answer:"""
        
        return prompt_template.format(context=context_text, query=query)
    
    def _calculate_confidence_score(
        self, 
        response: str, 
        query: str, 
        context_results: List[RetrievalResult]
    ) -> float:
        """Calculate confidence score for the response."""
        score = 0.5  # Base score
        
        # Length factor (not too short, not too long)
        if 50 <= len(response) <= 1000:
            score += 0.1
        
        # Citation presence
        citation_pattern = r'\[\d+\]'
        citations = re.findall(citation_pattern, response)
        if citations:
            score += 0.2
            # Bonus for multiple citations
            if len(citations) > 1:
                score += 0.1
        
        # Keyword overlap with query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words.intersection(response_words))
        if overlap > 0:
            score += min(0.2, overlap * 0.05)
        
        # Context utilization
        context_words = set()
        for result in context_results:
            context_words.update(result.content.lower().split())
        
        context_overlap = len(response_words.intersection(context_words))
        if context_overlap > 0:
            score += min(0.2, context_overlap * 0.01)
        
        return min(1.0, score)
    
    def _update_model_performance(
        self, 
        model_name: str, 
        task_type: TaskType, 
        response_time: float, 
        success: bool
    ) -> None:
        """Update model performance statistics."""
        model_perf = self._model_performance[model_name]
        model_perf['total_requests'] += 1
        
        if success:
            model_perf['successful_requests'] += 1
            
            # Update average response time
            total_successful = model_perf['successful_requests']
            current_avg = model_perf['average_response_time']
            model_perf['average_response_time'] = (
                (current_avg * (total_successful - 1) + response_time) / total_successful
            )
        
        # Update task-specific performance
        task_key = task_type.value
        if task_key not in model_perf['task_performance']:
            model_perf['task_performance'][task_key] = {
                'requests': 0,
                'successes': 0,
                'success_rate': 0.0
            }
        
        task_perf = model_perf['task_performance'][task_key]
        task_perf['requests'] += 1
        if success:
            task_perf['successes'] += 1
        task_perf['success_rate'] = task_perf['successes'] / task_perf['requests']
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and performance."""
        return {
            'available_models': {
                name: status for name, status in self._model_status.items()
            },
            'model_capabilities': {
                name: {
                    'max_context_length': caps.max_context_length,
                    'strengths': [s.value for s in caps.strengths],
                    'speed_score': caps.speed_score,
                    'quality_score': caps.quality_score,
                    'memory_usage': caps.memory_usage
                }
                for name, caps in self.models.items()
            },
            'performance_stats': self._model_performance
        }
    
    def refresh_model_status(self) -> None:
        """Refresh model availability status."""
        self._check_model_availability()