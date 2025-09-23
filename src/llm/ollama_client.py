"""
Ollama client for offline LLM inference.

Provides interface to Ollama models running locally for completely offline
text generation with support for multiple models and streaming responses.
"""

import json
import logging
import requests
from typing import Dict, List, Optional, Any, Iterator, Union
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class OllamaModelInfo:
    """Information about an Ollama model."""
    name: str
    size: str
    modified_at: str
    digest: str
    details: Dict[str, Any]


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 512
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    seed: Optional[int] = None
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class OllamaClient:
    """
    Client for interacting with Ollama models locally.
    
    Provides offline text generation capabilities with support for:
    - Multiple model management
    - Streaming and non-streaming responses
    - Model switching and optimization
    - Error handling and retry logic
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        """
        Initialize Ollama client.
        
        Args:
            base_url: Ollama server URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        
        # Performance tracking
        self._generation_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_generated': 0,
            'average_response_time': 0.0,
            'model_usage': {}
        }
        
        logger.info(f"OllamaClient initialized with base URL: {base_url}")
    
    def is_available(self) -> bool:
        """
        Check if Ollama server is available.
        
        Returns:
            True if server is reachable, False otherwise
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama server not available: {e}")
            return False
    
    def list_models(self) -> List[OllamaModelInfo]:
        """
        List available models.
        
        Returns:
            List of available model information
            
        Raises:
            RuntimeError: If request fails
        """
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            response.raise_for_status()
            
            data = response.json()
            models = []
            
            for model_data in data.get('models', []):
                model_info = OllamaModelInfo(
                    name=model_data['name'],
                    size=model_data.get('size', 'unknown'),
                    modified_at=model_data.get('modified_at', ''),
                    digest=model_data.get('digest', ''),
                    details=model_data.get('details', {})
                )
                models.append(model_info)
            
            logger.info(f"Found {len(models)} available models")
            return models
            
        except Exception as e:
            error_msg = f"Failed to list models: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def pull_model(self, model_name: str) -> bool:
        """
        Pull/download a model if not available locally.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Pulling model: {model_name}")
            
            response = self.session.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if data.get('status'):
                            logger.info(f"Pull status: {data['status']}")
                        if data.get('error'):
                            logger.error(f"Pull error: {data['error']}")
                            return False
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"Successfully pulled model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    def generate(
        self, 
        model: str, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate text using specified model.
        
        Args:
            model: Model name to use
            prompt: Input prompt
            config: Generation configuration
            
        Returns:
            Generated text response
            
        Raises:
            RuntimeError: If generation fails
        """
        if config is None:
            config = GenerationConfig()
        
        start_time = time.time()
        
        try:
            # Prepare request payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "num_predict": config.max_tokens,
                    "repeat_penalty": config.repeat_penalty,
                }
            }
            
            if config.stop_sequences:
                payload["options"]["stop"] = config.stop_sequences
            
            if config.seed is not None:
                payload["options"]["seed"] = config.seed
            
            # Make request
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            generated_text = result.get('response', '')
            
            # Update statistics
            response_time = time.time() - start_time
            self._update_generation_stats(model, response_time, len(generated_text), True)
            
            logger.info(f"Generated {len(generated_text)} characters using {model} in {response_time:.2f}s")
            return generated_text
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_generation_stats(model, response_time, 0, False)
            
            error_msg = f"Text generation failed with {model}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def generate_stream(
        self, 
        model: str, 
        prompt: str, 
        config: Optional[GenerationConfig] = None
    ) -> Iterator[str]:
        """
        Generate text with streaming response.
        
        Args:
            model: Model name to use
            prompt: Input prompt
            config: Generation configuration
            
        Yields:
            Text chunks as they are generated
            
        Raises:
            RuntimeError: If generation fails
        """
        if config is None:
            config = GenerationConfig()
        
        start_time = time.time()
        total_tokens = 0
        
        try:
            # Prepare request payload
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "num_predict": config.max_tokens,
                    "repeat_penalty": config.repeat_penalty,
                }
            }
            
            if config.stop_sequences:
                payload["options"]["stop"] = config.stop_sequences
            
            # Make streaming request
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
                stream=True
            )
            response.raise_for_status()
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        
                        if 'response' in data:
                            chunk = data['response']
                            total_tokens += len(chunk)
                            yield chunk
                        
                        if data.get('done', False):
                            break
                            
                        if data.get('error'):
                            raise RuntimeError(f"Generation error: {data['error']}")
                            
                    except json.JSONDecodeError:
                        continue
            
            # Update statistics
            response_time = time.time() - start_time
            self._update_generation_stats(model, response_time, total_tokens, True)
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_generation_stats(model, response_time, total_tokens, False)
            
            error_msg = f"Streaming generation failed with {model}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def chat(
        self, 
        model: str, 
        messages: List[Dict[str, str]], 
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate response using chat format.
        
        Args:
            model: Model name to use
            messages: List of chat messages with 'role' and 'content'
            config: Generation configuration
            
        Returns:
            Generated response
            
        Raises:
            RuntimeError: If generation fails
        """
        if config is None:
            config = GenerationConfig()
        
        start_time = time.time()
        
        try:
            # Prepare request payload
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": config.temperature,
                    "top_p": config.top_p,
                    "top_k": config.top_k,
                    "num_predict": config.max_tokens,
                    "repeat_penalty": config.repeat_penalty,
                }
            }
            
            # Make request
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            generated_text = result.get('message', {}).get('content', '')
            
            # Update statistics
            response_time = time.time() - start_time
            self._update_generation_stats(model, response_time, len(generated_text), True)
            
            return generated_text
            
        except Exception as e:
            response_time = time.time() - start_time
            self._update_generation_stats(model, response_time, 0, False)
            
            error_msg = f"Chat generation failed with {model}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _update_generation_stats(
        self, 
        model: str, 
        response_time: float, 
        tokens_generated: int, 
        success: bool
    ) -> None:
        """Update generation performance statistics."""
        self._generation_stats['total_requests'] += 1
        
        if success:
            self._generation_stats['successful_requests'] += 1
            self._generation_stats['total_tokens_generated'] += tokens_generated
        else:
            self._generation_stats['failed_requests'] += 1
        
        # Update average response time
        total_requests = self._generation_stats['total_requests']
        current_avg = self._generation_stats['average_response_time']
        self._generation_stats['average_response_time'] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
        
        # Update model usage
        if model not in self._generation_stats['model_usage']:
            self._generation_stats['model_usage'][model] = {
                'requests': 0,
                'tokens': 0,
                'avg_response_time': 0.0
            }
        
        model_stats = self._generation_stats['model_usage'][model]
        model_stats['requests'] += 1
        model_stats['tokens'] += tokens_generated
        model_stats['avg_response_time'] = (
            (model_stats['avg_response_time'] * (model_stats['requests'] - 1) + response_time) 
            / model_stats['requests']
        )
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        Get generation performance statistics.
        
        Returns:
            Dictionary containing performance metrics
        """
        return self._generation_stats.copy()
    
    def clear_stats(self) -> None:
        """Clear generation statistics."""
        self._generation_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_generated': 0,
            'average_response_time': 0.0,
            'model_usage': {}
        }
        logger.info("Generation statistics cleared")