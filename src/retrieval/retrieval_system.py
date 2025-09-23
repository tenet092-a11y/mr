"""
Unified retrieval system for multimodal content.

Integrates vector storage, embedding generation, and document processing
to provide fast, accurate retrieval of text, image, and audio content.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import time

from ..models import (
    ContentChunk, ContentType, RetrievalResult, 
    DocumentContent, ProcessingResult, GroundedResponse, Citation
)
from ..config import SystemConfig
from ..processors.simple_router import SimpleDocumentRouter
from ..embeddings.unified_embedding_generator import UnifiedEmbeddingGenerator
from .vectordb import QdrantVectorStore
from ..processors.pdf_processor import PDFProcessor
from ..processors.docx_processor import DOCXProcessor
from ..processors.image_processor import ImageProcessor
from ..processors.audio_processor import AudioProcessor
from ..llm.ollama_client import OllamaClient
from ..llm.hybrid_llm_manager import HybridLLMManager
from ..llm.citation_generator import CitationGenerator, CitationStyle
from ..llm.response_generator import ResponseGenerator, ResponseConfig


logger = logging.getLogger(__name__)


class MultimodalRetrievalSystem:
    """
    Unified retrieval system for multimodal content.
    
    Features:
    - Document processing and indexing
    - Text and image embedding generation
    - Fast vector similarity search
    - Cross-modal retrieval
    - Response generation with citations
    """
    
    def __init__(self, config: SystemConfig):
        """
        Initialize the multimodal retrieval system.
        
        Args:
            config: System configuration
        """
        self.config = config
        
        # Initialize components
        self.document_router = SimpleDocumentRouter(config.processing)
        self.embedding_generator = UnifiedEmbeddingGenerator(config.embedding)
        
        # Register document processors
        self.document_router.register_processor(PDFProcessor, ['pdf'])
        self.document_router.register_processor(DOCXProcessor, ['docx'])
        self.document_router.register_processor(ImageProcessor, ['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'])
        self.document_router.register_processor(AudioProcessor, ['mp3', 'wav', 'm4a', 'flac', 'ogg', 'wma', 'aac'])
        
        self.vector_store = QdrantVectorStore(config.storage, config.embedding)
        
        # Initialize LLM components
        self.ollama_client = OllamaClient(
            base_url=config.llm.ollama_base_url,
            timeout=300
        )
        self.llm_manager = HybridLLMManager(self.ollama_client)
        
        # Initialize citation generator
        citation_style = CitationStyle(
            format_type=config.llm.citation_style,
            include_page_numbers=config.llm.include_page_numbers,
            include_timestamps=config.llm.include_timestamps,
            max_excerpt_length=config.llm.max_excerpt_length,
            show_content_type=True
        )
        self.citation_generator = CitationGenerator(citation_style)
        
        # Initialize response generator
        self.response_generator = ResponseGenerator(
            llm_manager=self.llm_manager,
            citation_generator=self.citation_generator
        )
        
        # Load embedding model
        self.embedding_generator.load_model()
        
        logger.info("MultimodalRetrievalSystem initialized successfully")
    
    def add_document(self, file_path: str, document_id: Optional[str] = None) -> ProcessingResult:
        """
        Process and add a document to the retrieval system.
        
        Args:
            file_path: Path to the document file
            document_id: Optional document ID (generated if not provided)
            
        Returns:
            ProcessingResult with success status and details
        """
        try:
            # Route document to appropriate processor
            processor = self.document_router.route_document(file_path)
            
            # Process document
            processing_result = processor.process_document(file_path, document_id)
            
            if processing_result.success and processing_result.document_content:
                # Add chunks to vector store
                self.vector_store.add_chunks(processing_result.document_content.chunks)
                
                logger.info(
                    f"Successfully added document {processing_result.document_content.document_id} "
                    f"with {processing_result.chunks_created} chunks"
                )
            
            return processing_result
            
        except Exception as e:
            error_msg = f"Error adding document {file_path}: {str(e)}"
            logger.error(error_msg)
            return ProcessingResult(
                success=False,
                error_message=error_msg
            )
    
    def add_documents_batch(self, file_paths: List[str]) -> Dict[str, ProcessingResult]:
        """
        Process and add multiple documents in batch.
        
        Args:
            file_paths: List of document file paths
            
        Returns:
            Dictionary mapping file paths to processing results
        """
        results = {}
        
        for file_path in file_paths:
            try:
                results[file_path] = self.add_document(file_path)
            except Exception as e:
                results[file_path] = ProcessingResult(
                    success=False,
                    error_message=f"Batch processing error: {str(e)}"
                )
        
        # Log summary
        successful = sum(1 for r in results.values() if r.success)
        logger.info(f"Batch processing completed: {successful}/{len(file_paths)} documents successful")
        
        return results
    
    def search(
        self, 
        query: str, 
        top_k: int = 10,
        content_types: Optional[List[ContentType]] = None,
        similarity_threshold: float = 0.5,
        include_images: bool = True
    ) -> List[RetrievalResult]:
        """
        Search for relevant content using text query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            content_types: Filter by content types (None for all)
            similarity_threshold: Minimum similarity score
            include_images: Whether to include image results in cross-modal search
            
        Returns:
            List of retrieval results
        """
        try:
            start_time = time.time()
            
            if include_images and (content_types is None or ContentType.IMAGE in content_types):
                # Cross-modal search
                results = self.vector_store.search_cross_modal(
                    query=query,
                    top_k=top_k,
                    include_images=include_images,
                    similarity_threshold=similarity_threshold
                )
            else:
                # Text-only search
                results = self.vector_store.search(
                    query=query,
                    top_k=top_k,
                    content_types=content_types,
                    similarity_threshold=similarity_threshold
                )
            
            search_time = time.time() - start_time
            logger.info(f"Search completed in {search_time:.3f}s, returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in search: {e}")
            raise
    
    def search_by_image(
        self, 
        image_path: str, 
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[RetrievalResult]:
        """
        Search for relevant content using image similarity.
        
        Args:
            image_path: Path to query image
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of retrieval results
        """
        try:
            start_time = time.time()
            
            results = self.vector_store.search_by_image(
                image_path=image_path,
                top_k=top_k,
                similarity_threshold=similarity_threshold
            )
            
            search_time = time.time() - start_time
            logger.info(f"Image search completed in {search_time:.3f}s, returned {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in image search: {e}")
            raise
    
    def generate_response(
        self, 
        query: str, 
        context_results: List[RetrievalResult],
        max_length: int = 500,
        temperature: float = 0.7,
        enable_citations: bool = True
    ) -> GroundedResponse:
        """
        Generate a grounded response with citations using hybrid LLM models.
        
        Args:
            query: User query
            context_results: Retrieved context for response generation
            max_length: Maximum response length
            temperature: LLM temperature
            enable_citations: Whether to generate citations
            
        Returns:
            GroundedResponse with citations
        """
        try:
            # Check if Ollama is available
            if not self.ollama_client.is_available():
                logger.warning("Ollama server not available, using fallback response")
                return self._generate_fallback_response(query, context_results)
            
            # Configure response generation
            response_config = ResponseConfig(
                max_length=max_length,
                temperature=temperature,
                enable_citations=enable_citations,
                min_citation_confidence=self.config.llm.min_citation_confidence,
                validate_citations=self.config.llm.validate_citations,
                include_confidence_scores=self.config.llm.include_confidence_scores
            )
            
            # Generate response using hybrid LLM system
            grounded_response = self.response_generator.generate_response(
                query=query,
                context_results=context_results,
                config=response_config
            )
            
            return grounded_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Return fallback response on error
            return self._generate_fallback_response(query, context_results)
    
    def _generate_fallback_response(
        self, 
        query: str, 
        context_results: List[RetrievalResult]
    ) -> GroundedResponse:
        """Generate fallback response when LLM is unavailable."""
        if not context_results:
            return GroundedResponse(
                response_text="I couldn't find relevant information to answer your question.",
                citations=[],
                confidence_score=0.0,
                retrieval_results=[],
                query=query,
                generation_metadata={"fallback": True, "reason": "no_context"}
            )
        
        # Create citations from retrieval results
        citations = []
        for i, result in enumerate(context_results[:5]):  # Limit to top 5
            citation = Citation(
                citation_id=i + 1,
                source_file=result.source_location.file_path,
                location=result.source_location,
                excerpt=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                relevance_score=result.relevance_score,
                content_type=result.content_type
            )
            citations.append(citation)
        
        # Generate simple response
        response_text = self._generate_simple_response(query, context_results)
        
        confidence_score = min(1.0, sum(r.relevance_score for r in context_results[:3]) / 3)
        
        return GroundedResponse(
            response_text=response_text,
            citations=citations,
            confidence_score=confidence_score,
            retrieval_results=context_results,
            query=query,
            generation_metadata={
                "fallback": True,
                "reason": "llm_unavailable",
                "model": "simple_retrieval"
            }
        )
    
    def _generate_simple_response(self, query: str, results: List[RetrievalResult]) -> str:
        """Generate a simple response based on retrieved results."""
        if not results:
            return "I couldn't find relevant information to answer your question."
        
        # Combine top results
        context = "\n\n".join([result.content for result in results[:3]])
        
        # Simple template response
        response = f"Based on the available information:\n\n{context}\n\n"
        
        if len(results) > 1:
            response += f"[Sources: {len(results)} relevant documents found]"
        
        return response
    
    def query_with_response(
        self, 
        query: str, 
        top_k: int = 5,
        include_images: bool = True,
        generate_response: bool = True
    ) -> Dict[str, Any]:
        """
        Complete query pipeline: search + optional response generation.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            include_images: Whether to include image results
            generate_response: Whether to generate a response
            
        Returns:
            Dictionary with search results and optional response
        """
        try:
            # Search for relevant content
            search_results = self.search(
                query=query,
                top_k=top_k,
                include_images=include_images
            )
            
            result = {
                "query": query,
                "search_results": [
                    {
                        "content": r.content,
                        "similarity_score": r.similarity_score,
                        "content_type": r.content_type.value,
                        "source_file": r.source_location.file_path,
                        "metadata": r.metadata
                    }
                    for r in search_results
                ],
                "total_results": len(search_results)
            }
            
            # Generate response if requested
            if generate_response and search_results:
                response = self.generate_response(query, search_results)
                result["response"] = {
                    "text": response.response_text,
                    "confidence_score": response.confidence_score,
                    "citations": [
                        {
                            "id": c.citation_id,
                            "source": c.source_file,
                            "excerpt": c.excerpt,
                            "relevance_score": c.relevance_score
                        }
                        for c in response.citations
                    ]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in query pipeline: {e}")
            raise
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            vector_stats = self.vector_store.get_statistics()
            embedding_stats = self.embedding_generator.get_embedding_stats()
            
            # Get LLM statistics if available
            llm_stats = {}
            try:
                if self.ollama_client.is_available():
                    llm_stats = {
                        'ollama_available': True,
                        'generation_stats': self.ollama_client.get_generation_stats(),
                        'model_status': self.llm_manager.get_model_status(),
                        'response_stats': self.response_generator.get_generation_statistics()
                    }
                else:
                    llm_stats = {'ollama_available': False}
            except Exception as e:
                llm_stats = {'ollama_available': False, 'error': str(e)}
            
            return {
                "vector_store": vector_stats,
                "embedding_generator": embedding_stats,
                "llm_system": llm_stats,
                "supported_formats": self.document_router.get_supported_formats(),
                "configuration": {
                    "chunk_size": self.config.processing.chunk_size,
                    "embedding_dimension": self.config.embedding.embedding_dimension,
                    "text_model": self.config.embedding.text_model_name,
                    "image_model": self.config.embedding.image_model_name,
                    "primary_llm": self.config.llm.primary_model,
                    "secondary_llm": self.config.llm.secondary_model,
                    "citations_enabled": self.config.llm.enable_citations
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document and all its chunks from the system.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            self.vector_store.delete_document(document_id)
            logger.info(f"Successfully deleted document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
    
    def clear_all_data(self) -> None:
        """Clear all data from the retrieval system."""
        try:
            self.vector_store.clear_collection()
            logger.info("Cleared all data from retrieval system")
            
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            raise


def create_retrieval_system_example():
    """
    Example usage of MultimodalRetrievalSystem.
    """
    from ..config import SystemConfig
    
    # Initialize system
    config = SystemConfig()
    retrieval_system = MultimodalRetrievalSystem(config)
    
    # Example documents (you would add real files here)
    example_docs = [
        "FinPilot is an AI-powered financial assistant for tax optimization and investment advice.",
        "MindMate-AI helps professionals manage stress using evidence-based CBT techniques.",
        "Qdrant is an open-source vector search engine optimized for embeddings and ML applications."
    ]
    
    # Create temporary files for demonstration
    import tempfile
    import os
    
    temp_files = []
    try:
        for i, content in enumerate(example_docs):
            temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
            temp_file.write(content)
            temp_file.close()
            temp_files.append(temp_file.name)
            
            # Add document to system
            result = retrieval_system.add_document(temp_file.name)
            if result.success:
                print(f"✅ Added document: {content[:50]}...")
            else:
                print(f"❌ Failed to add document: {result.error_message}")
        
        # Test search functionality
        print("\n🔎 Testing search functionality:")
        
        queries = [
            "Which project helps with mental health?",
            "What is the financial assistant called?",
            "Tell me about vector search engines"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            results = retrieval_system.query_with_response(query, top_k=2)
            
            for i, result in enumerate(results["search_results"]):
                print(f"  {i+1}. {result['content'][:100]}... (score: {result['similarity_score']:.3f})")
            
            if "response" in results:
                print(f"  Response: {results['response']['text'][:200]}...")
        
        # Get system statistics
        print(f"\n📊 System Statistics:")
        stats = retrieval_system.get_system_stats()
        print(f"  - Total documents: {stats.get('vector_store', {}).get('points_count', 0)}")
        print(f"  - Vector dimension: {stats.get('vector_store', {}).get('vector_size', 0)}")
        print(f"  - Embeddings generated: {stats.get('embedding_generator', {}).get('total_embeddings', 0)}")
    
    finally:
        # Clean up temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass


if __name__ == "__main__":
    create_retrieval_system_example()
