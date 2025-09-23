#!/usr/bin/env python3
"""
Gradio Web Interface for Multimodal RAG System

A user-friendly web interface for interacting with the multimodal RAG system
including document upload, processing, and intelligent querying with adjustable parameters.
"""

import gradio as gr
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
import traceback

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.config import SystemConfig
    from src.retrieval.retrieval_system import MultimodalRetrievalSystem
    from src.models import ContentType, ProcessingResult
    from src.llm.ollama_client import OllamaClient
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


class GradioRAGInterface:
    """Gradio interface for the Multimodal RAG system."""
    
    def __init__(self):
        """Initialize the Gradio interface."""
        self.config = SystemConfig()
        self.retrieval_system = None
        self.is_initialized = False
        self.processed_documents = {}
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self) -> bool:
        """Initialize the retrieval system."""
        try:
            print("⚙️  Initializing Multimodal RAG System...")
            self.retrieval_system = MultimodalRetrievalSystem(self.config)
            self.is_initialized = True
            print("✅ System initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            return False
    
    def process_uploaded_file(self, file) -> Tuple[str, str]:
        """
        Process an uploaded file through Gradio.
        
        Args:
            file: Gradio file upload object
            
        Returns:
            Tuple of (status_message, document_list)
        """
        if not self.is_initialized:
            return "❌ System not initialized", ""
        
        if file is None:
            return "❌ No file uploaded", ""
        
        try:
            # Get the file path from Gradio file object
            file_path = file.name if hasattr(file, 'name') else str(file)
            
            print(f"📄 Processing uploaded file: {Path(file_path).name}")
            
            # Process the document
            result = self.retrieval_system.add_document(file_path)
            
            if result.success:
                # Store processing info
                self.processed_documents[file_path] = {
                    "document_id": result.document_content.document_id,
                    "content_type": result.document_content.content_type.value,
                    "chunks_created": result.chunks_created,
                    "processing_time": result.processing_time,
                    "file_name": Path(file_path).name
                }
                
                status_msg = f"✅ Successfully processed: {Path(file_path).name}\n"
                status_msg += f"   - Document ID: {result.document_content.document_id}\n"
                status_msg += f"   - Content Type: {result.document_content.content_type.value}\n"
                status_msg += f"   - Chunks Created: {result.chunks_created}\n"
                status_msg += f"   - Processing Time: {result.processing_time:.2f}s"
                
                # Update document list
                doc_list = self._format_document_list()
                
                return status_msg, doc_list
            else:
                return f"❌ Failed to process: {Path(file_path).name}\nError: {result.error_message}", ""
                
        except Exception as e:
            error_msg = f"❌ Error processing file: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return error_msg, ""
    
    def query_system(
        self, 
        query: str, 
        top_k: int = 5, 
        similarity_threshold: float = 0.3,
        include_images: bool = True,
        temperature: float = 0.7
    ) -> Tuple[str, str, str]:
        """
        Query the system with adjustable parameters.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score (lowered default)
            include_images: Whether to include image results
            temperature: LLM temperature for response generation
            
        Returns:
            Tuple of (search_results, response_text, debug_info)
        """
        if not self.is_initialized:
            return "❌ System not initialized", "", ""
        
        if not query or not query.strip():
            return "❌ Please enter a query", "", ""
        
        try:
            print(f"🔍 Processing query: {query}")
            print(f"📊 Parameters: top_k={top_k}, threshold={similarity_threshold}, include_images={include_images}")
            
            start_time = time.time()
            
            # Search for relevant content with custom parameters
            search_results = self.retrieval_system.search(
                query=query,
                top_k=top_k,
                include_images=include_images,
                similarity_threshold=similarity_threshold
            )
            
            search_time = time.time() - start_time
            
            # Format search results
            search_output = f"🔍 Search Results ({len(search_results)} found in {search_time:.2f}s)\n"
            search_output += "=" * 60 + "\n\n"
            
            if search_results:
                for i, result in enumerate(search_results, 1):
                    search_output += f"{i}. [{result.content_type.value.upper()}] Score: {result.similarity_score:.3f}\n"
                    search_output += f"   Source: {Path(result.source_location.file_path).name}\n"
                    search_output += f"   Content: {result.content[:200]}...\n"
                    if result.source_location.page_number:
                        search_output += f"   Page: {result.source_location.page_number}\n"
                    if result.source_location.timestamp_start:
                        search_output += f"   Time: {result.source_location.timestamp_start:.1f}s\n"
                    search_output += "\n"
            else:
                search_output += "❌ No results found. Try:\n"
                search_output += "   - Lowering the similarity threshold\n"
                search_output += "   - Using broader search terms\n"
                search_output += "   - Checking if the document contains relevant content\n"
            
            # Generate response if results found
            response_output = ""
            debug_info = f"Debug Information:\n"
            debug_info += f"- Query: {query}\n"
            debug_info += f"- Search time: {search_time:.3f}s\n"
            debug_info += f"- Results found: {len(search_results)}\n"
            debug_info += f"- Similarity threshold: {similarity_threshold}\n"
            debug_info += f"- Top K: {top_k}\n"
            debug_info += f"- Include images: {include_images}\n"
            
            if search_results:
                print("🤖 Generating response...")
                response_start = time.time()
                
                try:
                    response = self.retrieval_system.generate_response(
                        query=query,
                        context_results=search_results,
                        max_length=500,
                        temperature=temperature,
                        enable_citations=True
                    )
                    
                    response_time = time.time() - response_start
                    
                    response_output = f"🤖 Generated Response:\n"
                    response_output += "=" * 40 + "\n\n"
                    response_output += f"{response.response_text}\n\n"
                    
                    if response.citations:
                        response_output += f"📖 Citations ({len(response.citations)}):\n"
                        for citation in response.citations:
                            source_name = Path(citation.source_file).name
                            location_info = ""
                            if citation.location.page_number:
                                location_info += f", p.{citation.location.page_number}"
                            if citation.location.timestamp_start:
                                location_info += f", {citation.location.timestamp_start:.1f}s"
                            
                            response_output += f"[{citation.citation_id}] {source_name}{location_info}\n"
                            response_output += f"    Confidence: {citation.relevance_score:.3f}\n"
                            response_output += f"    Excerpt: {citation.excerpt[:150]}...\n\n"
                    
                    # Update debug info
                    debug_info += f"- Response generation time: {response_time:.3f}s\n"
                    debug_info += f"- Model used: {response.generation_metadata.get('model_used', 'unknown')}\n"
                    debug_info += f"- Response confidence: {response.confidence_score:.3f}\n"
                    debug_info += f"- Citations generated: {len(response.citations)}\n"
                    
                except Exception as e:
                    response_output = f"❌ Error generating response: {str(e)}\n"
                    response_output += "The search found results, but response generation failed.\n"
                    response_output += "This might be due to Ollama connectivity issues."
                    
                    debug_info += f"- Response generation error: {str(e)}\n"
            else:
                response_output = "❌ No response generated - no relevant content found.\n\n"
                response_output += "💡 Suggestions:\n"
                response_output += "- Try lowering the similarity threshold (e.g., 0.1-0.3)\n"
                response_output += "- Use simpler, more general terms\n"
                response_output += "- Check if your document contains the information you're looking for\n"
                response_output += "- Try different variations of your question\n"
            
            return search_output, response_output, debug_info
            
        except Exception as e:
            error_msg = f"❌ Error querying system: {str(e)}"
            debug_info = f"Error details: {traceback.format_exc()}"
            print(error_msg)
            return error_msg, "", debug_info
    
    def get_system_status(self) -> str:
        """Get comprehensive system status."""
        try:
            if not self.is_initialized:
                return "❌ System not initialized"
            
            status_output = "📊 System Status\n"
            status_output += "=" * 30 + "\n\n"
            
            # Basic status
            status_output += f"✅ System initialized: {self.is_initialized}\n"
            status_output += f"📄 Processed documents: {len(self.processed_documents)}\n\n"
            
            # Get detailed statistics
            stats = self.retrieval_system.get_system_stats()
            
            # Vector store statistics
            vector_stats = stats.get("vector_store", {})
            status_output += "📦 Vector Store:\n"
            status_output += f"   - Total documents: {vector_stats.get('points_count', 0)}\n"
            status_output += f"   - Vector dimension: {vector_stats.get('vector_size', 0)}\n"
            status_output += f"   - Database path: {vector_stats.get('database_path', 'Unknown')}\n\n"
            
            # Content distribution
            content_dist = vector_stats.get('content_type_distribution', {})
            if content_dist:
                status_output += "📊 Content Distribution:\n"
                for content_type, count in content_dist.items():
                    status_output += f"   - {content_type}: {count}\n"
                status_output += "\n"
            
            # Embedding statistics
            embedding_stats = stats.get("embedding_generator", {})
            status_output += "🧠 Embedding Generator:\n"
            status_output += f"   - Total embeddings: {embedding_stats.get('total_embeddings', 0)}\n"
            status_output += f"   - Text embeddings: {embedding_stats.get('text_embeddings', 0)}\n"
            status_output += f"   - Image embeddings: {embedding_stats.get('image_embeddings', 0)}\n"
            status_output += f"   - Audio embeddings: {embedding_stats.get('audio_embeddings', 0)}\n"
            status_output += f"   - Cache hit rate: {embedding_stats.get('cache_hit_rate', 0):.2%}\n"
            status_output += f"   - Device: {embedding_stats.get('device', 'Unknown')}\n\n"
            
            # LLM system status
            llm_stats = stats.get("llm_system", {})
            status_output += "🤖 LLM System:\n"
            status_output += f"   - Ollama available: {llm_stats.get('ollama_available', False)}\n"
            
            if llm_stats.get('ollama_available'):
                generation_stats = llm_stats.get('generation_stats', {})
                status_output += f"   - Total requests: {generation_stats.get('total_requests', 0)}\n"
                status_output += f"   - Success rate: {generation_stats.get('successful_requests', 0)}/{generation_stats.get('total_requests', 0)}\n"
                status_output += f"   - Avg response time: {generation_stats.get('average_response_time', 0):.2f}s\n"
                
                model_status = llm_stats.get('model_status', {})
                available_models = model_status.get('available_models', {})
                status_output += f"   - Available models: {list(available_models.keys())}\n"
            
            status_output += "\n"
            
            # Configuration
            config = stats.get("configuration", {})
            status_output += "⚙️  Configuration:\n"
            status_output += f"   - Chunk size: {config.get('chunk_size', 0)}\n"
            status_output += f"   - Text model: {config.get('text_model', 'Unknown')}\n"
            status_output += f"   - Image model: {config.get('image_model', 'Unknown')}\n"
            status_output += f"   - Primary LLM: {config.get('primary_llm', 'Unknown')}\n"
            status_output += f"   - Secondary LLM: {config.get('secondary_llm', 'Unknown')}\n"
            status_output += f"   - Citations enabled: {config.get('citations_enabled', False)}\n"
            
            return status_output
            
        except Exception as e:
            return f"❌ Error getting system status: {str(e)}"
    
    def _format_document_list(self) -> str:
        """Format the list of processed documents."""
        if not self.processed_documents:
            return "📭 No documents processed yet"
        
        doc_list = f"📚 Processed Documents ({len(self.processed_documents)}):\n\n"
        
        for file_path, doc_info in self.processed_documents.items():
            doc_list += f"📄 {doc_info['file_name']}\n"
            doc_list += f"   - Type: {doc_info['content_type']}\n"
            doc_list += f"   - Chunks: {doc_info['chunks_created']}\n"
            doc_list += f"   - Processing Time: {doc_info['processing_time']:.2f}s\n\n"
        
        return doc_list
    
    def clear_knowledge_base(self) -> Tuple[str, str]:
        """Clear all processed documents from the knowledge base."""
        try:
            if self.retrieval_system:
                self.retrieval_system.clear_all_data()
            
            self.processed_documents.clear()
            status_msg = "🧹 Knowledge base cleared successfully"
            doc_list = self._format_document_list()
            return status_msg, doc_list
            
        except Exception as e:
            error_msg = f"❌ Error clearing knowledge base: {str(e)}"
            return error_msg, ""
    
    def test_search_with_different_thresholds(self, query: str) -> str:
        """Test search with different similarity thresholds to help debug."""
        if not self.is_initialized or not query.strip():
            return "❌ System not initialized or empty query"
        
        try:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            
            output = f"🔍 Testing Query: '{query}'\n"
            output += "=" * 50 + "\n\n"
            
            for threshold in thresholds:
                results = self.retrieval_system.search(
                    query=query,
                    top_k=5,
                    similarity_threshold=threshold,
                    include_images=True
                )
                
                output += f"Threshold {threshold}: {len(results)} results"
                if results:
                    output += f" (top score: {results[0].similarity_score:.3f})"
                output += "\n"
                
                # Show top result for lower thresholds
                if results and threshold <= 0.3:
                    output += f"   Top result: {results[0].content[:100]}...\n"
            
            output += "\n💡 Recommendation: Use threshold 0.1-0.3 for broader search\n"
            return output
            
        except Exception as e:
            return f"❌ Error testing thresholds: {str(e)}"
    
    def create_interface(self):
        """Create the Gradio interface."""
        
        # Custom CSS for better styling
        css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .tab-nav {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        }
        .search-box {
            border: 2px solid #e2e8f0 !important;
            border-radius: 8px !important;
        }
        .status-box {
            background: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 8px !important;
            padding: 16px !important;
        }
        """
        
        with gr.Blocks(
            title="Multimodal RAG System",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="slate",
                neutral_hue="slate"
            ),
            css=css
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🚀 Multimodal RAG System
            
            **Features:** PDF, DOCX, Images, Audio processing | **Models:** Offline Phi3:mini + Llama3.2b
            
            Upload documents, process them, and ask intelligent questions with adjustable search parameters.
            """)
            
            with gr.Tabs():
                
                # Document Processing Tab
                with gr.Tab("📄 Document Processing"):
                    gr.Markdown("### Upload and Process Documents")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            file_upload = gr.File(
                                label="Upload Document",
                                file_types=[".pdf", ".docx", ".png", ".jpg", ".jpeg", ".mp3", ".wav", ".m4a"],
                                type="filepath"
                            )
                            
                            process_btn = gr.Button(
                                "🔄 Process Document", 
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=3):
                            processing_status = gr.Textbox(
                                label="Processing Status",
                                lines=8,
                                interactive=False,
                                elem_classes=["status-box"]
                            )
                    
                    with gr.Row():
                        document_list = gr.Textbox(
                            label="Processed Documents",
                            lines=10,
                            interactive=False,
                            value=self._format_document_list()
                        )
                    
                    with gr.Row():
                        clear_btn = gr.Button("🧹 Clear Knowledge Base", variant="secondary")
                
                # Query Interface Tab
                with gr.Tab("🔍 Intelligent Search"):
                    gr.Markdown("### Ask Questions About Your Documents")
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            query_input = gr.Textbox(
                                label="Your Question",
                                placeholder="What are pythogorean neutrosophic fuzzy graphs?",
                                lines=2,
                                elem_classes=["search-box"]
                            )
                            
                            with gr.Row():
                                search_btn = gr.Button(
                                    "🔍 Search & Generate Response", 
                                    variant="primary",
                                    size="lg"
                                )
                                test_thresholds_btn = gr.Button(
                                    "🧪 Test Different Thresholds",
                                    variant="secondary"
                                )
                        
                        with gr.Column(scale=2):
                            gr.Markdown("#### Search Parameters")
                            
                            top_k = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=5,
                                step=1,
                                label="Number of Results (top_k)"
                            )
                            
                            similarity_threshold = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.3,  # Lower default threshold
                                step=0.1,
                                label="Similarity Threshold"
                            )
                            
                            include_images = gr.Checkbox(
                                label="Include Image Results",
                                value=True
                            )
                            
                            temperature = gr.Slider(
                                minimum=0.0,
                                maximum=2.0,
                                value=0.7,
                                step=0.1,
                                label="Response Temperature"
                            )
                    
                    # Results display
                    with gr.Row():
                        with gr.Column():
                            search_results = gr.Textbox(
                                label="Search Results",
                                lines=12,
                                interactive=False
                            )
                        
                        with gr.Column():
                            response_text = gr.Textbox(
                                label="Generated Response",
                                lines=12,
                                interactive=False
                            )
                    
                    # Debug information
                    debug_info = gr.Textbox(
                        label="Debug Information",
                        lines=6,
                        interactive=False
                    )
                    
                    # Threshold testing results
                    threshold_results = gr.Textbox(
                        label="Threshold Test Results",
                        lines=8,
                        interactive=False
                    )
                
                # System Status Tab
                with gr.Tab("📊 System Status"):
                    gr.Markdown("### System Information and Statistics")
                    
                    status_btn = gr.Button("🔄 Refresh Status", variant="primary")
                    
                    system_status = gr.Textbox(
                        label="System Status",
                        lines=20,
                        interactive=False,
                        value=self.get_system_status()
                    )
                
                # Help Tab
                with gr.Tab("❓ Help"):
                    gr.Markdown("""
                    ### How to Use the Multimodal RAG System
                    
                    #### 1. Document Processing
                    - Upload PDF, DOCX, images, or audio files
                    - Click "Process Document" to add them to the knowledge base
                    - Monitor processing status and view processed documents list
                    
                    #### 2. Intelligent Search
                    - Enter your question in natural language
                    - Adjust search parameters:
                      - **Top K**: Number of relevant chunks to retrieve
                      - **Similarity Threshold**: Minimum similarity score (lower = more results)
                      - **Include Images**: Whether to search image content
                      - **Temperature**: Creativity of AI response (0.0 = focused, 1.0 = creative)
                    
                    #### 3. Troubleshooting Search Issues
                    - If no results found, try lowering the similarity threshold to 0.1-0.3
                    - Use the "Test Different Thresholds" button to find optimal settings
                    - Try broader, simpler search terms
                    - Check the debug information for detailed search metrics
                    
                    #### 4. Supported File Formats
                    - **Text**: PDF, DOCX
                    - **Images**: PNG, JPG, JPEG
                    - **Audio**: MP3, WAV, M4A
                    
                    #### 5. System Requirements
                    - Ollama server running locally (for AI responses)
                    - Models: phi3:mini, llama3.2:3b
                    - Python dependencies installed
                    
                    #### 6. Tips for Better Results
                    - Process multiple related documents for richer context
                    - Use specific but not overly technical terms
                    - Try different phrasings of your question
                    - Lower similarity threshold for exploratory searches
                    """)
            
            # Event handlers
            process_btn.click(
                fn=self.process_uploaded_file,
                inputs=[file_upload],
                outputs=[processing_status, document_list]
            )
            
            search_btn.click(
                fn=self.query_system,
                inputs=[query_input, top_k, similarity_threshold, include_images, temperature],
                outputs=[search_results, response_text, debug_info]
            )
            
            test_thresholds_btn.click(
                fn=self.test_search_with_different_thresholds,
                inputs=[query_input],
                outputs=[threshold_results]
            )
            
            clear_btn.click(
                fn=self.clear_knowledge_base,
                inputs=[],
                outputs=[processing_status, document_list]
            )
            
            status_btn.click(
                fn=self.get_system_status,
                inputs=[],
                outputs=[system_status]
            )
            
            # Auto-refresh document list when processing
            file_upload.change(
                fn=lambda: self._format_document_list(),
                inputs=[],
                outputs=[document_list]
            )
        
        return interface


def main():
    """Main function to launch the Gradio interface."""
    print("🚀 Starting Gradio Interface for Multimodal RAG System")
    print("=" * 60)
    
    try:
        # Create interface
        app = GradioRAGInterface()
        interface = app.create_interface()
        
        # Launch interface
        print("🌐 Launching web interface...")
        print("📱 The interface will open in your default web browser")
        print("🔗 You can also access it at: http://localhost:7860")
        print("=" * 60)
        
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True,
            quiet=False
        )
        
    except KeyboardInterrupt:
        print("\n👋 Interface stopped by user")
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()