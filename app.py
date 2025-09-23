#!/usr/bin/env python3
"""
Multimodal RAG Application

A comprehensive web application for processing and querying multimodal documents
including text (PDF, DOCX), images, and audio files using offline AI models.

Features:
- Document upload and processing
- Multimodal content indexing
- Semantic search across all content types
- Response generation with citations
- Offline operation with local models
"""

import os
import sys
import logging
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multimodal_rag.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultimodalRAGApp:
    """
    Main application class for the Multimodal RAG system.
    
    Provides a simple interface for document processing, indexing,
    and querying with support for text, image, and audio content.
    """
    
    def __init__(self):
        """Initialize the application."""
        self.config = SystemConfig()
        self.retrieval_system = None
        self.is_initialized = False
        self.processed_documents = {}
        
        # Create necessary directories
        self._create_directories()
        
        print("🚀 Multimodal RAG Application")
        print("=" * 50)
        print("Features: PDF, DOCX, Images, Audio processing")
        print("Models: Offline Phi3:mini + Llama3.2b")
        print("=" * 50)
    
    def _create_directories(self):
        """Create necessary directories for the application."""
        directories = [
            self.config.data_directory,
            self.config.storage.storage_directory,
            self.config.embedding.embedding_cache_dir,
            "uploads",
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def initialize_system(self) -> bool:
        """
        Initialize the retrieval system and check dependencies.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print("\n⚙️  Initializing Multimodal RAG System...")
            
            # Initialize retrieval system
            self.retrieval_system = MultimodalRetrievalSystem(self.config)
            
            # Check Ollama availability
            ollama_available = self._check_ollama_setup()
            if not ollama_available:
                print("⚠️  Ollama not available - responses will use fallback mode")
            
            self.is_initialized = True
            print("✅ System initialized successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            logger.error(f"System initialization failed: {e}")
            return False
    
    def _check_ollama_setup(self) -> bool:
        """Check if Ollama is properly set up."""
        try:
            client = OllamaClient()
            
            if not client.is_available():
                print("⚠️  Ollama server is not running")
                return False
            
            # Check for required models
            models = client.list_models()
            model_names = [model.name for model in models]
            
            required_models = ["phi3:mini", "llama3.2:3b"]
            missing_models = []
            
            for required_model in required_models:
                if not any(required_model in model_name for model_name in model_names):
                    missing_models.append(required_model)
            
            if missing_models:
                print(f"⚠️  Missing Ollama models: {missing_models}")
                print("   Install with:")
                for model in missing_models:
                    print(f"   ollama pull {model}")
                return False
            
            print("✅ Ollama setup complete")
            return True
            
        except Exception as e:
            print(f"⚠️  Ollama check failed: {e}")
            return False
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single file and add it to the knowledge base.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary with processing results
        """
        if not self.is_initialized:
            return {"success": False, "error": "System not initialized"}
        
        try:
            print(f"\n📄 Processing file: {Path(file_path).name}")
            
            # Validate file exists
            if not Path(file_path).exists():
                return {"success": False, "error": f"File not found: {file_path}"}
            
            # Process document
            result = self.retrieval_system.add_document(file_path)
            
            if result.success:
                # Store processing info
                self.processed_documents[file_path] = {
                    "document_id": result.document_content.document_id,
                    "content_type": result.document_content.content_type.value,
                    "chunks_created": result.chunks_created,
                    "processing_time": result.processing_time,
                    "metadata": result.document_content.metadata.__dict__
                }
                
                print(f"✅ Successfully processed: {Path(file_path).name}")
                print(f"   - Document ID: {result.document_content.document_id}")
                print(f"   - Content Type: {result.document_content.content_type.value}")
                print(f"   - Chunks Created: {result.chunks_created}")
                print(f"   - Processing Time: {result.processing_time:.2f}s")
                
                return {
                    "success": True,
                    "document_id": result.document_content.document_id,
                    "content_type": result.document_content.content_type.value,
                    "chunks_created": result.chunks_created,
                    "processing_time": result.processing_time
                }
            else:
                print(f"❌ Failed to process: {Path(file_path).name}")
                print(f"   Error: {result.error_message}")
                return {"success": False, "error": result.error_message}
                
        except Exception as e:
            error_msg = f"Error processing file {file_path}: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def process_directory(self, directory_path: str, recursive: bool = False) -> Dict[str, Any]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to directory containing files
            recursive: Whether to process subdirectories
            
        Returns:
            Dictionary with batch processing results
        """
        if not self.is_initialized:
            return {"success": False, "error": "System not initialized"}
        
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                return {"success": False, "error": f"Directory not found: {directory_path}"}
            
            print(f"\n📁 Processing directory: {directory.name}")
            print(f"   Recursive: {recursive}")
            
            # Find supported files
            supported_extensions = {'.pdf', '.docx', '.png', '.jpg', '.jpeg', '.mp3', '.wav', '.m4a'}
            
            if recursive:
                files = [f for f in directory.rglob("*") if f.suffix.lower() in supported_extensions]
            else:
                files = [f for f in directory.iterdir() if f.suffix.lower() in supported_extensions]
            
            print(f"   Found {len(files)} supported files")
            
            # Process files
            results = {
                "total_files": len(files),
                "successful": 0,
                "failed": 0,
                "results": [],
                "errors": []
            }
            
            for file_path in files:
                print(f"\n   Processing: {file_path.name}")
                file_result = self.process_file(str(file_path))
                
                if file_result["success"]:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "file": str(file_path),
                        "error": file_result["error"]
                    })
                
                results["results"].append({
                    "file": str(file_path),
                    **file_result
                })
            
            print(f"\n📊 Batch Processing Summary:")
            print(f"   Total Files: {results['total_files']}")
            print(f"   Successful: {results['successful']}")
            print(f"   Failed: {results['failed']}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error processing directory {directory_path}: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def query_system(self, query: str, top_k: int = 5, include_images: bool = True) -> Dict[str, Any]:
        """
        Query the system and generate a response with citations.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            include_images: Whether to include image results
            
        Returns:
            Dictionary with query results and response
        """
        if not self.is_initialized:
            return {"success": False, "error": "System not initialized"}
        
        try:
            print(f"\n🔍 Query: {query}")
            print("-" * 50)
            
            start_time = time.time()
            
            # Search for relevant content
            search_results = self.retrieval_system.search(
                query=query,
                top_k=top_k,
                include_images=include_images
            )
            
            search_time = time.time() - start_time
            print(f"🔍 Found {len(search_results)} relevant sources in {search_time:.2f}s")
            
            # Display search results
            for i, result in enumerate(search_results, 1):
                print(f"   {i}. [{result.content_type.value.upper()}] Score: {result.similarity_score:.3f}")
                print(f"      Source: {Path(result.source_location.file_path).name}")
                print(f"      Content: {result.content[:100]}...")
                if result.source_location.page_number:
                    print(f"      Page: {result.source_location.page_number}")
                if result.source_location.timestamp_start:
                    print(f"      Time: {result.source_location.timestamp_start:.1f}s")
                print()
            
            # Generate response if results found
            response_data = None
            if search_results:
                print("🤖 Generating response...")
                response_start = time.time()
                
                response = self.retrieval_system.generate_response(
                    query=query,
                    context_results=search_results,
                    max_length=500,
                    temperature=0.7,
                    enable_citations=True
                )
                
                response_time = time.time() - response_start
                
                print(f"✅ Response generated in {response_time:.2f}s")
                print(f"🤖 Model: {response.generation_metadata.get('model_used', 'unknown')}")
                print(f"🎯 Confidence: {response.confidence_score:.3f}")
                print(f"📚 Citations: {len(response.citations)}")
                
                print(f"\n💬 Response:")
                print(f"   {response.response_text}")
                
                if response.citations:
                    print(f"\n📖 Citations:")
                    for citation in response.citations:
                        source_name = Path(citation.source_file).name
                        location_info = ""
                        if citation.location.page_number:
                            location_info += f", p.{citation.location.page_number}"
                        if citation.location.timestamp_start:
                            location_info += f", {citation.location.timestamp_start:.1f}s"
                        
                        print(f"   [{citation.citation_id}] {source_name}{location_info}")
                        print(f"       Confidence: {citation.relevance_score:.3f}")
                        print(f"       Excerpt: {citation.excerpt[:100]}...")
                
                response_data = {
                    "text": response.response_text,
                    "confidence": response.confidence_score,
                    "model_used": response.generation_metadata.get('model_used', 'unknown'),
                    "citations": [
                        {
                            "id": c.citation_id,
                            "source": Path(c.source_file).name,
                            "excerpt": c.excerpt,
                            "confidence": c.relevance_score,
                            "page": c.location.page_number,
                            "timestamp": c.location.timestamp_start
                        }
                        for c in response.citations
                    ],
                    "generation_time": response_time
                }
            
            return {
                "success": True,
                "query": query,
                "search_results": [
                    {
                        "content": r.content,
                        "similarity_score": r.similarity_score,
                        "content_type": r.content_type.value,
                        "source_file": Path(r.source_location.file_path).name,
                        "page_number": r.source_location.page_number,
                        "timestamp": r.source_location.timestamp_start
                    }
                    for r in search_results
                ],
                "response": response_data,
                "search_time": search_time,
                "total_time": time.time() - start_time
            }
            
        except Exception as e:
            error_msg = f"Error querying system: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            traceback.print_exc()
            return {"success": False, "error": error_msg}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and statistics."""
        try:
            if not self.is_initialized:
                return {"initialized": False, "error": "System not initialized"}
            
            stats = self.retrieval_system.get_system_stats()
            
            return {
                "initialized": True,
                "processed_documents": len(self.processed_documents),
                "vector_store": stats.get("vector_store", {}),
                "embedding_generator": stats.get("embedding_generator", {}),
                "llm_system": stats.get("llm_system", {}),
                "configuration": stats.get("configuration", {}),
                "supported_formats": stats.get("supported_formats", [])
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"initialized": False, "error": str(e)}
    
    def list_processed_documents(self) -> List[Dict[str, Any]]:
        """List all processed documents with their metadata."""
        return [
            {
                "file_path": file_path,
                "file_name": Path(file_path).name,
                **doc_info
            }
            for file_path, doc_info in self.processed_documents.items()
        ]
    
    def clear_knowledge_base(self) -> bool:
        """Clear all processed documents from the knowledge base."""
        try:
            if self.retrieval_system:
                self.retrieval_system.clear_all_data()
            
            self.processed_documents.clear()
            print("🧹 Knowledge base cleared successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error clearing knowledge base: {e}")
            logger.error(f"Error clearing knowledge base: {e}")
            return False


def create_sample_documents():
    """Create sample documents for testing."""
    sample_docs = [
        {
            "filename": "ai_overview.txt",
            "content": """Artificial Intelligence and Machine Learning Overview

Artificial Intelligence (AI) is revolutionizing multiple industries including healthcare, finance, education, and technology. AI applications range from natural language processing and computer vision to autonomous vehicles and personalized recommendations.

Machine Learning (ML) is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers.

Key AI Technologies:
- Natural Language Processing (NLP) for text understanding
- Computer Vision for image and video analysis
- Speech Recognition and Synthesis
- Robotics and Autonomous Systems
- Recommendation Systems
- Predictive Analytics

The field continues to evolve rapidly with new breakthroughs in large language models, generative AI, and multimodal systems that can process multiple types of data simultaneously."""
        },
        {
            "filename": "fintech_solutions.txt",
            "content": """Financial Technology Solutions

FinPilot is an innovative AI-powered financial assistant designed to help individuals and businesses optimize their tax strategies and make informed investment decisions. The platform uses advanced machine learning algorithms to analyze financial data and provide personalized recommendations.

Key Features of FinPilot:
- Tax optimization strategies
- Investment portfolio analysis
- Risk assessment and management
- Automated financial reporting
- Compliance monitoring
- Real-time market analysis

The system integrates with major financial institutions and provides secure, encrypted data processing to ensure user privacy and regulatory compliance. FinPilot's AI algorithms continuously learn from market trends and user preferences to improve recommendation accuracy."""
        },
        {
            "filename": "mental_health_tech.txt",
            "content": """Mental Health Technology Platform

MindMate-AI is a comprehensive mental health platform that helps professionals manage stress and anxiety using evidence-based Cognitive Behavioral Therapy (CBT) techniques. The application provides personalized therapy sessions, mood tracking, and mindfulness exercises.

Core Features:
- Personalized CBT therapy sessions
- Mood and stress level tracking
- Mindfulness and meditation exercises
- Progress monitoring and analytics
- Crisis intervention protocols
- Professional therapist integration

The platform uses AI to analyze user patterns and provide customized interventions. MindMate-AI is designed specifically for high-pressure work environments and helps users develop coping strategies for workplace stress, anxiety, and burnout.

Clinical studies have shown significant improvements in user mental health outcomes when using the platform consistently over 8-12 weeks."""
        }
    ]
    
    # Create uploads directory
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    
    created_files = []
    for doc in sample_docs:
        file_path = uploads_dir / doc["filename"]
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(doc["content"])
        created_files.append(str(file_path))
        print(f"📝 Created sample document: {doc['filename']}")
    
    return created_files


def interactive_mode(app: MultimodalRAGApp):
    """Run the application in interactive mode."""
    print("\n🎯 Interactive Mode")
    print("Commands:")
    print("  1. process <file_path>     - Process a single file")
    print("  2. process_dir <dir_path>  - Process all files in directory")
    print("  3. query <question>        - Ask a question")
    print("  4. status                  - Show system status")
    print("  5. list                    - List processed documents")
    print("  6. clear                   - Clear knowledge base")
    print("  7. help                    - Show this help")
    print("  8. quit                    - Exit application")
    print()
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(' ', 1)
            command = parts[0].lower()
            
            if command == "quit" or command == "exit":
                print("👋 Goodbye!")
                break
            
            elif command == "help":
                print("\n📋 Available Commands:")
                print("  process <file_path>     - Process a single file")
                print("  process_dir <dir_path>  - Process directory (add 'recursive' for subdirs)")
                print("  query <question>        - Ask a question")
                print("  status                  - Show system status")
                print("  list                    - List processed documents")
                print("  clear                   - Clear knowledge base")
                print("  help                    - Show this help")
                print("  quit                    - Exit application")
            
            elif command == "process":
                if len(parts) < 2:
                    print("❌ Usage: process <file_path>")
                    continue
                
                file_path = parts[1].strip()
                result = app.process_file(file_path)
                
                if not result["success"]:
                    print(f"❌ Processing failed: {result['error']}")
            
            elif command == "process_dir":
                if len(parts) < 2:
                    print("❌ Usage: process_dir <directory_path> [recursive]")
                    continue
                
                dir_parts = parts[1].split()
                dir_path = dir_parts[0]
                recursive = len(dir_parts) > 1 and dir_parts[1].lower() == "recursive"
                
                result = app.process_directory(dir_path, recursive)
                
                if not result["success"]:
                    print(f"❌ Directory processing failed: {result['error']}")
            
            elif command == "query":
                if len(parts) < 2:
                    print("❌ Usage: query <your question>")
                    continue
                
                question = parts[1].strip()
                result = app.query_system(question)
                
                if not result["success"]:
                    print(f"❌ Query failed: {result['error']}")
            
            elif command == "status":
                status = app.get_system_status()
                
                if status["initialized"]:
                    print("\n📊 System Status:")
                    print(f"   Processed Documents: {status['processed_documents']}")
                    
                    vector_stats = status.get("vector_store", {})
                    print(f"   Vector Store Points: {vector_stats.get('points_count', 0)}")
                    print(f"   Vector Dimension: {vector_stats.get('vector_size', 0)}")
                    
                    embedding_stats = status.get("embedding_generator", {})
                    print(f"   Total Embeddings: {embedding_stats.get('total_embeddings', 0)}")
                    print(f"   Cache Hit Rate: {embedding_stats.get('cache_hit_rate', 0):.2%}")
                    
                    llm_stats = status.get("llm_system", {})
                    print(f"   Ollama Available: {llm_stats.get('ollama_available', False)}")
                    
                    config = status.get("configuration", {})
                    print(f"   Supported Formats: {', '.join(status.get('supported_formats', []))}")
                else:
                    print(f"❌ System not initialized: {status.get('error', 'Unknown error')}")
            
            elif command == "list":
                docs = app.list_processed_documents()
                
                if docs:
                    print(f"\n📚 Processed Documents ({len(docs)}):")
                    for doc in docs:
                        print(f"   📄 {doc['file_name']}")
                        print(f"      Type: {doc['content_type']}")
                        print(f"      Chunks: {doc['chunks_created']}")
                        print(f"      Processing Time: {doc['processing_time']:.2f}s")
                        print()
                else:
                    print("📭 No documents processed yet")
            
            elif command == "clear":
                confirm = input("⚠️  Are you sure you want to clear all data? (yes/no): ")
                if confirm.lower() in ['yes', 'y']:
                    if app.clear_knowledge_base():
                        print("✅ Knowledge base cleared")
                    else:
                        print("❌ Failed to clear knowledge base")
                else:
                    print("❌ Clear operation cancelled")
            
            else:
                print(f"❌ Unknown command: {command}")
                print("   Type 'help' for available commands")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            logger.error(f"Interactive mode error: {e}")


def main():
    """Main application entry point."""
    try:
        # Initialize application
        app = MultimodalRAGApp()
        
        # Initialize system
        if not app.initialize_system():
            print("❌ Failed to initialize system. Exiting.")
            return
        
        # Check command line arguments
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()
            
            if command == "demo":
                # Demo mode with sample documents
                print("\n🎬 Demo Mode - Creating sample documents...")
                sample_files = create_sample_documents()
                
                # Process sample documents
                for file_path in sample_files:
                    app.process_file(file_path)
                
                # Demo queries
                demo_queries = [
                    "What is FinPilot and how does it help with finances?",
                    "How does MindMate-AI help with mental health?",
                    "What are the key applications of artificial intelligence?",
                    "Compare the features of FinPilot and MindMate-AI"
                ]
                
                print("\n🎯 Demo Queries:")
                for query in demo_queries:
                    result = app.query_system(query)
                    if not result["success"]:
                        print(f"❌ Demo query failed: {result['error']}")
                    print("\n" + "="*60 + "\n")
            
            elif command == "process":
                if len(sys.argv) < 3:
                    print("❌ Usage: python app.py process <file_path>")
                    return
                
                file_path = sys.argv[2]
                app.process_file(file_path)
            
            elif command == "process_dir":
                if len(sys.argv) < 3:
                    print("❌ Usage: python app.py process_dir <directory_path> [recursive]")
                    return
                
                dir_path = sys.argv[2]
                recursive = len(sys.argv) > 3 and sys.argv[3].lower() == "recursive"
                app.process_directory(dir_path, recursive)
            
            elif command == "query":
                if len(sys.argv) < 3:
                    print("❌ Usage: python app.py query \"<your question>\"")
                    return
                
                query = " ".join(sys.argv[2:])
                app.query_system(query)
            
            elif command == "status":
                status = app.get_system_status()
                print("\n📊 System Status:")
                if status["initialized"]:
                    print(f"   ✅ System initialized")
                    print(f"   📄 Processed documents: {status['processed_documents']}")
                    vector_stats = status.get("vector_store", {})
                    print(f"   📦 Vector store points: {vector_stats.get('points_count', 0)}")
                else:
                    print(f"   ❌ System not initialized: {status.get('error', 'Unknown')}")
            
            elif command == "interactive":
                interactive_mode(app)
            
            else:
                print(f"❌ Unknown command: {command}")
                print("\nAvailable commands:")
                print("  demo                    - Run demo with sample documents")
                print("  process <file>          - Process a single file")
                print("  process_dir <dir>       - Process directory")
                print("  query \"<question>\"      - Ask a question")
                print("  status                  - Show system status")
                print("  interactive             - Enter interactive mode")
        
        else:
            # No command line arguments - enter interactive mode
            interactive_mode(app)
    
    except KeyboardInterrupt:
        print("\n\n👋 Application interrupted by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
        logger.error(f"Application error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()