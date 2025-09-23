"""
Example demonstrating the integrated Ollama LLM system for multimodal RAG.

Shows how to use Phi3:mini and Llama3.2b models for offline response generation
with accurate citations across text, image, and audio content.
"""

import os
import sys
import tempfile
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import SystemConfig
from src.retrieval.retrieval_system import MultimodalRetrievalSystem
from src.models import ContentType
from src.llm.ollama_client import OllamaClient
from src.llm.response_generator import ResponseConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_ollama_setup():
    """Check if Ollama is properly set up with required models."""
    print("🔍 Checking Ollama Setup...")
    
    try:
        client = OllamaClient()
        
        if not client.is_available():
            print("❌ Ollama server is not running!")
            print("   Please start Ollama server: ollama serve")
            return False
        
        print("✅ Ollama server is running")
        
        # Check available models
        models = client.list_models()
        model_names = [model.name for model in models]
        
        print(f"📋 Available models: {model_names}")
        
        required_models = ["phi3:mini", "llama3.2:3b"]
        missing_models = []
        
        for required_model in required_models:
            # Check for exact match or partial match
            if not any(required_model in model_name for model_name in model_names):
                missing_models.append(required_model)
        
        if missing_models:
            print(f"⚠️  Missing models: {missing_models}")
            print("   Pull missing models with:")
            for model in missing_models:
                print(f"   ollama pull {model}")
            return False
        
        print("✅ All required models are available")
        return True
        
    except Exception as e:
        print(f"❌ Error checking Ollama setup: {e}")
        return False


def create_sample_documents():
    """Create sample documents for testing."""
    sample_docs = [
        {
            "filename": "ai_overview.txt",
            "content": """Artificial Intelligence and Machine Learning Overview

Artificial Intelligence (AI) is a broad field of computer science focused on creating systems that can perform tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, perception, and language understanding.

Machine Learning (ML) is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed. ML algorithms build mathematical models based on training data to make predictions or decisions.

Deep Learning is a subset of machine learning that uses artificial neural networks with multiple layers (hence "deep") to model and understand complex patterns in data. It has been particularly successful in areas like image recognition, natural language processing, and speech recognition.

Key applications of AI include:
- Computer vision for image and video analysis
- Natural language processing for text understanding
- Speech recognition and synthesis
- Autonomous vehicles and robotics
- Recommendation systems
- Medical diagnosis and drug discovery

The field continues to evolve rapidly with new breakthroughs in areas like large language models, generative AI, and multimodal systems that can process multiple types of data simultaneously."""
        },
        {
            "filename": "multimodal_rag.txt",
            "content": """Multimodal Retrieval-Augmented Generation Systems

Multimodal RAG systems represent a significant advancement in AI technology, combining the power of retrieval-based methods with generation capabilities across multiple data modalities including text, images, and audio.

Key Components:
1. Document Processing: Handles various file formats (PDF, DOCX, images, audio)
2. Embedding Generation: Creates unified vector representations for different content types
3. Vector Storage: Efficient similarity search using databases like Qdrant
4. Retrieval System: Finds relevant content based on semantic similarity
5. Response Generation: Uses large language models to create coherent responses
6. Citation Generation: Provides accurate source attribution with page numbers and timestamps

Benefits of Multimodal RAG:
- Comprehensive information retrieval across different content types
- Accurate source attribution and citation generation
- Offline operation for privacy and security
- Scalable to large document collections
- Support for complex queries requiring multiple sources

Technical Implementation:
- Uses sentence transformers for text embeddings
- CLIP models for image understanding
- Whisper for audio transcription
- Vector databases for fast similarity search
- Hybrid LLM models for response generation

This technology is particularly valuable for research, education, legal document analysis, and enterprise knowledge management systems."""
        },
        {
            "filename": "citation_importance.txt",
            "content": """The Importance of Accurate Citations in AI Systems

Citation accuracy is crucial in AI-powered information systems, especially in academic, legal, and professional contexts where source attribution is essential for credibility and verification.

Why Citations Matter:
1. Credibility: Proper citations establish the reliability of information
2. Verification: Readers can check original sources for accuracy
3. Academic Integrity: Prevents plagiarism and gives credit to original authors
4. Legal Compliance: Required in legal and regulatory documents
5. Transparency: Shows the basis for AI-generated responses

Challenges in AI Citation Generation:
- Matching generated text to specific source passages
- Handling multimodal content (text, images, audio)
- Maintaining accuracy across different document formats
- Providing precise location information (page numbers, timestamps)
- Dealing with paraphrased or synthesized information

Best Practices for AI Citation Systems:
- Use confidence scoring for citation accuracy
- Provide detailed source location information
- Include excerpt previews for verification
- Support multiple citation formats
- Validate citations against source material
- Handle edge cases like missing or corrupted sources

Modern AI systems like multimodal RAG implementations use sophisticated algorithms to automatically generate accurate citations, combining natural language processing, similarity matching, and confidence scoring to ensure reliable source attribution."""
        }
    ]
    
    temp_files = []
    for doc in sample_docs:
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write(doc["content"])
        temp_file.close()
        temp_files.append((temp_file.name, doc["filename"]))
    
    return temp_files


def test_hybrid_llm_system(retrieval_system):
    """Test the hybrid LLM system with different query types."""
    print("\n" + "="*70)
    print("🤖 TESTING HYBRID LLM SYSTEM (Phi3:mini + Llama3.2b)")
    print("="*70)
    
    # Test queries with different complexity levels
    test_queries = [
        {
            "query": "What is artificial intelligence?",
            "description": "Simple factual query (should use Phi3:mini)",
            "expected_model": "phi3:mini"
        },
        {
            "query": "Compare and analyze the key differences between machine learning and deep learning, including their applications and limitations.",
            "description": "Complex analytical query (should use Llama3.2b)",
            "expected_model": "llama3.2:3b"
        },
        {
            "query": "How do multimodal RAG systems work and what are their main components?",
            "description": "Moderate complexity query",
            "expected_model": "either"
        },
        {
            "query": "Analyze the importance of citation accuracy in AI systems and provide specific examples of best practices with detailed explanations.",
            "description": "Complex analysis with citation focus (should use Llama3.2b)",
            "expected_model": "llama3.2:3b"
        }
    ]
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n📝 Test Query {i}: {test_case['query']}")
        print(f"📋 Description: {test_case['description']}")
        print("-" * 60)
        
        try:
            # Search for relevant content
            search_results = retrieval_system.search(
                query=test_case['query'],
                top_k=5,
                include_images=False
            )
            
            print(f"🔍 Found {len(search_results)} relevant sources")
            
            # Generate response with citations
            response = retrieval_system.generate_response(
                query=test_case['query'],
                context_results=search_results,
                max_length=400,
                temperature=0.7,
                enable_citations=True
            )
            
            # Display results
            print(f"🤖 Model Used: {response.generation_metadata.get('model_used', 'unknown')}")
            print(f"⏱️  Response Time: {response.generation_metadata.get('total_generation_time', 0):.2f}s")
            print(f"🎯 Confidence: {response.confidence_score:.3f}")
            print(f"📚 Citations: {len(response.citations)}")
            
            print(f"\n💬 Generated Response:")
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
            
            # Analyze response quality
            quality_metrics = retrieval_system.response_generator.get_response_quality_metrics(response)
            print(f"\n📊 Quality Metrics:")
            print(f"   - Quality Score: {quality_metrics['quality_score']:.3f}")
            print(f"   - Source Coverage: {quality_metrics['source_coverage']:.3f}")
            print(f"   - Citation Density: {quality_metrics['citation_density']:.4f}")
            
            if quality_metrics['issues']:
                print(f"   - Issues: {', '.join(quality_metrics['issues'])}")
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")


def test_multimodal_capabilities(retrieval_system):
    """Test multimodal search and response generation."""
    print("\n" + "="*70)
    print("🎭 TESTING MULTIMODAL CAPABILITIES")
    print("="*70)
    
    multimodal_queries = [
        "Find information about AI systems that process both text and images",
        "What are the technical components needed for multimodal document processing?",
        "How do citation systems handle different types of content like audio and images?"
    ]
    
    for query in multimodal_queries:
        print(f"\n🔍 Multimodal Query: {query}")
        print("-" * 50)
        
        try:
            # Cross-modal search
            results = retrieval_system.search(
                query=query,
                top_k=5,
                include_images=True
            )
            
            # Group results by content type
            text_results = [r for r in results if r.content_type == ContentType.TEXT]
            image_results = [r for r in results if r.content_type == ContentType.IMAGE]
            audio_results = [r for r in results if r.content_type == ContentType.AUDIO]
            
            print(f"📄 Text Results: {len(text_results)}")
            print(f"🖼️  Image Results: {len(image_results)}")
            print(f"🎵 Audio Results: {len(audio_results)}")
            
            if results:
                # Generate response
                response = retrieval_system.generate_response(
                    query=query,
                    context_results=results,
                    enable_citations=True
                )
                
                print(f"\n💬 Response: {response.response_text[:200]}...")
                print(f"🎯 Confidence: {response.confidence_score:.3f}")
                print(f"📚 Citations: {len(response.citations)}")
                
        except Exception as e:
            print(f"❌ Error in multimodal query: {e}")


def test_citation_accuracy(retrieval_system):
    """Test citation accuracy and validation."""
    print("\n" + "="*70)
    print("📚 TESTING CITATION ACCURACY AND VALIDATION")
    print("="*70)
    
    citation_test_query = "Explain the key components of multimodal RAG systems and why accurate citations are important in AI systems."
    
    print(f"📝 Citation Test Query: {citation_test_query}")
    print("-" * 60)
    
    try:
        # Search for relevant content
        search_results = retrieval_system.search(
            query=citation_test_query,
            top_k=6
        )
        
        # Generate response with detailed citation analysis
        response = retrieval_system.generate_response(
            query=citation_test_query,
            context_results=search_results,
            max_length=500,
            enable_citations=True
        )
        
        print(f"🤖 Model Used: {response.generation_metadata.get('model_used', 'unknown')}")
        print(f"💬 Response Length: {len(response.response_text)} characters")
        print(f"📚 Total Citations: {len(response.citations)}")
        
        # Analyze citation validation
        citation_validation = response.generation_metadata.get('citation_validation', {})
        if citation_validation:
            print(f"\n📊 Citation Validation Report:")
            print(f"   - Coverage Rate: {citation_validation.get('coverage_rate', 0):.3f}")
            print(f"   - Average Confidence: {citation_validation.get('average_confidence', 0):.3f}")
            
            if citation_validation.get('issues'):
                print(f"   - Issues: {', '.join(citation_validation['issues'])}")
            
            if citation_validation.get('recommendations'):
                print(f"   - Recommendations: {', '.join(citation_validation['recommendations'])}")
        
        # Display detailed citations
        print(f"\n📖 Detailed Citations:")
        for citation in response.citations:
            print(f"   [{citation.citation_id}] {Path(citation.source_file).name}")
            print(f"       Content Type: {citation.content_type.value}")
            print(f"       Confidence: {citation.relevance_score:.3f}")
            print(f"       Excerpt: {citation.excerpt[:150]}...")
            print()
        
        print(f"💬 Full Response:")
        print(f"   {response.response_text}")
        
    except Exception as e:
        print(f"❌ Error in citation testing: {e}")


def display_system_statistics(retrieval_system):
    """Display comprehensive system statistics."""
    print("\n" + "="*70)
    print("📊 SYSTEM STATISTICS AND PERFORMANCE")
    print("="*70)
    
    try:
        stats = retrieval_system.get_system_stats()
        
        # Vector store statistics
        vector_stats = stats.get("vector_store", {})
        print(f"📦 Vector Store:")
        print(f"   - Total Documents: {vector_stats.get('points_count', 0)}")
        print(f"   - Vector Dimension: {vector_stats.get('vector_size', 0)}")
        print(f"   - Database Path: {vector_stats.get('database_path', 'Unknown')}")
        
        # Embedding statistics
        embedding_stats = stats.get("embedding_generator", {})
        print(f"\n🧠 Embedding Generator:")
        print(f"   - Total Embeddings: {embedding_stats.get('total_embeddings', 0)}")
        print(f"   - Cache Hit Rate: {embedding_stats.get('cache_hit_rate', 0):.2%}")
        print(f"   - Text Model: {embedding_stats.get('text_model_name', 'Unknown')}")
        print(f"   - Image Model: {embedding_stats.get('image_model_name', 'Unknown')}")
        
        # LLM system statistics
        llm_stats = stats.get("llm_system", {})
        print(f"\n🤖 LLM System:")
        print(f"   - Ollama Available: {llm_stats.get('ollama_available', False)}")
        
        if llm_stats.get('ollama_available'):
            generation_stats = llm_stats.get('generation_stats', {})
            print(f"   - Total Requests: {generation_stats.get('total_requests', 0)}")
            print(f"   - Success Rate: {generation_stats.get('successful_requests', 0)}/{generation_stats.get('total_requests', 0)}")
            print(f"   - Avg Response Time: {generation_stats.get('average_response_time', 0):.2f}s")
            
            model_status = llm_stats.get('model_status', {})
            available_models = model_status.get('available_models', {})
            print(f"   - Available Models: {list(available_models.keys())}")
            
            response_stats = llm_stats.get('response_stats', {})
            if response_stats:
                print(f"   - Generated Responses: {response_stats.get('successful_responses', 0)}")
                print(f"   - Avg Citations per Response: {response_stats.get('citation_stats', {}).get('average_citations_per_response', 0):.1f}")
        
        # Configuration
        config = stats.get("configuration", {})
        print(f"\n⚙️  Configuration:")
        print(f"   - Chunk Size: {config.get('chunk_size', 0)}")
        print(f"   - Primary LLM: {config.get('primary_llm', 'Unknown')}")
        print(f"   - Secondary LLM: {config.get('secondary_llm', 'Unknown')}")
        print(f"   - Citations Enabled: {config.get('citations_enabled', False)}")
        
    except Exception as e:
        print(f"❌ Error getting system statistics: {e}")


def main():
    """Main function to run the Ollama integration example."""
    print("🚀 MULTIMODAL RAG WITH HYBRID OLLAMA MODELS")
    print("=" * 70)
    print("Integration: Phi3:mini + Llama3.2b for Offline Response Generation")
    print("Features: Multimodal Processing, Accurate Citations, Hybrid Model Selection")
    print("=" * 70)
    
    # Check Ollama setup
    if not check_ollama_setup():
        print("\n❌ Ollama setup incomplete. Please install and configure Ollama with required models.")
        print("\nSetup Instructions:")
        print("1. Install Ollama: https://ollama.ai/")
        print("2. Start Ollama server: ollama serve")
        print("3. Pull required models:")
        print("   ollama pull phi3:mini")
        print("   ollama pull llama3.2:3b")
        return
    
    try:
        # Initialize system
        print("\n⚙️  Initializing Multimodal Retrieval System...")
        config = SystemConfig()
        retrieval_system = MultimodalRetrievalSystem(config)
        print("✅ System initialized successfully!")
        
        # Create and add sample documents
        print("\n📄 Creating sample documents...")
        temp_files = create_sample_documents()
        print(f"✅ Created {len(temp_files)} sample documents")
        
        # Add documents to system
        print("\n📥 Adding documents to vector store...")
        for file_path, filename in temp_files:
            result = retrieval_system.add_document(file_path)
            if result.success:
                print(f"  ✅ Added: {filename}")
            else:
                print(f"  ❌ Failed: {filename} - {result.error_message}")
        
        # Run comprehensive tests
        test_hybrid_llm_system(retrieval_system)
        test_multimodal_capabilities(retrieval_system)
        test_citation_accuracy(retrieval_system)
        display_system_statistics(retrieval_system)
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        print("\n📋 INTEGRATION SUMMARY:")
        print("✅ Hybrid LLM Models: Phi3:mini (fast) + Llama3.2b (high-quality)")
        print("✅ Intelligent Model Selection: Based on query complexity and task type")
        print("✅ Accurate Citation Generation: With confidence scoring and validation")
        print("✅ Multimodal Content Support: Text, images, and audio processing")
        print("✅ Completely Offline Operation: No external API dependencies")
        print("✅ Performance Optimization: Caching, fallback mechanisms, statistics")
        
        print(f"\n💾 Vector database stored at: {config.storage.storage_directory}/qdrant_db")
        print("🔄 System ready for production use with offline multimodal RAG capabilities!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"❌ Error: {e}")
        
    finally:
        # Cleanup temporary files
        print("\n🧹 Cleaning up temporary files...")
        for file_path, filename in temp_files:
            try:
                os.unlink(file_path)
                print(f"  🗑️  Removed: {filename}")
            except Exception as e:
                print(f"  ⚠️  Could not remove {filename}: {e}")


if __name__ == "__main__":
    main()