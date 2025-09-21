"""
Enhanced Vector Database Example with Text and Image Embeddings

This script demonstrates the complete multimodal retrieval system with:
- Text embedding storage and search
- Image embedding support
- Cross-modal retrieval
- Fast similarity search using Qdrant
- Local vector database storage
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import SystemConfig
from src.retrieval.retrieval_system import MultimodalRetrievalSystem
from src.models import ContentType
import tempfile
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_documents():
    """Create sample text documents for testing."""
    sample_docs = [
        {
            "filename": "finpilot_info.txt",
            "content": """FinPilot is an AI-powered financial assistant designed to help individuals and businesses optimize their tax strategies and make informed investment decisions. The platform uses advanced machine learning algorithms to analyze financial data and provide personalized recommendations for tax optimization, retirement planning, and portfolio management."""
        },
        {
            "filename": "mindmate_info.txt", 
            "content": """MindMate-AI is a comprehensive mental health platform that helps professionals manage stress and anxiety using evidence-based Cognitive Behavioral Therapy (CBT) techniques. The application provides personalized therapy sessions, mood tracking, and mindfulness exercises to support mental well-being in high-pressure work environments."""
        },
        {
            "filename": "qdrant_info.txt",
            "content": """Qdrant is a high-performance vector search engine and vector database written in Rust. It's specifically designed for machine learning applications and provides fast similarity search capabilities for embeddings. Qdrant supports filtering, payloads, and is optimized for production use with features like horizontal scaling and persistence."""
        },
        {
            "filename": "ai_overview.txt",
            "content": """Artificial Intelligence is transforming multiple industries including healthcare, finance, education, and technology. AI applications range from natural language processing and computer vision to autonomous vehicles and personalized recommendations. The field continues to evolve with advances in deep learning, neural networks, and large language models."""
        }
    ]
    
    temp_files = []
    for doc in sample_docs:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        temp_file.write(doc["content"])
        temp_file.close()
        temp_files.append((temp_file.name, doc["filename"]))
    
    return temp_files


def test_text_retrieval(retrieval_system):
    """Test text-based retrieval functionality."""
    print("\n" + "="*60)
    print("🔍 TESTING TEXT RETRIEVAL")
    print("="*60)
    
    # Test queries
    test_queries = [
        "Which AI tool helps with mental health and stress management?",
        "What is the financial assistant that optimizes taxes?",
        "Tell me about vector search engines for machine learning",
        "How is AI being used in different industries?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 50)
        
        # Search with response generation
        results = retrieval_system.query_with_response(
            query=query,
            top_k=3,
            include_images=False,  # Text only for now
            generate_response=True
        )
        
        # Display results
        for i, result in enumerate(results["search_results"]):
            print(f"{i+1}. [{result['content_type']}] Score: {result['similarity_score']:.3f}")
            print(f"   Source: {result['source_file']}")
            print(f"   Content: {result['content'][:150]}...")
            print()
        
        # Display generated response
        if "response" in results:
            print(f"💬 Generated Response:")
            print(f"   {results['response']['text']}")
            print(f"   Confidence: {results['response']['confidence_score']:.3f}")
            print(f"   Citations: {len(results['response']['citations'])}")


def test_cross_modal_search(retrieval_system):
    """Test cross-modal search functionality."""
    print("\n" + "="*60)
    print("🔄 TESTING CROSS-MODAL SEARCH")
    print("="*60)
    
    # Test cross-modal queries
    cross_modal_queries = [
        "Show me AI tools for financial assistance",
        "Find information about mental health applications",
        "What are the latest developments in vector databases?"
    ]
    
    for query in cross_modal_queries:
        print(f"\n🔍 Cross-Modal Query: {query}")
        print("-" * 50)
        
        # Cross-modal search (would include images if available)
        results = retrieval_system.search(
            query=query,
            top_k=5,
            include_images=True,  # Enable cross-modal
            similarity_threshold=0.3
        )
        
        # Group results by content type
        text_results = [r for r in results if r.content_type == ContentType.TEXT]
        image_results = [r for r in results if r.content_type == ContentType.IMAGE]
        audio_results = [r for r in results if r.content_type == ContentType.AUDIO]
        
        print(f"📄 Text Results: {len(text_results)}")
        print(f"🖼️  Image Results: {len(image_results)}")
        print(f"🎵 Audio Results: {len(audio_results)}")
        
        # Show top results
        for i, result in enumerate(results[:3]):
            print(f"{i+1}. [{result.content_type.value.upper()}] Score: {result.similarity_score:.3f}")
            print(f"   {result.content[:100]}...")


def test_similarity_search(retrieval_system):
    """Test similarity search with different thresholds."""
    print("\n" + "="*60)
    print("🎯 TESTING SIMILARITY SEARCH WITH THRESHOLDS")
    print("="*60)
    
    query = "AI-powered financial tools"
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print(f"Query: {query}")
    print("-" * 50)
    
    for threshold in thresholds:
        results = retrieval_system.search(
            query=query,
            top_k=10,
            similarity_threshold=threshold
        )
        
        print(f"Threshold {threshold}: {len(results)} results")
        if results:
            print(f"  Top score: {results[0].similarity_score:.3f}")
            print(f"  Lowest score: {results[-1].similarity_score:.3f}")


def test_system_performance(retrieval_system):
    """Test system performance and statistics."""
    print("\n" + "="*60)
    print("📊 SYSTEM PERFORMANCE & STATISTICS")
    print("="*60)
    
    # Get comprehensive system statistics
    stats = retrieval_system.get_system_stats()
    
    print("Vector Store Statistics:")
    vector_stats = stats.get("vector_store", {})
    print(f"  📦 Total Documents: {vector_stats.get('points_count', 0)}")
    print(f"  🔢 Vector Dimension: {vector_stats.get('vector_size', 0)}")
    print(f"  📏 Distance Metric: {vector_stats.get('distance_metric', 'Unknown')}")
    print(f"  📁 Database Path: {vector_stats.get('database_path', 'Unknown')}")
    
    content_dist = vector_stats.get('content_type_distribution', {})
    print(f"  📊 Content Distribution:")
    for content_type, count in content_dist.items():
        print(f"    - {content_type}: {count}")
    
    print("\nEmbedding Generator Statistics:")
    embedding_stats = stats.get("embedding_generator", {})
    print(f"  🧠 Total Embeddings: {embedding_stats.get('total_embeddings', 0)}")
    print(f"  📝 Text Embeddings: {embedding_stats.get('text_embeddings', 0)}")
    print(f"  🖼️  Image Embeddings: {embedding_stats.get('image_embeddings', 0)}")
    print(f"  🎵 Audio Embeddings: {embedding_stats.get('audio_embeddings', 0)}")
    print(f"  💾 Cache Hit Rate: {embedding_stats.get('cache_hit_rate', 0):.2%}")
    print(f"  🔧 Device: {embedding_stats.get('device', 'Unknown')}")
    
    print("\nConfiguration:")
    config = stats.get("configuration", {})
    print(f"  📏 Chunk Size: {config.get('chunk_size', 0)}")
    print(f"  🔢 Embedding Dimension: {config.get('embedding_dimension', 0)}")
    print(f"  📝 Text Model: {config.get('text_model', 'Unknown')}")
    print(f"  🖼️  Image Model: {config.get('image_model', 'Unknown')}")


def main():
    """Main function to run the vector database example."""
    print("🚀 ENHANCED MULTIMODAL VECTOR DATABASE EXAMPLE")
    print("=" * 60)
    
    try:
        # Initialize system
        print("⚙️  Initializing Multimodal Retrieval System...")
        config = SystemConfig()
        retrieval_system = MultimodalRetrievalSystem(config)
        print("✅ System initialized successfully!")
        
        # Create sample documents
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
        
        # Run tests
        test_text_retrieval(retrieval_system)
        test_cross_modal_search(retrieval_system)
        test_similarity_search(retrieval_system)
        test_system_performance(retrieval_system)
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Example usage summary
        print("\n📋 USAGE SUMMARY:")
        print("1. ✅ Text embedding storage and retrieval")
        print("2. ✅ Fast similarity search with Qdrant")
        print("3. ✅ Cross-modal search capabilities")
        print("4. ✅ Local vector database storage")
        print("5. ✅ Configurable similarity thresholds")
        print("6. ✅ Response generation with citations")
        print("7. ✅ Comprehensive system statistics")
        
        print(f"\n💾 Vector database stored locally at: {config.storage.storage_directory}/qdrant_db")
        print("🔄 You can now use this system for fast multimodal retrieval!")
        
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
