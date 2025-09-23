#!/usr/bin/env python3
"""
Quick demo script for the Multimodal RAG system.

This script provides a simple way to test the system with sample documents
and demonstrates the key functionality without requiring manual setup.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from app import MultimodalRAGApp, create_sample_documents


def main():
    """Run a quick demo of the system."""
    print("🚀 MULTIMODAL RAG SYSTEM - QUICK DEMO")
    print("=" * 50)
    
    try:
        # Initialize application
        app = MultimodalRAGApp()
        
        # Initialize system
        print("⚙️  Initializing system...")
        if not app.initialize_system():
            print("❌ Failed to initialize system")
            return
        
        # Create and process sample documents
        print("\n📄 Creating sample documents...")
        sample_files = create_sample_documents()
        
        print(f"\n📥 Processing {len(sample_files)} sample documents...")
        for file_path in sample_files:
            result = app.process_file(file_path)
            if not result["success"]:
                print(f"❌ Failed to process {file_path}: {result['error']}")
        
        # Demo queries
        demo_queries = [
            "What is FinPilot and how does it help with financial planning?",
            "How does MindMate-AI support mental health for professionals?",
            "What are the main applications of artificial intelligence?",
            "Compare FinPilot and MindMate-AI - what problems do they solve?"
        ]
        
        print(f"\n🎯 Running {len(demo_queries)} demo queries...")
        print("=" * 60)
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n📝 Demo Query {i}: {query}")
            print("-" * 50)
            
            result = app.query_system(query, top_k=3)
            
            if not result["success"]:
                print(f"❌ Query failed: {result['error']}")
                continue
            
            print(f"⏱️  Total time: {result['total_time']:.2f}s")
            
            if result.get("response"):
                response = result["response"]
                print(f"\n💬 Response (Confidence: {response['confidence']:.3f}):")
                print(f"   {response['text']}")
                
                if response["citations"]:
                    print(f"\n📚 Citations ({len(response['citations'])}):")
                    for citation in response["citations"]:
                        print(f"   [{citation['id']}] {citation['source']} (confidence: {citation['confidence']:.3f})")
            
            print("\n" + "="*60)
        
        # Show final statistics
        print(f"\n📊 Final System Statistics:")
        status = app.get_system_status()
        
        if status["initialized"]:
            print(f"   📄 Documents Processed: {status['processed_documents']}")
            
            vector_stats = status.get("vector_store", {})
            print(f"   📦 Vector Store Points: {vector_stats.get('points_count', 0)}")
            
            embedding_stats = status.get("embedding_generator", {})
            print(f"   🧠 Total Embeddings: {embedding_stats.get('total_embeddings', 0)}")
            print(f"   💾 Cache Hit Rate: {embedding_stats.get('cache_hit_rate', 0):.2%}")
            
            llm_stats = status.get("llm_system", {})
            if llm_stats.get("ollama_available"):
                gen_stats = llm_stats.get("generation_stats", {})
                print(f"   🤖 LLM Requests: {gen_stats.get('total_requests', 0)}")
                print(f"   ✅ Success Rate: {gen_stats.get('successful_requests', 0)}/{gen_stats.get('total_requests', 0)}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"💾 Vector database stored at: {app.config.storage.storage_directory}/qdrant_db")
        print(f"📝 Logs available at: multimodal_rag.log")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()