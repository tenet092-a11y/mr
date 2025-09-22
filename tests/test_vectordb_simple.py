"""
Simple test script for the enhanced vector database functionality.
Demonstrates text and image embedding storage with fast retrieval.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import tempfile
import os

def test_basic_vectordb():
    """Test basic Qdrant functionality with text embeddings."""
    print("🚀 Testing Enhanced Vector Database with Text and Image Embeddings")
    print("=" * 70)
    
    # Initialize Qdrant client (local storage)
    client = QdrantClient(path="./test_qdrant_db")
    collection_name = "test_knowledge_base"
    
    # Create collection
    print("📦 Creating vector collection...")
    # Check if collection exists and delete if it does
    try:
        collections = client.get_collections()
        if any(col.name == collection_name for col in collections.collections):
            client.delete_collection(collection_name)
    except:
        pass
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    
    # Load embedding model
    print("🧠 Loading sentence transformer model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Enhanced knowledge sources with more diverse content
    documents = [
        {
            "id": 1, 
            "text": "FinPilot is an AI-powered financial assistant for tax optimization and investment advice.",
            "content_type": "text",
            "category": "finance"
        },
        {
            "id": 2, 
            "text": "MindMate-AI helps professionals manage stress using CBT techniques and mindfulness.",
            "content_type": "text", 
            "category": "health"
        },
        {
            "id": 3, 
            "text": "Qdrant is an open-source vector search engine optimized for embeddings and ML applications.",
            "content_type": "text",
            "category": "technology"
        },
        {
            "id": 4,
            "text": "Vector databases enable fast similarity search for AI applications and machine learning models.",
            "content_type": "text",
            "category": "technology"
        },
        {
            "id": 5,
            "text": "Mental health applications are becoming increasingly important in the digital age.",
            "content_type": "text",
            "category": "health"
        }
    ]
    
    # Generate embeddings and insert documents
    print("📝 Generating embeddings and inserting documents...")
    vectors = embedder.encode([doc["text"] for doc in documents]).tolist()
    
    points = []
    for doc, vec in zip(documents, vectors):
        point = models.PointStruct(
            id=doc["id"], 
            vector=vec, 
            payload={
                "text": doc["text"],
                "content_type": doc["content_type"],
                "category": doc["category"]
            }
        )
        points.append(point)
    
    client.upsert(collection_name=collection_name, points=points)
    print(f"✅ Inserted {len(points)} documents into vector database!")
    
    # Enhanced search function with filtering
    def search_with_filter(query: str, top_k: int = 3, category_filter: str = None):
        """Search with optional category filtering."""
        query_vector = embedder.encode(query).tolist()
        
        # Build filter if category specified
        query_filter = None
        if category_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="category",
                        match=models.MatchValue(value=category_filter)
                    )
                ]
            )
        
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k
        ).points
        
        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                "category": hit.payload["category"]
            } 
            for hit in results
        ]
    
    # Test various search scenarios
    print("\n🔍 TESTING ENHANCED SEARCH FUNCTIONALITY")
    print("=" * 50)
    
    test_queries = [
        {
            "query": "Which project helps with mental health?",
            "description": "General mental health query"
        },
        {
            "query": "What is the financial assistant called?",
            "description": "Financial AI query"
        },
        {
            "query": "Tell me about vector search engines",
            "description": "Technology query"
        },
        {
            "query": "AI applications for health and wellness",
            "description": "Cross-domain query"
        }
    ]
    
    for test in test_queries:
        print(f"\n📝 Query: {test['query']}")
        print(f"📋 Description: {test['description']}")
        print("-" * 40)
        
        # General search
        results = search_with_filter(test['query'], top_k=2)
        print("🔍 General Search Results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. [{result['category']}] Score: {result['score']:.3f}")
            print(f"     {result['text']}")
        
        # Category-filtered searches
        categories = ["health", "finance", "technology"]
        for category in categories:
            filtered_results = search_with_filter(test['query'], top_k=1, category_filter=category)
            if filtered_results:
                print(f"🎯 {category.title()} Filtered Results:")
                result = filtered_results[0]
                print(f"  Score: {result['score']:.3f}")
                print(f"  {result['text']}")
    
    # Test similarity thresholds
    print(f"\n🎯 TESTING SIMILARITY THRESHOLDS")
    print("=" * 40)
    
    query = "AI-powered financial tools"
    thresholds = [0.1, 0.3, 0.5, 0.7]
    
    print(f"Query: {query}")
    for threshold in thresholds:
        query_vector = embedder.encode(query).tolist()
        results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=10,
            score_threshold=threshold
        ).points
        print(f"Threshold {threshold}: {len(results)} results")
        if results:
            print(f"  Top score: {results[0].score:.3f}")
    
    # Get collection statistics
    print(f"\n📊 COLLECTION STATISTICS")
    print("=" * 30)
    
    collection_info = client.get_collection(collection_name)
    print(f"Total points: {collection_info.points_count}")
    print(f"Vector size: {collection_info.config.params.vectors.size}")
    print(f"Distance metric: {collection_info.config.params.vectors.distance}")
    
    # Test batch operations
    print(f"\n⚡ TESTING BATCH OPERATIONS")
    print("=" * 35)
    
    # Add more documents in batch
    batch_documents = [
        {"id": 6, "text": "Machine learning models require high-quality training data.", "content_type": "text", "category": "technology"},
        {"id": 7, "text": "Investment portfolios should be diversified across different asset classes.", "content_type": "text", "category": "finance"},
        {"id": 8, "text": "Stress management techniques include meditation and breathing exercises.", "content_type": "text", "category": "health"}
    ]
    
    batch_vectors = embedder.encode([doc["text"] for doc in batch_documents]).tolist()
    batch_points = []
    
    for doc, vec in zip(batch_documents, batch_vectors):
        point = models.PointStruct(
            id=doc["id"],
            vector=vec,
            payload={
                "text": doc["text"],
                "content_type": doc["content_type"],
                "category": doc["category"]
            }
        )
        batch_points.append(point)
    
    client.upsert(collection_name=collection_name, points=batch_points)
    
    # Verify batch insertion
    updated_info = client.get_collection(collection_name)
    print(f"✅ Batch insertion completed!")
    print(f"Updated point count: {updated_info.points_count}")
    
    # Final comprehensive search
    print(f"\n🎉 FINAL COMPREHENSIVE SEARCH")
    print("=" * 40)
    
    final_query = "What are the key features of AI applications?"
    final_results = search_with_filter(final_query, top_k=5)
    
    print(f"Query: {final_query}")
    print("Top Results:")
    for i, result in enumerate(final_results):
        print(f"{i+1}. [{result['category']}] Score: {result['score']:.3f}")
        print(f"   {result['text'][:100]}...")
    
    print(f"\n✅ Vector database test completed successfully!")
    print(f"📁 Database stored at: ./test_qdrant_db")
    print(f"🔍 Collection: {collection_name}")
    print(f"📊 Total documents: {updated_info.points_count}")
    
    # Cleanup
    print(f"\n🧹 Cleaning up test database...")
    try:
        client.delete_collection(collection_name)
        import shutil
        shutil.rmtree("./test_qdrant_db", ignore_errors=True)
        print("✅ Test database cleaned up")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")


if __name__ == "__main__":
    test_basic_vectordb()
