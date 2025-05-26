#!/usr/bin/env python3

"""
Simple test of just the vector and graph stores without full mem0 Memory class
This will verify connectivity and basic functionality
"""

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

def test_qdrant():
    """Test Qdrant vector store directly"""
    print("=== Testing Qdrant Vector Store ===")
    
    try:
        # Initialize client
        client = QdrantClient(host="localhost", port=6333)
        
        # Test connection
        client.get_collections()
        print("✓ Connected to Qdrant")
        
        # Create a test collection
        collection_name = "test_collection"
        try:
            client.delete_collection(collection_name)
        except:
            pass  # Collection might not exist
        
        # Create collection with correct vector size for sentence-transformers
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print("✓ Created test collection")
        
        # Test embedding model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        test_text = "This is a test sentence"
        embedding = model.encode([test_text])[0]
        print(f"✓ Generated embedding (size: {len(embedding)})")
        
        # Add a test point
        client.upsert(
            collection_name=collection_name,
            points=[{
                "id": 1,
                "vector": embedding.tolist(),
                "payload": {"text": test_text, "user_id": "test_user"}
            }]
        )
        print("✓ Added test vector to Qdrant")
        
        # Search for similar vectors
        search_results = client.search(
            collection_name=collection_name,
            query_vector=embedding.tolist(),
            limit=1
        )
        
        if search_results:
            print(f"✓ Retrieved similar vector: {search_results[0].payload['text']}")
        
        # Cleanup
        client.delete_collection(collection_name)
        print("✓ Qdrant test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Qdrant test failed: {e}")
        return False

def test_neo4j():
    """Test Neo4j graph store directly"""
    print("\n=== Testing Neo4j Graph Store ===")
    
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password123")
        )
        
        # Test connection
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            test_value = result.single()["test"]
            
        if test_value == 1:
            print("✓ Connected to Neo4j")
        
        # Test basic graph operations
        with driver.session() as session:
            # Clear any existing test data
            session.run("MATCH (n:TestMemory) DELETE n")
            
            # Create test nodes
            session.run("""
                CREATE (m1:TestMemory {text: 'I love programming', user_id: 'test_user'})
                CREATE (m2:TestMemory {text: 'Python is my favorite language', user_id: 'test_user'})
                CREATE (m1)-[:RELATED]->(m2)
            """)
            print("✓ Created test nodes and relationship")
            
            # Query test data
            result = session.run("""
                MATCH (m:TestMemory {user_id: 'test_user'})
                RETURN m.text as text
            """)
            
            memories = [record["text"] for record in result]
            print(f"✓ Retrieved {len(memories)} memories from Neo4j")
            for memory in memories:
                print(f"  - {memory}")
            
            # Cleanup
            session.run("MATCH (n:TestMemory) DELETE n")
            print("✓ Neo4j test completed successfully")
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"✗ Neo4j test failed: {e}")
        return False

def test_services():
    """Test if services are accessible via HTTP"""
    print("=== Testing Service HTTP Endpoints ===")
    
    # Test Qdrant
    try:
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            print("✓ Qdrant HTTP endpoint accessible")
        else:
            print(f"✗ Qdrant returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Qdrant: {e}")
        return False
    
    # Test Neo4j
    try:
        response = requests.get("http://localhost:7474", timeout=5)
        if response.status_code == 200:
            print("✓ Neo4j HTTP endpoint accessible")
        else:
            print(f"✗ Neo4j returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Neo4j: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Simple Mem0 Components Test")
    print("===========================")
    
    all_passed = True
    
    # Test service connectivity
    if not test_services():
        print("\n❌ Service connectivity failed!")
        all_passed = False
    
    # Test Qdrant
    if not test_qdrant():
        all_passed = False
    
    # Test Neo4j
    if not test_neo4j():
        all_passed = False
    
    if all_passed:
        print("\n🎉 All component tests passed!")
        print("✓ Qdrant vector store is working")
        print("✓ Neo4j graph store is working") 
        print("✓ Sentence transformers embedding model is working")
        print("\nThe hybrid memory infrastructure is ready!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)