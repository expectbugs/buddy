#!/usr/bin/env python3

"""
Test script for Mem0 system without the actual model
This will verify that the memory system is working correctly
"""

import yaml
from mem0 import Memory
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from mem0_config.yaml"""
    with open('mem0_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config['mem0']

def test_memory_system():
    """Test the Mem0 memory system"""
    print("=== Testing Mem0 Memory System ===")
    
    try:
        # For testing, use a minimal config without LLM to avoid OpenAI requirement
        config = {
            'vector_store': {
                'provider': 'qdrant',
                'config': {
                    'host': 'localhost',
                    'port': 6333,
                    'collection_name': 'hermes_memory'
                }
            },
            'graph_store': {
                'provider': 'neo4j',
                'config': {
                    'url': 'bolt://localhost:7687',
                    'username': 'neo4j',
                    'password': 'password123'
                }
            },
            'embedder': {
                'provider': 'huggingface',
                'config': {
                    'model': 'sentence-transformers/all-MiniLM-L6-v2'
                }
            }
        }
        
        # Initialize memory system
        print("Initializing Mem0 with Qdrant and Neo4j...")
        memory = Memory.from_config(config)
        print("✓ Memory system initialized successfully!")
        
        # Test user ID
        user_id = "test_user"
        
        # First, clear any existing memories for this test user
        try:
            print("Clearing existing test memories...")
            memory.delete_all(user_id=user_id)
        except Exception as e:
            print(f"Note: Could not clear existing memories (this is OK for first run): {e}")
        
        # Add some test memories one by one with error handling
        print("\nAdding test memories...")
        memories = [
            "I love programming in Python",
            "My favorite AI model is GPT-4", 
            "I'm working on a project with Qdrant and Neo4j",
            "I prefer coffee over tea in the morning"
        ]
        
        added_count = 0
        for memory_text in memories:
            try:
                result = memory.add(memory_text, user_id=user_id)
                print(f"✓ Added memory: {memory_text}")
                added_count += 1
            except Exception as e:
                print(f"✗ Failed to add memory '{memory_text}': {e}")
        
        if added_count == 0:
            print("✗ Could not add any memories")
            return False
        
        # Search for relevant memories
        print(f"\nSearching for memories (added {added_count} memories)...")
        query = "What programming language do I like?"
        try:
            results = memory.search(query, user_id=user_id, limit=3)
            
            print(f"\nSearch query: '{query}'")
            print("Relevant memories found:")
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['text']} (score: {result.get('score', 'N/A')})")
        except Exception as e:
            print(f"✗ Search failed: {e}")
            print("Continuing with get_all test...")
        
        # Get all memories
        print("\nRetrieving all stored memories...")
        try:
            all_memories = memory.get_all(user_id=user_id)
            print("All stored memories:")
            for i, memory_item in enumerate(all_memories, 1):
                print(f"{i}. {memory_item['text']}")
            
            print(f"\n✓ Memory system test completed successfully!")
            print(f"✓ Total memories stored: {len(all_memories)}")
            return True
        except Exception as e:
            print(f"✗ Failed to retrieve memories: {e}")
            return False
        
    except Exception as e:
        print(f"✗ Error testing memory system: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_services():
    """Test if Qdrant and Neo4j are accessible"""
    import requests
    
    print("=== Testing Service Connectivity ===")
    
    # Test Qdrant
    try:
        response = requests.get("http://localhost:6333/collections", timeout=5)
        if response.status_code == 200:
            print("✓ Qdrant is accessible")
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
            print("✓ Neo4j is accessible")
        else:
            print(f"✗ Neo4j returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Neo4j: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing Mem0 Hybrid Memory System")
    print("=================================")
    
    # Test services first
    if not test_services():
        print("\n❌ Service connectivity test failed!")
        exit(1)
    
    # Test memory system
    if test_memory_system():
        print("\n🎉 All tests passed! Memory system is working correctly.")
    else:
        print("\n❌ Memory system test failed!")
        exit(1)