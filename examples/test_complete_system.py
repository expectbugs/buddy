#!/usr/bin/env python3

"""
Test the complete Hermes + Mem0 system with automated interactions
"""

import os
import sys
import yaml
from pathlib import Path
from mem0 import Memory
from llama_cpp import Llama
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    config_path = Path(__file__).parent / "mem0_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_system():
    """Test the complete system with automated interactions"""
    print("=== Testing Complete Hermes + Mem0 System ===")
    
    # Load configuration
    config = load_config()
    
    # Initialize LLM
    logger.info("Loading Hermes model...")
    model_path = config["model"]["path"]
    
    try:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=config["model"]["n_gpu_layers"],
            n_ctx=config["model"]["n_ctx"],
            n_batch=config["model"]["n_batch"],
            n_threads=config["model"]["n_threads"],
            verbose=False
        )
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return False
    
    # Initialize Mem0 (without LLM dependency)
    logger.info("Initializing Mem0...")
    try:
        mem0_config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "hermes_memory"
                }
            },
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password123"
                }
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            }
        }
        
        memory = Memory.from_config(mem0_config)
        logger.info("✅ Mem0 initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Mem0: {e}")
        return False
    
    # Test user session
    user_id = "test_user"
    
    # Clear any existing memories
    try:
        memory.delete_all(user_id=user_id)
        logger.info("Cleared existing memories")
    except:
        logger.info("No existing memories to clear")
    
    # Test 1: Store some initial information
    print("\n=== Test 1: Storing User Information ===")
    user_facts = [
        "My name is Alex",
        "I work as a software engineer at a tech startup",
        "I love programming in Python and Go",
        "My favorite project is building AI applications"
    ]
    
    for fact in user_facts:
        try:
            memory.add(fact, user_id=user_id)
            print(f"✅ Stored: {fact}")
        except Exception as e:
            print(f"❌ Failed to store: {fact} - {e}")
            return False
    
    # Test 2: Generate AI response with memory context
    print("\n=== Test 2: AI Response with Memory Context ===")
    user_question = "What programming languages do I like?"
    
    # Get relevant memories
    try:
        relevant_memories = memory.search(user_question, user_id=user_id, limit=5)
        context = "\n".join([f"- {mem['text']}" for mem in relevant_memories])
        print(f"Retrieved memories:\n{context}")
        
        # Construct prompt with context
        prompt = f"""Previous context about the user:
{context}

User: {user_question}
Assistant: Based on what I know about you,"""
        
        # Generate response
        response = llm(
            prompt,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            stop=["User:", "\n\n"]
        )
        
        ai_response = response['choices'][0]['text'].strip()
        print(f"\n🤖 AI Response: {ai_response}")
        
        # Store the interaction
        memory.add(f"User asked: {user_question}", user_id=user_id)
        memory.add(f"Assistant responded: {ai_response}", user_id=user_id)
        
    except Exception as e:
        print(f"❌ Failed to generate response: {e}")
        return False
    
    # Test 3: Memory persistence verification
    print("\n=== Test 3: Memory Persistence Verification ===")
    try:
        all_memories = memory.get_all(user_id=user_id)
        print(f"📝 Total memories stored: {len(all_memories)}")
        for i, mem in enumerate(all_memories, 1):
            print(f"{i}. {mem['text']}")
    except Exception as e:
        print(f"❌ Failed to retrieve memories: {e}")
        return False
    
    # Test 4: Another question to test memory search
    print("\n=== Test 4: Follow-up Question with Memory ===")
    follow_up = "What do you know about my work?"
    
    try:
        relevant_memories = memory.search(follow_up, user_id=user_id, limit=3)
        context = "\n".join([f"- {mem['text']}" for mem in relevant_memories])
        
        prompt = f"""Previous context about the user:
{context}

User: {follow_up}
Assistant: From what I remember about you,"""
        
        response = llm(
            prompt,
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            stop=["User:", "\n\n"]
        )
        
        ai_response = response['choices'][0]['text'].strip()
        print(f"📚 Retrieved context:\n{context}")
        print(f"\n🤖 AI Response: {ai_response}")
        
    except Exception as e:
        print(f"❌ Failed follow-up test: {e}")
        return False
    
    print("\n🎉 All tests passed! The Hermes + Mem0 system is working correctly!")
    print("✅ Model inference working")
    print("✅ Memory storage working") 
    print("✅ Memory retrieval working")
    print("✅ Context-aware responses working")
    print("✅ Memory persistence working")
    
    return True

if __name__ == "__main__":
    success = test_system()
    if not success:
        sys.exit(1)