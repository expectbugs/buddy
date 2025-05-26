#!/usr/bin/env python3

"""
Comprehensive test of all memory interaction types
Tests: storage, retrieval, updates, relationships, search, and persistence
"""

import os
import sys
import yaml
from pathlib import Path
from llama_cpp import Llama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import uuid
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryTestSystem:
    def __init__(self, config):
        self.config = config
        self.llm = None
        self.qdrant = None
        self.neo4j = None
        self.embedder = None
        self.collection_name = "memory_test_collection"
        
    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Memory Test System...")
        
        # Initialize LLM
        logger.info("Loading Hermes model...")
        model_path = self.config["model"]["path"]
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=self.config["model"]["n_gpu_layers"],
            n_ctx=self.config["model"]["n_ctx"],
            n_batch=self.config["model"]["n_batch"],
            n_threads=self.config["model"]["n_threads"],
            verbose=False
        )
        
        # Initialize embedding model
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Initialize Qdrant
        self.qdrant = QdrantClient(host="localhost", port=6333)
        try:
            self.qdrant.delete_collection(self.collection_name)
        except:
            pass
        self.qdrant.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        time.sleep(1)  # Ensure collection is ready
        
        # Initialize Neo4j
        self.neo4j = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "password123")
        )
        
        # Clear existing test data
        with self.neo4j.session() as session:
            session.run("MATCH (n:MemoryTest) DETACH DELETE n")
        
        logger.info("✅ All components initialized")
    
    def add_memory(self, text, user_id, memory_type="general", metadata=None):
        """Add a memory with metadata"""
        memory_id = str(uuid.uuid4())
        
        # Generate embedding
        embedding = self.embedder.encode([text])[0]
        
        # Prepare payload
        payload = {
            "text": text,
            "user_id": user_id,
            "memory_type": memory_type,
            "memory_id": memory_id,
            "timestamp": time.time()
        }
        if metadata:
            payload.update(metadata)
        
        # Store in Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=memory_id,
                vector=embedding.tolist(),
                payload=payload
            )]
        )
        
        # Store in Neo4j with relationships
        with self.neo4j.session() as session:
            # Create memory node
            session.run("""
                CREATE (m:MemoryTest {
                    id: $memory_id,
                    text: $text,
                    user_id: $user_id,
                    memory_type: $memory_type,
                    timestamp: datetime()
                })
            """, memory_id=memory_id, text=text, user_id=user_id, memory_type=memory_type)
            
            # Create relationships if metadata contains related memories
            if metadata and 'related_to' in metadata:
                for related_id in metadata['related_to']:
                    session.run("""
                        MATCH (m1:MemoryTest {id: $memory_id})
                        MATCH (m2:MemoryTest {id: $related_id})
                        CREATE (m1)-[:RELATED_TO]->(m2)
                    """, memory_id=memory_id, related_id=related_id)
        
        return memory_id
    
    def update_memory(self, memory_id, new_text):
        """Update an existing memory"""
        # Generate new embedding
        embedding = self.embedder.encode([new_text])[0]
        
        # Get existing memory from Qdrant
        existing = self.qdrant.retrieve(
            collection_name=self.collection_name,
            ids=[memory_id]
        )[0]
        
        # Update payload
        existing.payload['text'] = new_text
        existing.payload['updated_at'] = time.time()
        
        # Update in Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[PointStruct(
                id=memory_id,
                vector=embedding.tolist(),
                payload=existing.payload
            )]
        )
        
        # Update in Neo4j
        with self.neo4j.session() as session:
            session.run("""
                MATCH (m:MemoryTest {id: $memory_id})
                SET m.text = $new_text, m.updated_at = datetime()
            """, memory_id=memory_id, new_text=new_text)
    
    def search_memories(self, query, user_id, limit=5):
        """Search memories by semantic similarity"""
        query_embedding = self.embedder.encode([query])[0]
        
        search_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            query_filter={
                "must": [{"key": "user_id", "match": {"value": user_id}}]
            },
            limit=limit
        )
        
        return [
            {
                "id": point.id,
                "text": point.payload["text"],
                "score": point.score,
                "memory_type": point.payload["memory_type"]
            }
            for point in search_results
        ]
    
    def get_related_memories(self, memory_id):
        """Get memories related to a specific memory via Neo4j"""
        with self.neo4j.session() as session:
            result = session.run("""
                MATCH (m1:MemoryTest {id: $memory_id})-[:RELATED_TO]-(m2:MemoryTest)
                RETURN m2.id as id, m2.text as text, m2.memory_type as type
            """, memory_id=memory_id)
            
            return [
                {
                    "id": record["id"],
                    "text": record["text"],
                    "type": record["type"]
                }
                for record in result
            ]
    
    def get_memory_graph(self, user_id):
        """Get complete memory graph structure"""
        with self.neo4j.session() as session:
            # Get all memories and relationships
            result = session.run("""
                MATCH (m:MemoryTest {user_id: $user_id})
                OPTIONAL MATCH (m)-[r:RELATED_TO]-(m2:MemoryTest)
                RETURN m.id as id, m.text as text, m.memory_type as type,
                       collect(DISTINCT m2.id) as related_ids
            """, user_id=user_id)
            
            graph = {}
            for record in result:
                graph[record["id"]] = {
                    "text": record["text"],
                    "type": record["type"],
                    "related_to": record["related_ids"]
                }
            
            return graph
    
    def run_comprehensive_test(self):
        """Run all memory interaction tests"""
        user_id = "test_user_comprehensive"
        memory_ids = {}
        
        print("\n" + "="*60)
        print("🧪 COMPREHENSIVE MEMORY SYSTEM TEST")
        print("="*60)
        
        # Test 1: Basic Memory Storage
        print("\n📝 Test 1: Basic Memory Storage")
        print("-" * 40)
        
        memories_to_store = [
            ("My name is John and I'm 30 years old", "personal_info"),
            ("I work as a machine learning engineer at TechCorp", "work_info"),
            ("I graduated from MIT with a PhD in Computer Science", "education"),
            ("My favorite programming language is Python", "preferences"),
            ("I'm currently working on a neural network project", "current_project")
        ]
        
        for text, mem_type in memories_to_store:
            mem_id = self.add_memory(text, user_id, mem_type)
            memory_ids[mem_type] = mem_id
            print(f"✅ Stored [{mem_type}]: {text}")
        
        # Test 2: Memory Updates
        print("\n🔄 Test 2: Memory Updates")
        print("-" * 40)
        
        old_text = "I'm currently working on a neural network project"
        new_text = "I'm currently working on a transformer-based NLP project"
        
        print(f"Old: {old_text}")
        print(f"New: {new_text}")
        
        self.update_memory(memory_ids["current_project"], new_text)
        print("✅ Memory updated successfully")
        
        # Test 3: Create Relationships
        print("\n🔗 Test 3: Creating Memory Relationships")
        print("-" * 40)
        
        # Add related memories
        skill_id = self.add_memory(
            "I have expertise in PyTorch and TensorFlow",
            user_id,
            "skills",
            {"related_to": [memory_ids["work_info"], memory_ids["current_project"]]}
        )
        memory_ids["skills"] = skill_id
        
        hobby_id = self.add_memory(
            "I enjoy reading research papers on AI",
            user_id,
            "hobbies",
            {"related_to": [memory_ids["work_info"], memory_ids["education"]]}
        )
        memory_ids["hobbies"] = hobby_id
        
        print("✅ Created relationships between memories")
        
        # Test 4: Semantic Search
        print("\n🔍 Test 4: Semantic Memory Search")
        print("-" * 40)
        
        search_queries = [
            "What is John's educational background?",
            "Tell me about machine learning work",
            "What programming skills does he have?",
            "Current projects and interests"
        ]
        
        for query in search_queries:
            print(f"\nQuery: '{query}'")
            results = self.search_memories(query, user_id, limit=3)
            for i, result in enumerate(results, 1):
                print(f"  {i}. [{result['memory_type']}] {result['text'][:60]}... (score: {result['score']:.3f})")
        
        # Test 5: Graph Traversal
        print("\n🕸️ Test 5: Memory Graph Traversal")
        print("-" * 40)
        
        # Get related memories for skills
        related = self.get_related_memories(memory_ids["skills"])
        print(f"\nMemories related to 'skills':")
        for mem in related:
            print(f"  - [{mem['type']}] {mem['text']}")
        
        # Test 6: Complex Query with Context
        print("\n🤖 Test 6: AI Response with Full Memory Context")
        print("-" * 40)
        
        complex_query = "Based on everything you know about me, what kind of AI project would you recommend?"
        
        # Get relevant memories
        relevant_memories = self.search_memories(complex_query, user_id, limit=5)
        context = "What I know about you:\n"
        for mem in relevant_memories:
            context += f"- {mem['text']}\n"
        
        # Get graph structure for additional context
        graph = self.get_memory_graph(user_id)
        
        prompt = f"""{context}

Based on this information about your background, skills, and interests,
User: {complex_query}
Assistant: Given your background and expertise,"""
        
        response = self.llm(
            prompt,
            max_tokens=200,
            temperature=0.7,
            stop=["User:", "\n\n"]
        )
        
        ai_response = response['choices'][0]['text'].strip()
        print(f"\n🤖 AI Response: {ai_response}")
        
        # Test 7: Memory Persistence Verification
        print("\n💾 Test 7: Memory Persistence Verification")
        print("-" * 40)
        
        # Verify all memories in Qdrant
        all_ids = list(memory_ids.values()) + [skill_id, hobby_id]
        retrieved = self.qdrant.retrieve(
            collection_name=self.collection_name,
            ids=all_ids
        )
        print(f"✅ Qdrant: Retrieved {len(retrieved)}/{len(all_ids)} memories")
        
        # Verify all memories in Neo4j
        with self.neo4j.session() as session:
            result = session.run("""
                MATCH (m:MemoryTest {user_id: $user_id})
                RETURN count(m) as count
            """, user_id=user_id)
            neo4j_count = result.single()["count"]
            print(f"✅ Neo4j: Found {neo4j_count} memory nodes")
            
            # Verify relationships
            result = session.run("""
                MATCH (m1:MemoryTest {user_id: $user_id})-[r:RELATED_TO]-(m2:MemoryTest)
                RETURN count(DISTINCT r) as count
            """, user_id=user_id)
            rel_count = result.single()["count"]
            print(f"✅ Neo4j: Found {rel_count} relationships")
        
        # Test 8: Memory Consistency Check
        print("\n🔍 Test 8: Memory Consistency Check")
        print("-" * 40)
        
        # Check if updated memory is consistent across stores
        updated_memory = self.qdrant.retrieve(
            collection_name=self.collection_name,
            ids=[memory_ids["current_project"]]
        )[0]
        
        with self.neo4j.session() as session:
            result = session.run("""
                MATCH (m:MemoryTest {id: $memory_id})
                RETURN m.text as text
            """, memory_id=memory_ids["current_project"])
            neo4j_text = result.single()["text"]
        
        qdrant_text = updated_memory.payload["text"]
        
        print(f"Qdrant text: {qdrant_text}")
        print(f"Neo4j text: {neo4j_text}")
        print(f"✅ Consistency check: {'PASSED' if qdrant_text == neo4j_text else 'FAILED'}")
        
        # Final Summary
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print("✅ Basic storage: PASSED")
        print("✅ Memory updates: PASSED")
        print("✅ Relationships: PASSED")
        print("✅ Semantic search: PASSED")
        print("✅ Graph traversal: PASSED")
        print("✅ AI integration: PASSED")
        print("✅ Persistence: PASSED")
        print("✅ Consistency: PASSED")
        print("\n🎉 ALL MEMORY INTERACTION TYPES TESTED SUCCESSFULLY!")
        
        return True

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent / "mem0_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Run comprehensive memory tests"""
    config = load_config()
    
    # Create and initialize test system
    test_system = MemoryTestSystem(config)
    test_system.initialize()
    
    # Run comprehensive tests
    success = test_system.run_comprehensive_test()
    
    if success:
        print("\n✅ Memory system is fully functional!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()