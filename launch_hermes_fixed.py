#!/usr/bin/env python3

"""
Fixed Hermes-2-Pro-Mistral-10.7B Launcher with Local Memory System
No OpenAI dependencies - uses local model for everything
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path
from dotenv import load_dotenv
from llama_cpp import Llama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import uuid
import time
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalMemorySystem:
    """Memory system that works entirely locally without OpenAI"""
    
    def __init__(self, config):
        self.config = config
        self.llm = None
        self.qdrant = None
        self.neo4j = None
        self.embedder = None
        self.collection_name = "hermes_local_memory"
        self.user_id = "default_user"
        
    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Hermes with Local Memory System...")
        
        # Initialize LLM
        logger.info("Loading Hermes model...")
        model_path = self.config["model"]["path"]
        
        # Check if model exists
        if not Path(model_path).exists():
            logger.error(f"Model file not found at: {model_path}")
            sys.exit(1)
            
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=self.config["model"]["n_gpu_layers"],
            n_ctx=self.config["model"]["n_ctx"],
            n_batch=self.config["model"]["n_batch"],
            n_threads=self.config["model"]["n_threads"],
            verbose=False
        )
        logger.info("✅ Model loaded successfully")
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        logger.info("✅ Embedding model loaded")
        
        # Initialize Qdrant
        try:
            logger.info("Connecting to Qdrant...")
            qdrant_host = os.getenv("QDRANT_HOST", "localhost")
            qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
            self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
            
            # Check if collection exists, create if not
            collections = self.qdrant.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name}")
                self.qdrant.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
            time.sleep(1)  # Ensure collection is ready
            logger.info("✅ Qdrant initialized")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            logger.error("Please ensure Qdrant is running: sudo rc-service qdrant start")
            sys.exit(1)
        
        # Initialize Neo4j
        try:
            logger.info("Connecting to Neo4j...")
            neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
            neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")
            neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            self.neo4j = GraphDatabase.driver(
                neo4j_uri,
                auth=(neo4j_username, neo4j_password)
            )
            
            # Test connection (but don't delete existing data!)
            with self.neo4j.session() as session:
                result = session.run("MATCH (n:HermesMemory) RETURN count(n) as count")
                count = result.single()["count"]
                logger.info(f"Found {count} existing memories in Neo4j")
            logger.info("✅ Neo4j initialized")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            logger.error("Please ensure Neo4j is running: sudo rc-service neo4j start")
            sys.exit(1)
    
    def add_memory(self, text, memory_type="conversation"):
        """Add a memory to both stores"""
        memory_id = str(uuid.uuid4())
        timestamp = time.time()
        
        try:
            # Generate embedding
            embedding = self.embedder.encode([text])[0]
            
            # Store in Qdrant
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=memory_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": text,
                        "user_id": self.user_id,
                        "memory_type": memory_type,
                        "timestamp": timestamp
                    }
                )]
            )
            
            # Store in Neo4j
            with self.neo4j.session() as session:
                session.run("""
                    CREATE (m:HermesMemory {
                        id: $memory_id,
                        text: $text,
                        user_id: $user_id,
                        memory_type: $memory_type,
                        timestamp: $timestamp
                    })
                """, memory_id=memory_id, text=text, user_id=self.user_id, 
                     memory_type=memory_type, timestamp=timestamp)
            
            return memory_id
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return None
    
    def search_memories(self, query, limit=5):
        """Search for relevant memories"""
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query])[0]
            
            # Search in Qdrant
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter={
                    "must": [{"key": "user_id", "match": {"value": self.user_id}}]
                },
                limit=limit
            )
            
            return [
                {
                    "text": point.payload["text"],
                    "score": point.score,
                    "memory_type": point.payload.get("memory_type", "unknown")
                }
                for point in search_results
            ]
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    def get_all_memories(self):
        """Get all memories from Neo4j"""
        try:
            with self.neo4j.session() as session:
                result = session.run("""
                    MATCH (m:HermesMemory {user_id: $user_id})
                    RETURN m.text as text, m.memory_type as type, m.timestamp as timestamp
                    ORDER BY m.timestamp DESC
                """, user_id=self.user_id)
                
                return [
                    {
                        "text": record["text"],
                        "type": record["type"],
                        "timestamp": record["timestamp"]
                    }
                    for record in result
                ]
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []
    
    def clear_memories(self):
        """Clear all memories for current user"""
        try:
            # Clear from Neo4j
            with self.neo4j.session() as session:
                result = session.run("""
                    MATCH (m:HermesMemory {user_id: $user_id})
                    DETACH DELETE m
                    RETURN count(m) as deleted_count
                """, user_id=self.user_id)
                count = result.single()["deleted_count"]
                
            # Clear from Qdrant - recreate collection
            self.qdrant.delete_collection(self.collection_name)
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            
            logger.info(f"Cleared {count} memories")
            return count
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return 0
    
    def chat(self):
        """Interactive chat with memory"""
        print("\n" + "="*60)
        print("🤖 Hermes-2-Pro-Mistral with Local Memory System")
        print("="*60)
        print("\nCommands:")
        print("  /memory  - View all stored memories")
        print("  /clear   - Clear all memories")
        print("  /exit    - Exit the chat")
        print("\nStart chatting! The AI will remember your conversation.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == '/exit':
                    print("Goodbye!")
                    break
                
                if user_input.lower() == '/memory':
                    memories = self.get_all_memories()
                    print(f"\n📚 Stored Memories ({len(memories)} total):")
                    if memories:
                        for i, mem in enumerate(memories, 1):
                            print(f"{i}. {mem['text']}")
                    else:
                        print("No memories stored yet.")
                    print()
                    continue
                
                if user_input.lower() == '/clear':
                    count = self.clear_memories()
                    print(f"✅ Cleared {count} memories\n")
                    continue
                
                # Store user input in memory
                self.add_memory(f"User: {user_input}", "user_message")
                
                # Search for relevant memories - increase limit for better context
                relevant_memories = self.search_memories(user_input, limit=10)
                
                # Build context from memories
                context = ""
                if relevant_memories:
                    context = "Based on our previous conversations:\n"
                    seen_texts = set()  # Avoid duplicates
                    for mem in relevant_memories:
                        # Don't include the exact same message and avoid duplicates
                        if mem['text'] != f"User: {user_input}" and mem['text'] not in seen_texts:
                            context += f"- {mem['text']}\n"
                            seen_texts.add(mem['text'])
                    
                    # Also get ALL user facts for better context
                    all_memories = self.get_all_memories()
                    important_facts = []
                    for mem in all_memories:
                        text = mem['text'].lower()
                        # Extract important facts about relationships and preferences
                        if any(keyword in text for keyword in ['name is', 'i am', 'my brother', 'my cousin', 'my sister', 
                                                               'favorite color', 'i work', 'i like', 'i have']):
                            if mem['text'] not in seen_texts:
                                important_facts.append(mem['text'])
                                seen_texts.add(mem['text'])
                    
                    if important_facts:
                        context += "\nImportant facts:\n"
                        for fact in important_facts[-5:]:  # Last 5 important facts
                            context += f"- {fact}\n"
                    
                    context += "\n"
                
                # Construct prompt with system context
                system_prompt = """You are Buddy, a helpful AI assistant with a persistent memory system. 
You remember all conversations and can recall information from previous interactions.
Always identify yourself as Buddy when asked who you are.
"""
                
                prompt = f"""{system_prompt}

{context}User: {user_input}
Assistant: """
                
                # Generate response
                response = self.llm(
                    prompt,
                    max_tokens=200,
                    temperature=self.config["model"]["temperature"],
                    top_p=self.config["model"]["top_p"],
                    repeat_penalty=self.config["model"]["repeat_penalty"],
                    stop=["User:", "\n\n"]
                )
                
                assistant_response = response['choices'][0]['text'].strip()
                print(f"Assistant: {assistant_response}\n")
                
                # Store assistant response
                self.add_memory(f"Assistant: {assistant_response}", "assistant_message")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /exit to quit.\n")
                continue
            except Exception as e:
                logger.error(f"Error during chat: {e}")
                print(f"Sorry, an error occurred: {e}\n")
                continue

def check_services():
    """Check if required services are running"""
    services_ok = True
    
    # Check Qdrant
    try:
        result = subprocess.run(['rc-service', 'qdrant', 'status'], 
                              capture_output=True, text=True)
        if 'started' not in result.stdout and 'running' not in result.stdout.lower():
            logger.warning("Qdrant service not running")
            services_ok = False
    except:
        logger.error("Could not check Qdrant status")
        services_ok = False
    
    # Check Neo4j
    try:
        result = subprocess.run(['rc-service', 'neo4j', 'status'], 
                              capture_output=True, text=True)
        if 'running' not in result.stdout.lower() and 'started' not in result.stdout:
            logger.warning("Neo4j service not running")
            services_ok = False
    except:
        logger.error("Could not check Neo4j status")
        services_ok = False
    
    if not services_ok:
        print("\n⚠️  Required services not running!")
        print("Please start them with:")
        print("  sudo rc-service qdrant start")
        print("  sudo rc-service neo4j start\n")
        return False
    
    return True

def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent / "mem0_config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default config with environment variable support
        model_path = os.getenv("MODEL_PATH", 
                              os.path.expanduser("~/models/Hermes-2-Pro-Mistral-10.7B-Q6_K/hermes-2-pro-mistral-10.7b.Q6_K.gguf"))
        return {
            "model": {
                "path": model_path,
                "n_gpu_layers": 99,
                "n_ctx": 8192,
                "n_batch": 512,
                "n_threads": 8,
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "max_tokens": 2048
            }
        }

def main():
    """Main entry point"""
    # Check services
    if not check_services():
        sys.exit(1)
    
    # Load config
    config = load_config()
    
    # Create and initialize system
    system = LocalMemorySystem(config)
    system.initialize()
    
    # Start chat
    system.chat()
    
    # Cleanup
    system.neo4j.close()

if __name__ == "__main__":
    main()