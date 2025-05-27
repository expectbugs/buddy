#!/usr/bin/env python3

"""
Buddy v0.4.0-dev - AI Assistant with Memory Consolidation and Smart Context
Phase 2 Implementation: Memory summarization, episode detection, and intelligent
context expansion for efficient long-term memory management
"""

import os
import sys
import subprocess
import yaml
import json
import asyncio
import signal
import atexit
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from llama_cpp import Llama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import uuid
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
from logging.handlers import RotatingFileHandler
import numpy as np

# Import the memory summarizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_summarizer import MemorySummarizer

# Load environment variables
load_dotenv()

# Configure logging with rotation
log_dir = Path("/var/log/buddy")
log_dir.mkdir(parents=True, exist_ok=True)

# Main application logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            log_dir / "buddy.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
    ]
)
logger = logging.getLogger(__name__)

# Interaction logger for append-only log
interaction_logger = logging.getLogger('interactions')
interaction_logger.setLevel(logging.INFO)
interaction_handler = RotatingFileHandler(
    log_dir / "interactions.jsonl",
    maxBytes=50*1024*1024,  # 50MB
    backupCount=10
)
interaction_handler.setFormatter(logging.Formatter('%(message)s'))
interaction_logger.addHandler(interaction_handler)
interaction_logger.propagate = False


class MemoryType(Enum):
    """Types of memories with different handling strategies"""
    PERSONAL_FACT = "personal_fact"
    PREFERENCE = "preference"
    RELATIONSHIP = "relationship"
    PROJECT = "project"
    TECHNICAL = "technical"
    CONVERSATION = "conversation"
    TEMPORARY = "temporary"
    SUMMARY = "summary"
    EPISODE = "episode"


class MemoryOperation(Enum):
    """Operations that can be performed on memories"""
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    SKIP = "skip"


@dataclass
class MemoryCandidate:
    """Candidate memory for processing"""
    text: str
    memory_type: MemoryType
    priority: float  # 0.0 to 1.0
    metadata: Dict
    related_memory_id: Optional[str] = None


class RobustMemorySystem:
    """Enhanced memory system with robust persistence and graceful shutdown"""
    
    def __init__(self, config):
        self.config = config
        self.llm = None
        self.qdrant = None
        self.neo4j = None
        self.neo4j_pool = None  # Connection pool
        self.embedder = None
        self.collection_name = "hermes_advanced_memory"
        self.user_id = "default_user"
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.memory_queue = queue.Queue()
        self.processing_lock = threading.Lock()
        self.memory_thread = None
        self.shutdown_event = threading.Event()
        
        # Memory filtering criteria
        self.min_priority_threshold = 0.3
        self.trivial_patterns = [
            "hello", "hi", "hey", "goodbye", "bye", "thanks", "thank you",
            "ok", "okay", "sure", "yes", "no", "maybe"
        ]
        
        # Episode tracking
        self.current_episode_id = str(uuid.uuid4())
        self.previous_embedding = None
        self.episode_similarity_threshold = 0.3  # Lower threshold = less sensitive
        self.interaction_count = 0
        self.last_summary_count = 0
        self.summary_threshold = 5  # Summarize every 5 interactions for testing
        
        # Log line tracking
        self.log_line_count = 0
        
        # Memory summarizer (initialized after components)
        self.summarizer = None
        
        # Register shutdown handlers
        self._register_shutdown_handlers()
        
    def _register_shutdown_handlers(self):
        """Register handlers for graceful shutdown"""
        # Handle SIGTERM and SIGINT
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Register atexit handler
        atexit.register(self._cleanup)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        self._cleanup()
        sys.exit(0)
        
    def _cleanup(self):
        """Cleanup resources gracefully"""
        logger.info("Starting cleanup process...")
        
        # Stop memory processing thread
        if self.memory_thread and self.memory_thread.is_alive():
            self.shutdown_event.set()
            # Process remaining items in queue
            logger.info(f"Processing {self.memory_queue.qsize()} remaining memory items...")
            while not self.memory_queue.empty():
                try:
                    self.memory_queue.get_nowait()
                    self.memory_queue.task_done()
                except queue.Empty:
                    break
            self.memory_thread.join(timeout=5)
            
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=False)
            logger.info("ThreadPoolExecutor shut down")
            
        # Close Neo4j connections
        if self.neo4j:
            try:
                self.neo4j.close()
                logger.info("Neo4j connection closed")
            except Exception as e:
                logger.error(f"Error closing Neo4j: {e}")
                
        # Ensure Qdrant data is persisted
        if self.qdrant:
            try:
                # Force a collection update to ensure persistence
                collection_info = self.qdrant.get_collection(self.collection_name)
                logger.info(f"Qdrant collection '{self.collection_name}' has {collection_info.points_count} points")
            except Exception as e:
                logger.error(f"Error checking Qdrant: {e}")
                
        logger.info("Cleanup completed")
        
    def verify_qdrant_persistence(self):
        """Verify Qdrant is configured for persistence"""
        try:
            # Check if we can access collection info
            collections = self.qdrant.get_collections()
            logger.info(f"Qdrant has {len(collections.collections)} collections")
            
            # Create a test point to verify write capability
            test_id = str(uuid.uuid4())
            test_embedding = self.embedder.encode(["persistence test"])[0]
            
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=test_id,
                    vector=test_embedding.tolist(),
                    payload={"test": True, "timestamp": time.time()}
                )]
            )
            
            # Verify it was written
            retrieved = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=[test_id]
            )
            
            if retrieved:
                # Clean up test point
                self.qdrant.delete(
                    collection_name=self.collection_name,
                    points_selector=[test_id]
                )
                logger.info("✅ Qdrant persistence verification successful")
                return True
            else:
                logger.error("❌ Qdrant persistence verification failed")
                return False
                
        except Exception as e:
            logger.error(f"Qdrant persistence check failed: {e}")
            return False
            
    def verify_neo4j_persistence(self):
        """Verify Neo4j is configured for persistence"""
        try:
            with self.neo4j.session() as session:
                # Create test node
                test_id = "persistence_test_" + str(uuid.uuid4())
                result = session.run("""
                    CREATE (n:PersistenceTest {id: $id, timestamp: $timestamp})
                    RETURN n.id as id
                """, id=test_id, timestamp=time.time())
                
                created_id = result.single()["id"]
                
                # Verify it exists
                result = session.run("""
                    MATCH (n:PersistenceTest {id: $id})
                    DELETE n
                    RETURN count(n) as count
                """, id=test_id)
                
                count = result.single()["count"]
                
                if count == 1:
                    logger.info("✅ Neo4j persistence verification successful")
                    return True
                else:
                    logger.error("❌ Neo4j persistence verification failed")
                    return False
                    
        except Exception as e:
            logger.error(f"Neo4j persistence check failed: {e}")
            return False
    
    def initialize(self):
        """Initialize all components with enhanced error handling and verification"""
        logger.info("Initializing Robust Hermes Memory System v0.3.0-dev...")
        
        # Initialize LLM
        logger.info("Loading Hermes model...")
        model_path = self.config["model"]["path"]
        
        if not Path(model_path).exists():
            logger.error(f"Model file not found at: {model_path}")
            sys.exit(1)
            
        # Use CPU if CUDA issues detected
        n_gpu_layers = self.config["model"]["n_gpu_layers"] if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '-1' else 0
        
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
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
        
        # Initialize Qdrant with persistence verification
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                logger.info(f"Connecting to Qdrant (attempt {retry_count + 1}/{max_retries})...")
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
                    collection_info = self.qdrant.get_collection(self.collection_name)
                    logger.info(f"Using existing collection: {self.collection_name} with {collection_info.points_count} points")
                
                # Verify persistence
                if self.verify_qdrant_persistence():
                    logger.info("✅ Qdrant initialized with persistence verified")
                    break
                else:
                    raise Exception("Qdrant persistence verification failed")
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to connect to Qdrant after {max_retries} attempts: {e}")
                    sys.exit(1)
                else:
                    logger.warning(f"Qdrant connection attempt {retry_count} failed, retrying...")
                    time.sleep(2)
        
        # Initialize Neo4j with connection pooling and retry logic
        retry_count = 0
        while retry_count < max_retries:
            try:
                logger.info(f"Connecting to Neo4j (attempt {retry_count + 1}/{max_retries})...")
                neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
                neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")
                neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
                
                # Create driver with connection pooling
                self.neo4j = GraphDatabase.driver(
                    neo4j_uri,
                    auth=(neo4j_username, neo4j_password),
                    max_connection_lifetime=3600,  # 1 hour
                    max_connection_pool_size=50,
                    connection_acquisition_timeout=30.0
                )
                
                # Test connection and count existing memories
                with self.neo4j.session() as session:
                    result = session.run("MATCH (n:HermesMemory) RETURN count(n) as count")
                    count = result.single()["count"]
                    logger.info(f"Found {count} existing memories in Neo4j")
                
                # Verify persistence
                if self.verify_neo4j_persistence():
                    logger.info("✅ Neo4j initialized with persistence verified")
                    break
                else:
                    raise Exception("Neo4j persistence verification failed")
                    
            except Exception as e:
                retry_count += 1
                if retry_count >= max_retries:
                    logger.error(f"Failed to connect to Neo4j after {max_retries} attempts: {e}")
                    sys.exit(1)
                else:
                    logger.warning(f"Neo4j connection attempt {retry_count} failed, retrying...")
                    time.sleep(2)
        
        # Initialize memory summarizer
        logger.info("Initializing memory summarizer...")
        self.summarizer = MemorySummarizer(
            llm=self.llm,
            neo4j=self.neo4j,
            qdrant=self.qdrant,
            embedder=self.embedder,
            collection_name=self.collection_name
        )
        logger.info("✅ Memory summarizer initialized")
    
    def log_interaction(self, user_input: str, assistant_response: str, memory_extracted: List[Dict]):
        """Log interaction to append-only log file with episode tracking"""
        try:
            # Get current log line position
            log_file_path = log_dir / "interactions.jsonl"
            current_line = 0
            if log_file_path.exists():
                with open(log_file_path, 'r') as f:
                    current_line = sum(1 for _ in f)
            
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "episode_id": self.current_episode_id,
                "log_line": current_line,
                "user_input": user_input,
                "assistant_response": assistant_response,
                "memory_extracted": [
                    {
                        "text": m.text,
                        "type": m.memory_type.value,
                        "priority": m.priority
                    } for m in memory_extracted
                ] if memory_extracted else []
            }
            interaction_logger.info(json.dumps(log_entry))
            self.log_line_count = current_line + 1
            self.interaction_count += 1
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")
    
    def extract_memory_candidates(self, user_input: str, assistant_response: str) -> List[MemoryCandidate]:
        """Extract potential memories from an exchange using LLM"""
        # Quick check for trivial exchanges
        user_lower = user_input.lower().strip()
        if len(user_lower.split()) <= 3 and any(pattern in user_lower for pattern in self.trivial_patterns):
            return []
        
        prompt = f"""SYSTEMATICALLY extract ALL factual information from the user's statement. Be thorough and comprehensive.

ANALYZE THE USER INPUT FOR:

1. PERSONAL FACTS:
   - Names (my name is X, I am Y)
   - Occupations (I am a doctor, X is an engineer)
   - Locations (I live in X, we are in Y)
   - Demographics and basic info

2. RELATIONSHIPS:
   - Family (my brother X, our cousin Y, my sister Z)
   - Friends (my friend A, our buddy B)
   - Professional (my colleague C, my boss D)
   - Social connections and how people relate

3. PREFERENCES & OPINIONS:
   - Likes/dislikes (I like X, Y hates Z)
   - Favorites (my favorite is A, B prefers C)
   - Attitudes (X loves Y, Z disagrees with A)

4. ACTIVITIES & PROJECTS:
   - Jobs/careers (X works as Y, we are Z)
   - Hobbies (X and I are artists, Y plays sports)
   - Current activities (we are on a bus, visiting X)

5. FACTUAL STATEMENTS:
   - Technical details, specifications
   - Objective information
   - Important decisions or commitments

EXTRACTION RULES:
- Extract EVERY factual detail, don't skip anything
- Focus ONLY on what the USER said, ignore assistant responses
- Use clear, specific language: "X is user's brother" not "user has a brother"
- Capture both direct facts AND implied relationships
- Maintain exact meaning, don't interpret

User said: "{user_input}"
Assistant said: "{assistant_response}"

Return a comprehensive JSON array with ALL facts from the user's statement:

Format: [{{"text": "specific fact", "type": "personal_fact|relationship|preference|project|technical", "priority": 0.0-1.0}}]

Examples:

User: "I'm Sarah, a teacher. My brother Tom is a chef who loves pasta but hates fish."
[
  {{"text": "User's name is Sarah", "type": "personal_fact", "priority": 0.9}},
  {{"text": "Sarah is a teacher", "type": "personal_fact", "priority": 0.8}},
  {{"text": "Tom is user's brother", "type": "relationship", "priority": 0.8}},
  {{"text": "Tom is a chef", "type": "personal_fact", "priority": 0.8}},
  {{"text": "Tom loves pasta", "type": "preference", "priority": 0.7}},
  {{"text": "Tom hates fish", "type": "preference", "priority": 0.7}}
]

Extract ALL facts comprehensively:

JSON array:"""
        
        try:
            response = self.llm(
                prompt,
                max_tokens=500,  # Increased for complex inputs
                temperature=0.2,  # Even lower temperature for more consistent extraction
                stop=["\n\n", "User:", "Assistant:", "Question:"]  # Added Question: to stop
            )
            
            if not response.get('choices') or len(response['choices']) == 0:
                logger.error("Invalid extraction response: no choices found")
                return []
            
            response_text = response['choices'][0].get('text', '').strip()
            
            # Clean up response text - more robust parsing
            if "```json" in response_text:
                parts = response_text.split("```json")
                if len(parts) > 1:
                    parts2 = parts[1].split("```")
                    if len(parts2) > 0:
                        response_text = parts2[0].strip()
            elif "```" in response_text:
                parts = response_text.split("```")
                if len(parts) > 2:
                    response_text = parts[1].strip()
            
            # Remove any extra text after the JSON array
            # Find the JSON array bounds more carefully
            start_idx = response_text.find('[')
            if start_idx != -1:
                # Find the matching closing bracket
                bracket_count = 0
                end_idx = start_idx
                for i, char in enumerate(response_text[start_idx:], start_idx):
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                        if bracket_count == 0:
                            end_idx = i
                            break
                
                if end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx+1]
                    try:
                        memories_data = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Initial JSON parse failed: {e}, attempting cleanup")
                        # If JSON parsing fails, try to clean up common issues
                        json_str = json_str.replace("'", '"')  # Replace single quotes
                        json_str = json_str.replace('True', 'true').replace('False', 'false')
                        memories_data = json.loads(json_str)
                
                    candidates = []
                    for mem in memories_data:
                        # Validate and create candidate
                        if isinstance(mem, dict) and 'text' in mem:
                            # Get type with fallback
                            mem_type = mem.get('type', 'conversation')
                            if mem_type not in [t.value for t in MemoryType]:
                                mem_type = 'conversation'
                            
                            # Get priority with fallback
                            try:
                                priority = float(mem.get('priority', 0.5))
                                priority = max(0.0, min(1.0, priority))  # Clamp to 0-1
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Invalid priority value '{mem.get('priority')}' for memory: {e}. Using default 0.5")
                                priority = 0.5
                            
                            candidate = MemoryCandidate(
                                text=mem['text'],
                                memory_type=MemoryType(mem_type),
                                priority=priority,
                                metadata={"source": "extraction", "raw_input": user_input}
                            )
                            
                            # Only add if it passes basic checks
                            if len(candidate.text.strip()) > 5:  # Not too short
                                candidates.append(candidate)
                    
                    return candidates
            
            # If no JSON found, return empty list
            return []
            
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parsing failed: {e}, response was: {response_text[:100]}...")
        except Exception as e:
            logger.error(f"Failed to extract memories: {e}")
        
        return []
    
    def should_store_memory(self, candidate: MemoryCandidate) -> bool:
        """Determine if a memory candidate should be stored"""
        # Check priority threshold
        if candidate.priority < self.min_priority_threshold:
            return False
        
        # Check for trivial content
        text_lower = candidate.text.lower()
        if any(pattern in text_lower for pattern in self.trivial_patterns) and len(text_lower.split()) < 5:
            return False
        
        # Temporary memories should have very high priority
        if candidate.memory_type == MemoryType.TEMPORARY and candidate.priority < 0.8:
            return False
        
        return True
    
    def determine_memory_operation(self, candidate: MemoryCandidate) -> Tuple[MemoryOperation, Optional[str]]:
        """Determine what operation to perform on a memory candidate"""
        # Search for similar existing memories
        try:
            embedding = self.embedder.encode([candidate.text])[0]
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=embedding.tolist(),
                query_filter={
                    "must": [{"key": "user_id", "match": {"value": self.user_id}}]
                },
                limit=3
            )
            
            # Check for very similar memories (potential duplicates or updates)
            for result in search_results:
                if result.score > 0.9:  # Very high similarity
                    existing_text = result.payload.get("text", "")
                    
                    # If it's essentially the same, skip
                    if result.score > 0.95:
                        return MemoryOperation.SKIP, None
                    
                    # If it's an update (similar but not identical)
                    if candidate.priority > result.payload.get("priority", 0):
                        return MemoryOperation.UPDATE, result.id
            
            # Check for contradictions in relationship/fact memories
            if candidate.memory_type in [MemoryType.PERSONAL_FACT, MemoryType.RELATIONSHIP]:
                for result in search_results:
                    if result.score > 0.7:  # Moderately similar
                        existing_type = result.payload.get("memory_type", "")
                        if existing_type == candidate.memory_type.value:
                            # Potential contradiction, needs update
                            return MemoryOperation.UPDATE, result.id
            
        except Exception as e:
            logger.error(f"Error determining memory operation: {e}")
        
        return MemoryOperation.ADD, None
    
    def add_memory_advanced(self, candidate: MemoryCandidate) -> Optional[str]:
        """Add a memory with advanced metadata"""
        if not self.should_store_memory(candidate):
            return None
        
        memory_id = str(uuid.uuid4())
        timestamp = time.time()
        
        try:
            # Generate embedding
            embedding = self.embedder.encode([candidate.text])[0]
            
            # Store in Qdrant with rich metadata
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=memory_id,
                    vector=embedding.tolist(),
                    payload={
                        "text": candidate.text,
                        "user_id": self.user_id,
                        "memory_type": candidate.memory_type.value,
                        "priority": candidate.priority,
                        "timestamp": timestamp,
                        "last_accessed": timestamp,
                        "access_count": 0,
                        **candidate.metadata
                    }
                )]
            )
            
            # Store in Neo4j with type information
            with self.neo4j.session() as session:
                session.run("""
                    CREATE (m:HermesMemory {
                        id: $memory_id,
                        text: $text,
                        user_id: $user_id,
                        memory_type: $memory_type,
                        priority: $priority,
                        timestamp: $timestamp,
                        last_accessed: $timestamp,
                        access_count: 0
                    })
                """, memory_id=memory_id, text=candidate.text, user_id=self.user_id, 
                     memory_type=candidate.memory_type.value, priority=candidate.priority,
                     timestamp=timestamp)
            
            logger.info(f"Added {candidate.memory_type.value} memory with priority {candidate.priority}")
            return memory_id
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return None
    
    def update_memory(self, memory_id: str, candidate: MemoryCandidate):
        """Update an existing memory"""
        try:
            timestamp = time.time()
            
            # Update in Qdrant
            # First get the existing memory
            existing = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=[memory_id]
            )[0]
            
            # Update with new information
            new_embedding = self.embedder.encode([candidate.text])[0]
            
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[PointStruct(
                    id=memory_id,
                    vector=new_embedding.tolist(),
                    payload={
                        "text": candidate.text,
                        "user_id": self.user_id,
                        "memory_type": candidate.memory_type.value,
                        "priority": candidate.priority,
                        "timestamp": existing.payload["timestamp"],  # Keep original timestamp
                        "last_updated": timestamp,
                        "last_accessed": timestamp,
                        "access_count": existing.payload.get("access_count", 0) + 1,
                        "previous_text": existing.payload["text"],  # Store previous version
                        **candidate.metadata
                    }
                )]
            )
            
            # Update in Neo4j
            with self.neo4j.session() as session:
                session.run("""
                    MATCH (m:HermesMemory {id: $memory_id})
                    SET m.text = $text,
                        m.memory_type = $memory_type,
                        m.priority = $priority,
                        m.last_updated = $timestamp,
                        m.last_accessed = $timestamp,
                        m.access_count = m.access_count + 1,
                        m.previous_text = m.text
                """, memory_id=memory_id, text=candidate.text, 
                     memory_type=candidate.memory_type.value, priority=candidate.priority,
                     timestamp=timestamp)
            
            logger.info(f"Updated memory {memory_id}")
        except Exception as e:
            logger.error(f"Failed to update memory: {e}")
    
    def search_memories_advanced(self, query: str, limit: int = 10, 
                               memory_types: Optional[List[MemoryType]] = None) -> List[Dict]:
        """Advanced memory search with filtering and ranking"""
        try:
            # Generate query embedding
            query_embedding = self.embedder.encode([query])[0]
            
            # Build filter
            must_conditions = [{"key": "user_id", "match": {"value": self.user_id}}]
            if memory_types:
                must_conditions.append({
                    "key": "memory_type",
                    "match": {"any": [mt.value for mt in memory_types]}
                })
            
            # Search in Qdrant
            search_results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                query_filter={"must": must_conditions},
                limit=limit * 2  # Get more results for filtering
            )
            
            # Post-process and rank results
            memories = []
            for point in search_results:
                # Calculate combined score (similarity + priority + recency)
                similarity_score = point.score
                priority_score = point.payload.get("priority", 0.5)
                
                # Recency score (memories accessed recently get a boost)
                last_accessed = point.payload.get("last_accessed", 0)
                recency_score = 1.0 / (1.0 + (time.time() - last_accessed) / 86400)  # Day-based decay
                
                combined_score = (similarity_score * 0.5 + priority_score * 0.3 + recency_score * 0.2)
                
                memories.append({
                    "id": point.id,
                    "text": point.payload["text"],
                    "score": combined_score,
                    "similarity": similarity_score,
                    "memory_type": point.payload.get("memory_type", "unknown"),
                    "priority": priority_score,
                    "timestamp": point.payload.get("timestamp", 0)
                })
            
            # Sort by combined score and return top results
            memories.sort(key=lambda x: x["score"], reverse=True)
            
            # Update access count for retrieved memories
            for mem in memories[:limit]:
                self._update_access_count(mem["id"])
            
            return memories[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    def _update_access_count(self, memory_id: str):
        """Update access count and timestamp for a memory"""
        try:
            timestamp = time.time()
            
            # Update in Neo4j (faster for this operation)
            with self.neo4j.session() as session:
                session.run("""
                    MATCH (m:HermesMemory {id: $memory_id})
                    SET m.last_accessed = $timestamp,
                        m.access_count = m.access_count + 1
                """, memory_id=memory_id, timestamp=timestamp)
        except Exception as e:
            logger.warning(f"Failed to update access count for memory {memory_id}: {e}")
            # Continue anyway - access tracking is not critical
    
    def process_interaction_background(self, user_input: str, assistant_response: str):
        """Process interaction in background thread with episode detection"""
        # Generate embedding for episode detection
        user_embedding = self.embedder.encode([user_input])[0]
        
        # Check for episode boundary
        if self.previous_embedding is not None:
            if self.summarizer.detect_episode_boundary(user_embedding, self.previous_embedding):
                logger.info(f"Episode boundary detected, ending episode {self.current_episode_id}")
                # Queue episode summary creation
                self.memory_queue.put(("EPISODE_END", self.current_episode_id))
                # Start new episode
                self.current_episode_id = str(uuid.uuid4())
                logger.info(f"Started new episode {self.current_episode_id}")
        
        self.previous_embedding = user_embedding
        
        # Add to queue for processing
        self.memory_queue.put((user_input, assistant_response))
        
        # Check if we need periodic summarization
        if self.summarizer.should_summarize(self.interaction_count, self.last_summary_count):
            logger.info("Triggering periodic summarization")
            self.memory_queue.put(("SUMMARIZE", self.interaction_count))
    
    def _memory_processor(self):
        """Background thread that processes memory queue"""
        while not self.shutdown_event.is_set():
            try:
                # Get from queue (blocks until item available or timeout)
                item = self.memory_queue.get(timeout=1)
                
                # Handle different types of tasks
                if isinstance(item, tuple) and len(item) == 2:
                    task_type = item[0]
                    
                    if task_type == "EPISODE_END":
                        # Create episode summary
                        episode_id = item[1]
                        logger.info(f"Creating summary for episode {episode_id}")
                        self.summarizer.create_episode_summary(episode_id, self.user_id)
                        
                    elif task_type == "SUMMARIZE":
                        # Periodic summarization
                        count = item[1]
                        logger.info(f"Starting periodic summarization at {count} interactions")
                        self._perform_periodic_summarization()
                        
                    else:
                        # Regular memory processing
                        user_input, assistant_response = item
                        
                        # Extract memory candidates
                        candidates = self.extract_memory_candidates(user_input, assistant_response)
                        
                        # Log the interaction with extracted memories
                        self.log_interaction(user_input, assistant_response, candidates)
                        
                        # Process each candidate
                        for candidate in candidates:
                            operation, related_id = self.determine_memory_operation(candidate)
                            
                            if operation == MemoryOperation.ADD:
                                self.add_memory_advanced(candidate)
                            elif operation == MemoryOperation.UPDATE and related_id:
                                self.update_memory(related_id, candidate)
                
                self.memory_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in memory processor: {e}")
    
    def _perform_periodic_summarization(self):
        """Perform periodic summarization of recent interactions"""
        try:
            # Calculate line range for summarization
            start_line = self.last_summary_count
            end_line = self.interaction_count - 1
            
            # Load interactions for summarization
            interactions = self.summarizer.load_interaction_range(start_line, end_line)
            
            if interactions:
                # Generate and store summary
                summary = self.summarizer.generate_summary(interactions, summary_type="periodic")
                if summary:
                    self.summarizer.store_summary_memory(summary, self.user_id)
                    self.last_summary_count = self.interaction_count
                    logger.info(f"Created periodic summary covering {len(interactions)} interactions")
        except Exception as e:
            logger.error(f"Failed to perform periodic summarization: {e}")
    
    def get_memory_with_context(self, memory: Dict, relevance_score: float) -> Dict:
        """Get memory with optional full context expansion"""
        if memory.get("memory_type") == "summary" and relevance_score > 0.85:
            # High relevance - load full context
            metadata = memory.get("metadata", {})
            if metadata:
                full_context = self.summarizer.load_full_context(metadata)
                if full_context:
                    return {
                        "summary": memory["text"],
                        "full_context": full_context,
                        "expanded": True,
                        "relevance": relevance_score
                    }
        
        # Return just the memory text for non-summaries or lower relevance
        return {
            "text": memory["text"],
            "expanded": False,
            "relevance": relevance_score
        }
    
    def get_conversation_context(self, user_input: str) -> str:
        """Get relevant context for the conversation with smart expansion"""
        # Search for relevant memories with different types
        personal_memories = self.search_memories_advanced(
            user_input, 
            limit=3, 
            memory_types=[MemoryType.PERSONAL_FACT, MemoryType.RELATIONSHIP]
        )
        
        preference_memories = self.search_memories_advanced(
            user_input,
            limit=2,
            memory_types=[MemoryType.PREFERENCE]
        )
        
        recent_memories = self.search_memories_advanced(
            user_input,
            limit=5,
            memory_types=[MemoryType.CONVERSATION]
        )
        
        # Check for summaries in all memory types
        all_memories = personal_memories + preference_memories + recent_memories
        summaries = self.search_memories_advanced(
            user_input,
            limit=2,
            memory_types=[MemoryType.SUMMARY, MemoryType.EPISODE]
        )
        all_memories.extend(summaries)
        
        # Build context with smart expansion
        context = ""
        expanded_any = False
        
        if personal_memories:
            context += "Personal information:\n"
            for mem in personal_memories:
                expanded_mem = self.get_memory_with_context(mem, mem.get('similarity', 0.5))
                if expanded_mem['expanded']:
                    context += f"- [Summary] {expanded_mem['summary']}\n"
                    context += f"  [Expanded context]:\n{expanded_mem['full_context']}\n"
                    expanded_any = True
                else:
                    context += f"- {mem['text']}\n"
            context += "\n"
        
        if preference_memories:
            context += "User preferences:\n"
            for mem in preference_memories:
                context += f"- {mem['text']}\n"
            context += "\n"
        
        if recent_memories:
            context += "Recent conversation:\n"
            for mem in recent_memories:
                context += f"- {mem['text']}\n"
            context += "\n"
            
        # Include high-relevance summaries
        if summaries:
            context += "Relevant summaries:\n"
            for mem in summaries:
                expanded_mem = self.get_memory_with_context(mem, mem.get('similarity', 0.5))
                if expanded_mem['expanded']:
                    context += f"- [Summary] {expanded_mem['summary']}\n"
                    if not expanded_any:  # Only show one expanded context
                        context += f"  [Full conversation context]:\n{expanded_mem['full_context'][:1000]}...\n"
                        expanded_any = True
                else:
                    context += f"- {mem['text']}\n"
            context += "\n"
        
        return context
    
    def chat(self):
        """Interactive chat with improved memory"""
        print("\n" + "="*60)
        print("🤖 Buddy v0.4.0-dev - AI Assistant with Memory Consolidation")
        print("="*60)
        print("\nCommands:")
        print("  /memory    - View stored memories by type")
        print("  /clear     - Clear all memories")
        print("  /stats     - View memory statistics")
        print("  /summarize - Force memory summarization")
        print("  /episodes  - List conversation episodes")
        print("  /exit      - Exit the chat")
        print("\nStart chatting! I'll remember and intelligently summarize our conversations.\n")
        
        while True:
            try:
                # Get user input
                try:
                    user_input = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == '/exit':
                    print("Goodbye!")
                    break
                
                if user_input.lower() == '/memory':
                    self.show_memories_by_type()
                    continue
                
                if user_input.lower() == '/stats':
                    self.show_memory_stats()
                    continue
                
                if user_input.lower() == '/clear':
                    count = self.clear_memories()
                    print(f"✅ Cleared {count} memories\n")
                    continue
                
                if user_input.lower() == '/summarize':
                    print("Forcing memory summarization...")
                    self._perform_periodic_summarization()
                    print("✅ Summarization complete\n")
                    continue
                
                if user_input.lower() == '/episodes':
                    self.show_episodes()
                    continue
                
                # Get conversation context
                context = self.get_conversation_context(user_input)
                
                # Construct prompt
                system_prompt = """You are Buddy, a helpful AI assistant with an advanced memory system. 
You remember important information about users including their personal facts, preferences, and relationships.
You can distinguish between important information worth remembering and casual conversation.
Always identify yourself as Buddy when asked who you are."""
                
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
                
                # Process the interaction for memory extraction (in background)
                self.process_interaction_background(user_input, assistant_response)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /exit to quit.\n")
                continue
            except Exception as e:
                logger.error(f"Error during chat: {e}")
                print(f"Sorry, an error occurred: {e}\n")
                continue
    
    def show_memories_by_type(self):
        """Display memories organized by type"""
        try:
            with self.neo4j.session() as session:
                result = session.run("""
                    MATCH (m:HermesMemory {user_id: $user_id})
                    RETURN m.memory_type as type, collect(m.text) as memories
                    ORDER BY type
                """, user_id=self.user_id)
                
                print("\n📚 Stored Memories by Type:")
                print("-" * 40)
                
                for record in result:
                    memory_type = record["type"]
                    memories = record["memories"]
                    
                    if memories:
                        print(f"\n{memory_type.upper()}:")
                        for i, mem in enumerate(memories, 1):
                            print(f"  {i}. {mem}")
                
                print()
        except Exception as e:
            logger.error(f"Failed to show memories by type: {e}")
    
    def show_memory_stats(self):
        """Display memory statistics"""
        try:
            with self.neo4j.session() as session:
                # Get counts by type
                result = session.run("""
                    MATCH (m:HermesMemory {user_id: $user_id})
                    RETURN m.memory_type as type, count(m) as count
                """, user_id=self.user_id)
                
                print("\n📊 Memory Statistics:")
                print("-" * 40)
                
                total = 0
                for record in result:
                    print(f"{record['type']}: {record['count']} memories")
                    total += record['count']
                
                print(f"\nTotal memories: {total}")
                
                # Get average priority by type
                result = session.run("""
                    MATCH (m:HermesMemory {user_id: $user_id})
                    RETURN m.memory_type as type, avg(m.priority) as avg_priority
                """, user_id=self.user_id)
                
                print("\nAverage priority by type:")
                for record in result:
                    if record['avg_priority']:
                        print(f"{record['type']}: {record['avg_priority']:.2f}")
                
                # Check interaction log size
                log_file = log_dir / "interactions.jsonl"
                if log_file.exists():
                    size_mb = log_file.stat().st_size / (1024 * 1024)
                    print(f"\nInteraction log size: {size_mb:.2f} MB")
                
                print()
        except Exception as e:
            logger.error(f"Failed to show memory stats: {e}")
    
    def show_episodes(self):
        """Display conversation episodes"""
        try:
            with self.neo4j.session() as session:
                # Get episode summaries
                result = session.run("""
                    MATCH (m:HermesMemory {user_id: $user_id})
                    WHERE m.memory_type = 'episode' OR m.memory_type = 'summary'
                    RETURN m.text, m.timestamp, m.summary_type, m.interaction_count
                    ORDER BY m.timestamp DESC
                    LIMIT 10
                """, user_id=self.user_id)
                
                print("\n📖 Conversation Episodes & Summaries:")
                print("-" * 50)
                
                for record in result:
                    timestamp = datetime.fromtimestamp(record['m.timestamp']).strftime('%Y-%m-%d %H:%M')
                    summary_type = record.get('m.summary_type', 'unknown')
                    interaction_count = record.get('m.interaction_count', 0)
                    
                    print(f"\n[{timestamp}] {summary_type.capitalize()} ({interaction_count} interactions)")
                    print(f"  {record['m.text'][:100]}...")
                
                print()
        except Exception as e:
            logger.error(f"Failed to show episodes: {e}")
    
    def clear_memories(self):
        """Clear all memories"""
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
            
            return count
        except Exception as e:
            logger.error(f"Failed to clear memories: {e}")
            return 0


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
    except Exception as e:
        logger.error(f"Could not check Qdrant status: {e}")
        services_ok = False
    
    # Check Neo4j
    try:
        result = subprocess.run(['rc-service', 'neo4j', 'status'], 
                              capture_output=True, text=True)
        if 'running' not in result.stdout.lower() and 'started' not in result.stdout:
            logger.warning("Neo4j service not running")
            services_ok = False
    except Exception as e:
        logger.error(f"Could not check Neo4j status: {e}")
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
    # Check for test config first
    test_config_path = Path(__file__).parent / "mem0_config_test.yaml"
    if test_config_path.exists():
        config_path = test_config_path
    else:
        config_path = Path(__file__).parent / "mem0_config.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default config with explicit Qdrant storage path
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
            },
            "qdrant": {
                "storage_path": "/var/lib/qdrant/storage"
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
    system = RobustMemorySystem(config)
    system.initialize()
    
    # Start memory processor thread
    system.memory_thread = threading.Thread(target=system._memory_processor)
    system.memory_thread.start()
    
    try:
        # Start chat
        system.chat()
    finally:
        # Cleanup will be handled by shutdown handlers
        logger.info("Shutting down...")


if __name__ == "__main__":
    main()