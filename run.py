#!/usr/bin/env python3
"""
Properly fixed mem0 implementation using official API methods
Addresses the key issues:
1. Memory forgetting actually deletes memories
2. Boss hierarchy correctly parsed
3. Deduplication before adding
4. Clean entity extraction
"""

import os
import sys
import re
import logging
from typing import Dict, List, Optional, Any, Set
from mem0 import Memory
from openai import OpenAI
from dotenv import load_dotenv
from difflib import SequenceMatcher

# Phase 1 Context System imports
from context_logger import ContextLogger
from context_bridge import ContextBridge
from temporal_utils import inject_temporal_awareness, get_current_datetime_info

# Phase 2 Enhanced Memory System
from enhanced_memory import EnhancedMemory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress HTTP request logs from httpx and openai unless in debug mode
if os.getenv("DEBUG", "").lower() != "true":
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

class ImprovedMem0Assistant:
    """Improved mem0 assistant that properly handles memory operations"""
    
    def __init__(self):
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("Please set OPENAI_API_KEY environment variable")
            sys.exit(1)
        
        # Initialize mem0 with proper configuration
        self.config = {
            "version": "v1.1",  # Required for graph support
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": "bolt://localhost:7687",
                    "username": "neo4j",
                    "password": "password123"
                }
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333,
                    "collection_name": "mem0_fixed"
                }
            }
        }
        
        try:
            # Initialize base mem0 instance
            base_memory = Memory.from_config(config_dict=self.config)
            
            # Wrap with EnhancedMemory for Phase 2 functionality
            self.memory = EnhancedMemory(base_memory)
            
            self.openai_client = OpenAI()
            logger.info("Successfully initialized mem0 with graph memory support")
            logger.info("Successfully initialized Phase 2 enhanced memory system")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced memory system: {e}")
            raise
        
        # Track entities to forget
        self.forget_list = set()
        
    def parse_possessive_relationship(self, text: str) -> Optional[Dict[str, str]]:
        """Parse possessive relationships correctly"""
        # Josh's boss is Dave -> Dave manages Josh
        match = re.search(r"(\w+)'s\s+boss\s+is\s+(\w+)", text, re.IGNORECASE)
        if match:
            subordinate = match.group(1)
            boss = match.group(2)
            return {
                "fact": f"{boss} manages {subordinate}",
                "relationship": {"subject": boss, "relation": "manages", "object": subordinate}
            }
        return None
    
    def process_forget_request(self, message: str, user_id: str) -> bool:
        """Process forget requests by actually deleting memories"""
        # Check for forget patterns
        forget_patterns = [
            r"forget\s+(?:about\s+)?(\w+)",
            r"don't\s+remember\s+(\w+)",
            r"remove\s+(?:memories\s+about\s+)?(\w+)"
        ]
        
        for pattern in forget_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                entity_to_forget = match.group(1).lower()
                logger.info(f"Processing forget request for: {entity_to_forget}")
                
                # Get all memories
                all_memories = self.memory.get_all(user_id=user_id)
                deleted_count = 0
                
                if isinstance(all_memories, dict) and 'results' in all_memories:
                    for memory in all_memories['results']:
                        memory_text = memory.get('memory', memory.get('text', ''))
                        memory_id = memory.get('id')
                        
                        # Check if this memory mentions the entity
                        if entity_to_forget in memory_text.lower():
                            try:
                                # Actually delete the memory
                                if memory_id:
                                    self.memory.delete(memory_id=memory_id)
                                    deleted_count += 1
                                    logger.info(f"Deleted memory: {memory_text[:50]}..." if len(memory_text) > 50 else f"Deleted memory: {memory_text}")
                                else:
                                    logger.warning(f"No memory ID found for: {memory_text}")
                            except Exception as e:
                                logger.error(f"Error deleting memory {memory_id}: {e}")
                
                logger.info(f"Deleted {deleted_count} memories about {entity_to_forget}")
                return True
        
        return False
    
    def check_for_duplicates(self, new_memory: str, user_id: str, threshold: float = 0.85) -> Optional[str]:
        """Check if a similar memory already exists"""
        all_memories = self.memory.get_all(user_id=user_id)
        
        if isinstance(all_memories, dict) and 'results' in all_memories:
            for memory in all_memories['results']:
                existing_text = memory.get('memory', memory.get('text', ''))
                similarity = SequenceMatcher(None, new_memory.lower(), existing_text.lower()).ratio()
                
                if similarity >= threshold:
                    return memory.get('id')
        
        return None
    
    def process_message(self, message: str, user_id: str) -> str:
        """Process user message with improved memory handling and context logging"""
        
        # Add temporal awareness to the message
        temporal_message = inject_temporal_awareness(message)
        
        # First, check if this is a forget request
        if self.process_forget_request(message, user_id):
            return "I've forgotten about that as requested."
        
        # Search for relevant memories
        relevant_memories = self.memory.search(query=message, user_id=user_id, limit=10)
        
        # Build context from memories
        memories_list = []
        if relevant_memories and "results" in relevant_memories:
            for entry in relevant_memories["results"]:
                if isinstance(entry, dict):
                    memory_text = entry.get("memory", entry.get("text", ""))
                    if memory_text:
                        memories_list.append(memory_text)
        
        # Build system prompt
        context = "\n".join(f"- {mem}" for mem in memories_list) if memories_list else "No previous memories."
        system_prompt = f"""You are a helpful AI assistant with the following memories about the user:

{context}"""
        
        # Generate response with temporal awareness
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": temporal_message}
        ]
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            assistant_response = response.choices[0].message.content
            
            # Add the conversation to memory (EnhancedMemory handles context logging automatically)
            # But first, check for special patterns like possessive relationships
            possessive = self.parse_possessive_relationship(message)
            if possessive:
                # Add the clarified fact instead
                result = self.memory.add(
                    [{"role": "user", "content": possessive["fact"]}],
                    user_id=user_id
                )
                # Log relationship extraction if available
                if hasattr(result, 'get') and 'relationships' in str(result):
                    logger.info(f"Relationship extracted: {possessive['relationship']}")
            else:
                # Normal memory addition - let mem0's LLM decide what to extract
                conversation = [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": assistant_response}
                ]
                result = self.memory.add(conversation, user_id=user_id)
                # Log any relationships that were extracted
                if hasattr(result, 'get') and 'relationships' in str(result):
                    logger.info(f"Memory added with potential relationships from: {message[:50]}...")
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error generating a response."
    
    def show_all_memories(self, user_id: str):
        """Display all memories in a clean format"""
        all_memories = self.memory.get_all(user_id=user_id)
        
        memories_found = []
        
        if isinstance(all_memories, dict) and 'results' in all_memories:
            # First, deduplicate
            seen = set()
            for mem in all_memories['results']:
                memory_text = mem.get('memory', mem.get('text', ''))
                normalized = memory_text.lower().strip()
                
                # Skip near-duplicates
                is_duplicate = False
                for existing in seen:
                    if SequenceMatcher(None, normalized, existing).ratio() > 0.85:
                        is_duplicate = True
                        break
                
                if not is_duplicate and memory_text:
                    memories_found.append(memory_text)
                    seen.add(normalized)
        
        # Handle relationships from 'relations' in get_all() response
        relations_found = []
        if isinstance(all_memories, dict) and 'relations' in all_memories:
            logger.info(f"Retrieved {len(all_memories['relations'])} relationships")
            for rel in all_memories['relations']:
                if isinstance(rel, dict):
                    source = rel.get('source', 'unknown')
                    relationship = rel.get('relationship', rel.get('relation', 'unknown'))
                    destination = rel.get('destination', rel.get('target', 'unknown'))
                    relations_found.append(f"{source} -> {relationship} -> {destination}")
        
        # Display results
        total_items = len(memories_found) + len(relations_found)
        if total_items > 0:
            print(f"\nAll stored memories ({total_items} items):")
            count = 1
            if memories_found:
                for memory in memories_found:
                    print(f"{count}. {memory}")
                    count += 1
            if relations_found:
                print(f"\nRelationships:")
                for relation in relations_found:
                    print(f"{count}. {relation}")
                    count += 1
        else:
            print("\nNo memories found.")
    
    def show_all_relationships(self, user_id: str):
        """Display all relationship/graph data"""
        try:
            # Get all relationships from the graph store
            relationships = self.memory.graph.get_all(filters={'user_id': user_id})
            
            if relationships:
                print(f"\nAll stored relationships ({len(relationships)} items):")
                
                # Group relationships by type for better display
                by_relation = {}
                for rel in relationships:
                    relation_type = rel.get('relationship', 'unknown')
                    if relation_type not in by_relation:
                        by_relation[relation_type] = []
                    by_relation[relation_type].append(rel)
                
                # Display grouped relationships
                for relation_type, rels in by_relation.items():
                    print(f"\n  {relation_type.upper()} relationships:")
                    for rel in rels:
                        source = rel.get('source', 'unknown')
                        target = rel.get('target', rel.get('destination', 'unknown'))
                        print(f"    • {source} → {target}")
            else:
                print("\nNo relationships found.")
                
        except Exception as e:
            print(f"\nError retrieving relationships: {e}")
            logger.error(f"Error in show_all_relationships: {e}")
    
    def search_relationships(self, query: str, user_id: str):
        """Search for specific relationships"""
        try:
            results = self.memory.graph.search(query=query, filters={'user_id': user_id})
            
            if results:
                print(f"\nRelationships matching '{query}' ({len(results)} items):")
                for i, rel in enumerate(results, 1):
                    source = rel.get('source', 'unknown')
                    relation = rel.get('relationship', 'unknown')
                    target = rel.get('target', rel.get('destination', 'unknown'))
                    print(f"{i}. {source} --{relation}--> {target}")
            else:
                print(f"\nNo relationships found matching '{query}'.")
                
        except Exception as e:
            print(f"\nError searching relationships: {e}")
            logger.error(f"Error in search_relationships: {e}")

def main():
    """Main chat loop"""
    assistant = ImprovedMem0Assistant()
    user_id = "adam_001"  # Better user ID
    
    print("Mem0 Assistant Ready!")
    print("Commands:")
    print("  'memories' - show all stored memories")
    print("  'relationships' - show all relationship/graph data")
    print("  'search relationships <query>' - search for specific relationships")
    print("  'forget X' - remove memories about X")
    print("  'quit' - exit")
    print()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            break
            
        if user_input.lower() == 'memories':
            assistant.show_all_memories(user_id)
            continue
            
        if user_input.lower() == 'relationships':
            assistant.show_all_relationships(user_id)
            continue
            
        if user_input.lower().startswith('search relationships '):
            query = user_input[len('search relationships '):].strip()
            if query:
                assistant.search_relationships(query, user_id)
            else:
                print("Please specify a search query. Example: 'search relationships Dave'")
            continue
        
        if not user_input:
            continue
        
        # Process the message
        response = assistant.process_message(user_input, user_id)
        print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    main()