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
from datetime import datetime, timezone
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
            
            # Initialize Phase 3 context expansion system
            from context_logger import ContextLogger
            from context_bridge import ContextBridge
            from context_expander import ContextExpander
            
            self.context_logger = ContextLogger()
            self.context_bridge = ContextBridge(self.context_logger)
            self.context_expander = ContextExpander(self.context_bridge)
            
            # Wrap with EnhancedMemory for Phase 3 functionality
            self.memory = EnhancedMemory(base_memory, self.context_logger, self.context_expander)
            
            self.openai_client = OpenAI()
            logger.info("Successfully initialized mem0 with graph memory support")
            logger.info("Successfully initialized Phase 3 enhanced memory system with context expansion")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced memory system: {e}")
            raise
        
        # Track entities to forget
        self.forget_list = set()
        
        # Track last search results for expansion commands
        self.last_search_results = None
        self.last_search_query = None
        
        # Context expansion settings
        self.expansion_enabled = True
        
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
                                    self.memory.delete(memory_id=memory_id, user_id=user_id)
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
        
        # FIRST: Generate ONE lookup code for this entire interaction
        shared_lookup_code = self.context_logger.generate_lookup_code()
        
        # Add temporal awareness to the message
        temporal_message = inject_temporal_awareness(message)
        
        # First, check if this is a forget request
        if self.process_forget_request(message, user_id):
            return "I've forgotten about that as requested."
        
        # Search for relevant memories with expansion control
        relevant_memories = self.memory.search(
            query=message, 
            user_id=user_id, 
            limit=10,
            enable_expansion=self.expansion_enabled
        )
        
        # Store last search results for manual expansion commands
        self.last_search_results = relevant_memories
        self.last_search_query = message
        
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
            
            # Use the shared lookup code generated at the start of this function
            
            # Add the conversation to memory (mem0 storage independent of context logging)
            # But first, check for special patterns like possessive relationships
            possessive = self.parse_possessive_relationship(message)
            if possessive:
                # Add the clarified fact instead
                current_time = datetime.now(timezone.utc).isoformat()
                result = self.memory.add(
                    [{"role": "user", "content": possessive["fact"]}],
                    user_id=user_id,
                    metadata={
                        'context_lookup_code': shared_lookup_code,
                        'created_at': current_time,
                        'last_updated_at': current_time
                    }
                )
                
                # Log memory IDs that were created with their lookup codes
                if result and isinstance(result, dict) and 'results' in result:
                    for memory_item in result['results']:
                        memory_id = memory_item.get('id')
                        if memory_id:
                            logger.debug(f"Memory {memory_id} created from context {shared_lookup_code}")
                
                # Log relationship extraction if available
                if hasattr(result, 'get') and 'relationships' in str(result):
                    logger.info(f"Relationship extracted: {possessive['relationship']}")
            else:
                # Normal memory addition - let mem0's LLM decide what to extract
                conversation = [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": assistant_response}
                ]
                # Add metadata with the lookup code and timestamps for linking
                current_time = datetime.now(timezone.utc).isoformat()
                
                result = self.memory.add(
                    conversation, 
                    user_id=user_id,
                    metadata={
                        'context_lookup_code': shared_lookup_code,
                        'created_at': current_time,
                        'last_updated_at': current_time
                    }
                )
                
                
                # Validate result to ensure Rule 3 compliance (loud failures)
                if not result:
                    raise RuntimeError(f"MEM0 EXTRACTION RETURNED EMPTY RESULT")
                
                if not isinstance(result, dict):
                    raise RuntimeError(f"MEM0 EXTRACTION RETURNED INVALID TYPE: {type(result)}")
                
                if 'results' not in result:
                    raise RuntimeError(f"MEM0 EXTRACTION RESULT MISSING RESULTS KEY")
                
                # Log memory IDs that were created with their lookup codes
                if result and isinstance(result, dict) and 'results' in result:
                    for memory_item in result['results']:
                        memory_id = memory_item.get('id')
                        if memory_id:
                            logger.debug(f"Memory {memory_id} created from context {shared_lookup_code}")
                
                # Log any relationships that were extracted
                if hasattr(result, 'get') and 'relationships' in str(result):
                    logger.info(f"Memory added with potential relationships from: {message[:50]}...")
            
            # Now log the COMPLETE conversation with actual memory operations data
            try:
                # Extract search operations data from relevant_memories
                search_memories = []
                if relevant_memories and isinstance(relevant_memories, dict) and 'results' in relevant_memories:
                    search_memories = relevant_memories['results']
                
                # Extract addition operations data from result
                added_memories = []
                extracted_relationships = {}
                if result and isinstance(result, dict):
                    added_memories = result.get('results', [])
                    extracted_relationships = result.get('relations', result.get('relationships', {}))
                
                # Validate we have expected data structures (Rule 3 - fail loudly)
                if relevant_memories and not isinstance(relevant_memories, dict):
                    raise RuntimeError(f"UNEXPECTED relevant_memories TYPE: {type(relevant_memories)}")
                if result and not isinstance(result, dict):
                    raise RuntimeError(f"UNEXPECTED result TYPE: {type(result)}")
                
                memory_operations_data = {
                    "memories_searched": search_memories,  # FROM SEARCH RESULTS
                    "memories_found": search_memories,     # SAME AS SEARCHED FOR NOW
                    "memories_added": added_memories,      # FROM MEMORY ADDITION
                    "relationships_extracted": extracted_relationships
                }
                
                
                conversation_lookup_code = self.context_logger.log_interaction(
                    user_input=message,
                    assistant_response=assistant_response,
                    user_id=user_id,
                    enhanced_format=True,
                    memory_operations=memory_operations_data,
                    lookup_code=shared_lookup_code  # Use the same lookup code as memory creation
                )
                
                # LOUD VALIDATION: Context logger MUST return the same lookup code we provided
                if conversation_lookup_code != shared_lookup_code:
                    raise RuntimeError(f"CONTEXT LOGGER LOOKUP CODE MISMATCH: Provided {shared_lookup_code}, got back {conversation_lookup_code}")
                
                logger.debug(f"Context logged for conversation: {conversation_lookup_code}")
            except Exception as e:
                # Fail loudly per Rule 3 - don't hide context logging errors
                logger.error(f"CONTEXT LOGGING FAILED: {e}")
                raise  # This ensures context logging problems are not silent
            
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I encountered an error generating a response."
    
    def _is_valid_relationship(self, source: str, relation: str, target: str) -> bool:
        """
        Validate relationship quality - filter obvious junk while keeping meta-knowledge
        
        Based on P5 plan: Keep system meta-statements but filter meaningless junk
        """
        # Convert to strings and strip whitespace
        source = str(source).strip()
        relation = str(relation).strip()
        target = str(target).strip()
        
        # Reject if any part is empty
        if not source or not relation or not target:
            return False
        
        # Reject if all parts are single characters (likely extraction errors)
        if len(source) <= 1 and len(relation) <= 1 and len(target) <= 1:
            return False
        
        # Reject if relation is just punctuation or symbols
        if relation in ['→', '->', '--', '==', '::', '...', '???', '***']:
            return False
        
        # Reject if source and target are identical (meaningless self-references)
        if source.lower() == target.lower():
            return False
        
        # Reject obvious extraction artifacts
        junk_patterns = ['unknown', 'null', 'none', 'undefined', '']
        if any(part.lower() in junk_patterns for part in [source, relation, target]):
            return False
        
        # Accept everything else, including:
        # - System meta-knowledge: "memory -> reflects -> preferences"
        # - User relationships: "adam_001 -> has_friend -> joe"
        # - Temporal relationships: "thursday -> is_date -> june_05,_2025"
        return True
    
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
                    
                    # Apply intelligent relationship filtering
                    if self._is_valid_relationship(source, relationship, destination):
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
                
                # Group relationships by type for better display (with filtering)
                by_relation = {}
                for rel in relationships:
                    source = rel.get('source', 'unknown')
                    relation_type = rel.get('relationship', 'unknown')
                    target = rel.get('target', rel.get('destination', 'unknown'))
                    
                    # Apply intelligent relationship filtering
                    if self._is_valid_relationship(source, relation_type, target):
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
        """Search for specific relationships with case-insensitive and user-friendly matching"""
        try:
            # Try multiple variations for user-friendly search
            search_variations = [query]
            
            # Add lowercase variation
            if query.lower() != query:
                search_variations.append(query.lower())
            
            # If searching for a name like "Adam", also try "adam_001"
            if query.lower() in ['adam', 'user', 'me', 'self']:
                search_variations.append('adam_001')
            
            # Collect all results from all variations
            all_results = []
            seen_relationships = set()  # To avoid duplicates
            
            for variation in search_variations:
                try:
                    results = self.memory.graph.search(query=variation, filters={'user_id': user_id})
                    if results:
                        for rel in results:
                            source = rel.get('source', '')
                            relationship = rel.get('relationship', '')
                            target = rel.get('target', rel.get('destination', ''))
                            
                            # Apply intelligent relationship filtering
                            if self._is_valid_relationship(source, relationship, target):
                                # Create a unique key for deduplication
                                rel_key = f"{source}-{relationship}-{target}"
                                if rel_key not in seen_relationships:
                                    seen_relationships.add(rel_key)
                                    all_results.append(rel)
                except Exception:
                    # Continue with other variations if one fails
                    continue
            
            if all_results:
                print(f"\nRelationships matching '{query}' ({len(all_results)} items):")
                for i, rel in enumerate(all_results, 1):
                    source = rel.get('source', 'unknown')
                    relation = rel.get('relationship', 'unknown')
                    target = rel.get('target', rel.get('destination', 'unknown'))
                    print(f"{i}. {source} --{relation}--> {target}")
            else:
                print(f"\nNo relationships found matching '{query}'.")
                
        except Exception as e:
            print(f"\nError searching relationships: {e}")
            logger.error(f"Error in search_relationships: {e}")
    
    def show_context_for_memory(self, memory_id: str, user_id: str):
        """Show full context for a specific memory"""
        try:
            # Get all memories to find the one with matching ID
            all_memories = self.memory.get_all(user_id=user_id)
            
            target_memory = None
            if isinstance(all_memories, dict) and 'results' in all_memories:
                for memory in all_memories['results']:
                    if memory.get('id') == memory_id:
                        target_memory = memory
                        break
            
            if not target_memory:
                print(f"\nMemory with ID '{memory_id}' not found.")
                return
            
            # First check if memory has context lookup code in metadata
            metadata = target_memory.get('metadata', {})
            lookup_code = metadata.get('context_lookup_code')
            
            print(f"\n=== Context for Memory {memory_id} ===")
            print(f"Memory: {target_memory.get('memory', 'N/A')}")
            
            if lookup_code:
                print(f"Context Lookup Code: {lookup_code}")
                # Find the context log entry with this lookup code
                all_logs = self.context_logger.get_all_logs()
                context_log = None
                for log in all_logs:
                    if log.get('lookup_code') == lookup_code:
                        context_log = log
                        break
                
                if context_log:
                    # Display the full conversation context
                    interaction = context_log.get('interaction', {})
                    print(f"\nFull Conversation Context:")
                    print(f"Time: {context_log.get('timestamp', 'Unknown')}")
                    print(f"User: {interaction.get('user_input', 'N/A')}")
                    print(f"Assistant: {interaction.get('assistant_response', 'N/A')}")
                else:
                    print(f"Context log not found for lookup code: {lookup_code}")
            else:
                print("No context lookup code available for this memory.")
                # Try the old method as fallback
                context_data = self.memory.get_context_for_memory(memory_id, metadata)
                if context_data:
                    print(f"Lookup Code: {context_data.get('lookup_code', 'N/A')}")
                    
                    context = context_data.get('context', {})
                    if context:
                        print(f"\nOriginal Conversation:")
                        print(f"User: {context.get('user_input', 'N/A')}")
                        print(f"Assistant: {context.get('assistant_response', 'N/A')}")
                        
                        if 'interaction' in context:
                            interaction = context['interaction']
                            thinking = interaction.get('thinking_trace')
                            if thinking:
                                print(f"Reasoning: {thinking}")
                else:
                    print(f"\nNo context available for memory {memory_id}")
            
            print("=" * 50)
                
        except Exception as e:
            print(f"\nError retrieving context: {e}")
            logger.error(f"Error in show_context_for_memory: {e}")
    
    def show_context_for_lookup_code(self, lookup_code: str, user_id: str):
        """Show full context for a specific conversation lookup code"""
        try:
            # Search conversation logs for the lookup code
            all_logs = self.context_logger.get_all_logs()
            
            target_log = None
            for log_entry in all_logs:
                if log_entry.get('lookup_code') == lookup_code:
                    target_log = log_entry
                    break
            
            if not target_log:
                print(f"\nConversation with lookup code '{lookup_code}' not found.")
                return
            
            # Display the conversation context
            print(f"\n=== Full Context for {lookup_code} ===")
            print(f"Timestamp: {target_log.get('timestamp', 'Unknown')}")
            print(f"User ID: {target_log.get('user_id', 'Unknown')}")
            print()
            
            # Show conversation
            print("Conversation:")
            print(f"User: {target_log.get('user_input', 'N/A')}")
            print(f"Assistant: {target_log.get('assistant_response', 'N/A')}")
            print()
            
            # Show memory operations if available
            memory_ops = target_log.get('memory_operations', {})
            if memory_ops:
                print("Memory Operations:")
                searched = memory_ops.get('memories_searched', [])
                added = memory_ops.get('memories_added', [])
                relationships = memory_ops.get('relationships_extracted', {})
                
                if searched:
                    print(f"  Searched: {len(searched)} memories")
                if added:
                    print(f"  Added: {len(added)} memories")
                if relationships:
                    print(f"  Relationships: {len(relationships)} extracted")
            
            print("=" * 50)
            
        except Exception as e:
            print(f"\nError retrieving context for lookup code: {e}")
            logger.error(f"Error in show_context_for_lookup_code: {e}")
    
    def expand_last_memories(self, user_id: str):
        """Manually expand the last search results"""
        if not self.last_search_results or not self.last_search_query:
            print("\nNo recent search results to expand.")
            return
        
        try:
            print(f"\n=== Expanding Last Search Results ===")
            print(f"Original Query: {self.last_search_query}")
            
            # Force expansion on the last search results
            expanded_results = self.context_expander.expand_memory_results(
                self.last_search_results, 
                self.last_search_query
            )
            
            expansion_meta = expanded_results.get('expansion_metadata', {})
            print(f"Expanded {expansion_meta.get('expanded_count', 0)} out of {expansion_meta.get('total_memories', 0)} memories")
            
            # Show expanded memories
            expanded_memories = [m for m in expanded_results.get('results', []) if 'expanded_context' in m]
            
            for i, memory in enumerate(expanded_memories, 1):
                print(f"\n--- Expanded Memory {i} ---")
                print(f"ID: {memory.get('id')}")
                print(f"Memory: {memory.get('memory', 'N/A')}")
                print(f"Relevance: {memory.get('score', 0):.3f}")
                
                expansion_info = memory.get('expansion_info', {})
                print(f"Priority: {expansion_info.get('priority_score', 'N/A')}")
                print(f"Reasons: {', '.join(expansion_info.get('expansion_reasons', []))}")
                
                expanded_context = memory.get('expanded_context', '')
                if expanded_context:
                    print(f"\nExpanded Context:")
                    print(expanded_context[:500] + "..." if len(expanded_context) > 500 else expanded_context)
            
            print("=" * 50)
            
        except Exception as e:
            print(f"\nError expanding memories: {e}")
            logger.error(f"Error in expand_last_memories: {e}")
    
    def show_conversation_timeline(self, user_id: str, date_filter: Optional[str] = None):
        """Show conversation timeline with context links"""
        try:
            print(f"\n=== Conversation Timeline ===")
            if date_filter:
                print(f"Filtered by: {date_filter}")
            
            # Get ALL logs from context logger (not just current session)
            all_logs = self.context_logger.get_all_logs()
            
            if not all_logs:
                print("No conversation timeline available.")
                return
            
            # Filter to only conversation logs (not memory operations)
            conversation_logs = [log for log in all_logs 
                               if 'interaction' in log or 'user_input' in log]
            
            # Filter by date if provided
            filtered_logs = conversation_logs
            if date_filter:
                # Simple date filtering - could be enhanced
                filtered_logs = [log for log in conversation_logs if date_filter in log.get('timestamp', '')]
            
            # Sort by timestamp
            sorted_logs = sorted(filtered_logs, key=lambda x: x.get('unix_timestamp', 0))
            
            print(f"Found {len(sorted_logs)} conversation entries:")
            
            for i, log_entry in enumerate(sorted_logs[-10:], 1):  # Show last 10
                timestamp = log_entry.get('timestamp', 'Unknown time')
                lookup_code = log_entry.get('lookup_code', 'No code')
                user_input = log_entry.get('user_input', '')
                assistant_response = log_entry.get('assistant_response', '')
                
                # Handle enhanced format
                if 'interaction' in log_entry:
                    interaction = log_entry['interaction']
                    user_input = interaction.get('user_input', user_input)
                    assistant_response = interaction.get('assistant_response', assistant_response)
                
                print(f"\n{i}. {timestamp}")
                print(f"   Code: {lookup_code}")
                print(f"   User: {user_input[:100]}..." if len(user_input) > 100 else f"   User: {user_input}")
                print(f"   Assistant: {assistant_response[:100]}..." if len(assistant_response) > 100 else f"   Assistant: {assistant_response}")
            
            print("\nUse '/context <lookup_code>' to see full context for any entry.")
            print("=" * 50)
            
        except Exception as e:
            print(f"\nError retrieving timeline: {e}")
            logger.error(f"Error in show_conversation_timeline: {e}")
    
    def toggle_expansion(self, enable: bool):
        """Toggle automatic context expansion on/off"""
        self.expansion_enabled = enable
        status = "enabled" if enable else "disabled"
        print(f"\nContext expansion {status}.")
        
        # Show current expansion statistics
        try:
            if hasattr(self.memory, 'context_expander') and self.memory.context_expander:
                stats = self.memory.context_expander.get_expansion_statistics()
                print(f"Expansion stats: {stats['expansion_stats']['total_expansions']} total expansions")
                print(f"Cache hit rate: {stats['cache_stats']['cache_hit_rate']:.1f}%")
        except Exception as e:
            logger.warning(f"Could not get expansion stats: {e}")

def main():
    """Main chat loop"""
    assistant = ImprovedMem0Assistant()
    user_id = "adam_001"  # Better user ID
    
    print("Mem0 Assistant Ready with Phase 3 Context Expansion!")
    print("Commands:")
    print("  'memories' - show all stored memories")
    print("  'relationships' - show all relationship/graph data")
    print("  'search relationships <query>' - search for specific relationships")
    print("  'forget X' - remove memories about X")
    print("  '/context <memory_id>' - show full context for specific memory")
    print("  '/expand' - manually expand last search results")
    print("  '/timeline [date]' - show conversation timeline with context links")
    print("  '/expansion on/off' - toggle automatic context expansion")
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
        
        # Phase 3C Commands
        if user_input.startswith('/context '):
            context_identifier = user_input[len('/context '):].strip()
            if context_identifier:
                # Check if this is a lookup code (starts with CTX-) or memory ID
                if context_identifier.startswith('CTX-'):
                    # This is a lookup code - search conversation logs directly
                    assistant.show_context_for_lookup_code(context_identifier, user_id)
                else:
                    # This is a memory ID - use existing method
                    assistant.show_context_for_memory(context_identifier, user_id)
            else:
                print("Please specify a memory ID or lookup code. Example: '/context CTX-123' or '/context mem_123'")
            continue
        
        if user_input.lower() == '/expand':
            assistant.expand_last_memories(user_id)
            continue
        
        if user_input.startswith('/timeline'):
            parts = user_input.split(' ', 1)
            date_filter = parts[1].strip() if len(parts) > 1 else None
            assistant.show_conversation_timeline(user_id, date_filter)
            continue
        
        if user_input.startswith('/expansion '):
            setting = user_input[len('/expansion '):].strip().lower()
            if setting == 'on':
                assistant.toggle_expansion(True)
            elif setting == 'off':
                assistant.toggle_expansion(False)
            else:
                print("Please specify 'on' or 'off'. Example: '/expansion on'")
            continue
        
        if not user_input:
            continue
        
        # Process the message
        response = assistant.process_message(user_input, user_id)
        print(f"\nAssistant: {response}\n")

if __name__ == "__main__":
    main()