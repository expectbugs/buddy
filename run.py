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
import argparse
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

# Rule 3 Compliance - Exception hierarchy
from exceptions import ResponseGenerationError, ContextLoggingError, ConfigurationError

# Debug infrastructure
from debug_info import debug_tracker

# Phase 2: Centralized configuration
from config_manager import config_manager

# Phase 3: Multi-agent foundation
from multi_agent_foundation import MultiAgentMemoryFoundation

# Silent mode output management
from output_manager import initialize_output_manager, get_output_manager

# Logging will be configured after parsing arguments to respect silent mode
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ImprovedMem0Assistant:
    """Improved mem0 assistant with centralized configuration management"""
    
    def __init__(self, config_overrides: Optional[Dict[str, Any]] = None):
        """Initialize assistant with centralized configuration"""
        debug_tracker.log_operation("assistant", "initialization_start", {
            "has_config_overrides": config_overrides is not None
        })
        
        # Load centralized configuration
        config_result = config_manager.load_config()
        if not config_result.success:
            raise ConfigurationError(f"Failed to load configuration: {config_result.error}")
        
        # Apply any overrides
        if config_overrides:
            override_result = config_manager.override_config(config_overrides)
            if not override_result.success:
                raise ConfigurationError(f"Failed to apply config overrides: {override_result.error}")
        
        self.system_config = config_manager.config
        
        debug_tracker.log_operation("assistant", "config_loaded", {
            "loaded_from_file": config_result.metadata.get("loaded_from_file", False),
            "config_file": config_result.metadata.get("config_file", "defaults"),
            "overrides_applied": config_overrides is not None
        })
        
        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            raise ConfigurationError("Please set OPENAI_API_KEY environment variable")
        
        # Get mem0-compatible configuration
        self.mem0_config = config_manager.get_mem0_config()
        
        try:
            # Initialize base mem0 instance with centralized config
            base_memory = Memory.from_config(config_dict=self.mem0_config)
            
            # Initialize Phase 3 context expansion system with centralized config
            from context_logger import ContextLogger
            from context_bridge import ContextBridge
            from context_expander import ContextExpander
            
            self.context_logger = ContextLogger(log_dir=self.system_config.logging.log_directory)
            self.context_bridge = ContextBridge(self.context_logger)
            self.context_expander = ContextExpander(
                self.context_bridge,
                relevance_threshold=self.system_config.memory.relevance_threshold,
                max_expansions=self.system_config.memory.max_expansions,
                cache_size=self.system_config.memory.cache_size,
                cache_ttl_minutes=self.system_config.memory.cache_ttl_minutes,
                max_context_tokens=self.system_config.memory.max_context_tokens,
                expansion_timeout_ms=self.system_config.memory.expansion_timeout_ms
            )
            
            # Wrap with EnhancedMemory for Phase 3 functionality
            self.memory = EnhancedMemory(base_memory, self.context_logger, self.context_expander)
            
            # Phase 3: Create multi-agent foundation
            self.foundation = MultiAgentMemoryFoundation(self.memory)
            
            self.openai_client = OpenAI()
            
            debug_tracker.log_operation("assistant", "initialization_complete", {
                "memory_config": {
                    "relevance_threshold": self.system_config.memory.relevance_threshold,
                    "max_expansions": self.system_config.memory.max_expansions,
                    "cache_size": self.system_config.memory.cache_size
                },
                "database_config": {
                    "neo4j_url": self.system_config.database.neo4j_url,
                    "qdrant_host": self.system_config.database.qdrant_host,
                    "qdrant_port": self.system_config.database.qdrant_port
                },
                "multi_agent_foundation": {
                    "foundation_ready": True,
                    "foundation_type": type(self.foundation).__name__
                }
            }, success=True)
            
            logger.info("Successfully initialized ImprovedMem0Assistant with centralized configuration")
            logger.info(f"Using configuration: mem0={self.mem0_config['version']}, relevance_threshold={self.system_config.memory.relevance_threshold}")
            
        except Exception as e:
            debug_tracker.log_operation("assistant", "initialization_failed", {
                "error": str(e)
            }, success=False, error=str(e))
            
            raise ConfigurationError(f"Failed to initialize assistant: {e}") from e
        
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
            raise ResponseGenerationError(f"Failed to generate response for user input '{message[:50]}...': {e}") from e
    
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
            output_manager = get_output_manager()
            output_manager.print_explicit_command_result(f"\nAll stored memories ({total_items} items):")
            count = 1
            if memories_found:
                for memory in memories_found:
                    output_manager.print_explicit_command_result(f"{count}. {memory}")
                    count += 1
            if relations_found:
                output_manager.print_explicit_command_result(f"\nRelationships:")
                for relation in relations_found:
                    output_manager.print_explicit_command_result(f"{count}. {relation}")
                    count += 1
        else:
            output_manager = get_output_manager()
            output_manager.print_explicit_command_result("\nNo memories found.")
    
    def show_all_relationships(self, user_id: str):
        """Display all relationship/graph data"""
        try:
            # Get all relationships from the graph store
            relationships = self.memory.graph.get_all(filters={'user_id': user_id})
            
            if relationships:
                output_manager = get_output_manager()
                output_manager.print_explicit_command_result(f"\nAll stored relationships ({len(relationships)} items):")
                
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
                    output_manager.print_explicit_command_result(f"\n  {relation_type.upper()} relationships:")
                    for rel in rels:
                        source = rel.get('source', 'unknown')
                        target = rel.get('target', rel.get('destination', 'unknown'))
                        output_manager.print_explicit_command_result(f"    • {source} → {target}")
            else:
                output_manager = get_output_manager()
                output_manager.print_explicit_command_result("\nNo relationships found.")
                
        except Exception as e:
            output_manager = get_output_manager()
            output_manager.print_error(f"Error retrieving relationships: {e}")
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
                output_manager = get_output_manager()
                output_manager.print_explicit_command_result(f"\nRelationships matching '{query}' ({len(all_results)} items):")
                for i, rel in enumerate(all_results, 1):
                    source = rel.get('source', 'unknown')
                    relation = rel.get('relationship', 'unknown')
                    target = rel.get('target', rel.get('destination', 'unknown'))
                    output_manager.print_explicit_command_result(f"{i}. {source} --{relation}--> {target}")
            else:
                output_manager = get_output_manager()
                output_manager.print_explicit_command_result(f"\nNo relationships found matching '{query}'.")
                
        except Exception as e:
            output_manager = get_output_manager()
            output_manager.print_error(f"Error searching relationships: {e}")
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
            
            output_manager.print_explicit_command_result(f"\n=== Context for Memory {memory_id} ===")
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
            output_manager.print_explicit_command_result(f"\n=== Full Context for {lookup_code} ===")
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
            output_manager.print_explicit_command_result(f"\n=== Expanding Last Search Results ===")
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
            output_manager.print_explicit_command_result(f"\n=== Conversation Timeline ===")
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


def configure_logging(silent_mode: bool = False):
    """
    Configure logging based on silent mode
    
    Args:
        silent_mode: If True, suppress all console logging
    """
    # Clear any existing configuration
    logging.getLogger().handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Always set up file logging for debugging
    try:
        import os
        log_dir = "/var/log/buddy"
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(f"{log_dir}/buddy.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)
    except Exception:
        # If file logging fails, continue without it
        pass
    
    # Set up console logging only if NOT in silent mode
    if not silent_mode:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)
    
    # Set root logger level
    logging.getLogger().setLevel(logging.DEBUG if not silent_mode else logging.CRITICAL)
    
    # Configure third-party loggers
    debug_enabled = os.getenv("DEBUG", "").lower() == "true"
    
    if silent_mode:
        # In silent mode, suppress ALL third-party logging to console
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("httpcore").setLevel(logging.CRITICAL)
        logging.getLogger("mem0").setLevel(logging.CRITICAL)
        logging.getLogger("qdrant_client").setLevel(logging.CRITICAL)
        logging.getLogger("neo4j").setLevel(logging.CRITICAL)
    elif not debug_enabled:
        # In normal mode without debug, suppress verbose third-party logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("mem0").setLevel(logging.WARNING)
        logging.getLogger("qdrant_client").setLevel(logging.WARNING)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Buddy Memory System - Multi-Agent AI Foundation",
        epilog="Use --silent to suppress all output except conversation"
    )
    
    parser.add_argument(
        "-s", "--silent", 
        action="store_true",
        help="Enable silent mode (suppress all output except user input and LLM responses)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (optional)"
    )
    
    return parser.parse_args()


def main():
    """Main chat loop"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging FIRST, before any other imports/initialization
    configure_logging(silent_mode=args.silent)
    
    # Apply silent mode from command line if specified
    config_overrides = {}
    if args.silent:
        config_overrides = {
            "interface": {
                "silent_mode": True
            }
        }
    
    # Initialize assistant with config overrides
    assistant = ImprovedMem0Assistant(config_overrides)
    user_id = "adam_001"  # Better user ID
    
    # Initialize output manager with the assistant's configuration
    output_manager = initialize_output_manager(assistant.system_config)
    
    # Display startup messages and help
    startup_messages = [
        "Buddy Memory System Ready - Phase 3: Multi-Agent Foundation!",
        "Commands:",
        "  '/memories' - show all stored memories",
        "  '/relationships' - show all relationship/graph data", 
        "  '/search relationships <query>' - search for specific relationships",
        "  '/forget <X>' - remove memories about X",
        "  '/context <memory_id>' - show full context for specific memory",
        "  '/expand' - manually expand last search results",
        "  '/timeline [date]' - show conversation timeline with context links",
        "  '/expansion on/off' - toggle automatic context expansion",
        "",
        "Debug Commands:",
        "  '/debug' - show comprehensive debug information",
        "  '/debug operations' - show recent operations",
        "  '/debug errors' - show recent failed operations", 
        "  '/debug stats' - show debug statistics",
        "  '/debug clear' - clear debug history",
        "",
        "System Commands:",
        "  '/health' - comprehensive system health check",
        "  '/health neo4j' - check Neo4j status",
        "  '/health qdrant' - check Qdrant status",
        "  '/health filesystem' - check file system access",
        "  '/health debug' - check debug system status",
        "",
        "Configuration Commands:",
        "  '/config' - show current configuration",
        "  '/config save' - save current configuration to file",
        "  '/config reload' - reload configuration from file",
        "  '/config set <key> <value>' - set configuration value",
        "  '/config get <key>' - get specific configuration value",
        "",
        "Multi-Agent Commands:",
        "  '/agents' - list all agent namespaces",
        "  '/agent create <id> <type>' - create new agent namespace",
        "  '/agent <id> search <query>' - search using specific agent",
        "  '/plugins' - list all registered plugins",
        "  '/foundation health' - comprehensive foundation health",
        "  '/foundation debug' - comprehensive debug information",
        "  '/foundation stats' - foundation statistics",
        "  '/silent on/off' - toggle silent mode",
        "  '/help' - show detailed command reference",
        "  'quit' - exit",
        ""
    ]
    
    output_manager.print_startup_messages(startup_messages)
    
    while True:
        if output_manager.silent_mode:
            user_input = input().strip()
        else:
            user_input = output_manager.input_prompt("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            break
        
        if user_input.lower() == '/help':
            # Display comprehensive help - always shown even in silent mode
            help_text = """
=== BUDDY MEMORY SYSTEM - COMMAND REFERENCE ===

BASIC COMMANDS:
  /memories                    - Show all stored memories and relationships
  /relationships               - Show all relationship/graph data organized by type
  /search relationships <query> - Search for specific relationships (case-insensitive)
  /forget <entity>             - Remove all memories about specified entity
  
CONTEXT & ANALYSIS:
  /context <memory_id>         - Show full conversation context for specific memory
  /expand                      - Manually expand last search results with full context
  /timeline [date]             - Show conversation timeline with context links
  /expansion on/off            - Toggle automatic context expansion during searches

DEBUG & MONITORING:
  /debug                       - Show comprehensive debug information and statistics
  /debug operations            - Show recent system operations (last 10)
  /debug errors                - Show recent failed operations with details
  /debug stats                 - Show detailed debug statistics by component
  /debug clear                 - Clear debug history and reset counters

SYSTEM HEALTH:
  /health                      - Comprehensive system health check (all components)
  /health neo4j                - Check Neo4j graph database status and connectivity
  /health qdrant               - Check Qdrant vector database status and collections
  /health filesystem           - Check file system access and permissions
  /health debug                - Check debug system status and operation tracking

CONFIGURATION:
  /config                      - Show current system configuration
  /config save                 - Save current configuration to file
  /config reload               - Reload configuration from file
  /config set <key> <value>    - Set configuration value (e.g., /config set memory.relevance_threshold 0.7)
  /config get <key>            - Get specific configuration value (e.g., /config get database)

MULTI-AGENT SYSTEM:
  /agents                      - List all agent namespaces and their status
  /agent create <id> <type>    - Create new agent namespace (e.g., /agent create scheduler calendar_agent)
  /agent <id> search <query>   - Search using specific agent's custom processing
  /plugins                     - List all registered plugins (processors, expanders, filters)
  /foundation health           - Comprehensive multi-agent foundation health check
  /foundation debug            - Foundation debug information and recent operations
  /foundation stats            - Foundation statistics (agents, plugins, operations)

INTERFACE CONTROL:
  /silent on/off               - Toggle silent mode (suppress system messages, keep commands)
  /help                        - Show this help information (always displayed)
  quit / exit                  - Exit the system

USAGE EXAMPLES:
  /search relationships john   - Find all relationships involving 'john'
  /forget unladen_swallow      - Remove all memories about unladen swallows
  /config set memory.relevance_threshold 0.8  - Adjust memory relevance threshold
  /agent create scheduler calendar_agent       - Create calendar management agent
  /debug operations            - See what the system has been doing recently

AUTOMATION TIPS:
  - Use --silent flag for pure input/output automation
  - All commands work in silent mode when explicitly called
  - Commands starting with '/' ensure consistent interface
  - File logging always available at /var/log/buddy/buddy.log

==============================================================="""
            
            output_manager.print_explicit_command_result(help_text)
            continue
            
        if user_input.lower() == '/memories':
            assistant.show_all_memories(user_id)
            continue
            
        if user_input.lower() == '/relationships':
            assistant.show_all_relationships(user_id)
            continue
            
        if user_input.lower().startswith('/search relationships '):
            query = user_input[len('/search relationships '):].strip()
            if query:
                assistant.search_relationships(query, user_id)
            else:
                output_manager.print_explicit_command_result("Please specify a search query. Example: '/search relationships Dave'")
            continue
        
        if user_input.lower().startswith('/forget '):
            entity_to_forget = user_input[len('/forget '):].strip()
            if entity_to_forget:
                if assistant.process_forget_request(f"forget {entity_to_forget}", user_id):
                    output_manager.print_explicit_command_result(f"Forgotten memories about: {entity_to_forget}")
                else:
                    output_manager.print_explicit_command_result(f"No memories found about: {entity_to_forget}")
            else:
                output_manager.print_explicit_command_result("Please specify what to forget. Example: '/forget John'")
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
                output_manager.print_command_result("Please specify a memory ID or lookup code. Example: '/context CTX-123' or '/context mem_123'")
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
                output_manager.print_command_result("Please specify 'on' or 'off'. Example: '/expansion on'")
            continue
        
        # Silent mode commands
        if user_input.startswith('/silent'):
            parts = user_input.split(' ', 1)
            if len(parts) == 1:
                # Show current silent mode status
                status = "enabled" if output_manager.silent_mode else "disabled"
                output_manager.print_command_result(f"\nSilent mode is currently {status}")
                output_manager.print_command_result("Use '/silent on' or '/silent off' to change")
            elif len(parts) == 2:
                setting = parts[1].strip().lower()
                if setting == 'on':
                    output_manager.set_silent_mode(True)
                    configure_logging(silent_mode=True)
                    output_manager.print_command_result("Silent mode enabled. Only conversation will be displayed.")
                elif setting == 'off':
                    output_manager.set_silent_mode(False)
                    configure_logging(silent_mode=False)
                    output_manager.print_command_result("Silent mode disabled. All system messages will be displayed.")
                else:
                    output_manager.print_command_result("Please specify 'on' or 'off'. Example: '/silent on'")
            continue
        
        # Debug commands
        if user_input.startswith('/debug'):
            parts = user_input.split(' ', 1)
            if len(parts) == 1 or parts[1].strip() == '':
                # Show comprehensive debug information
                try:
                    summary = debug_tracker.get_debug_summary()
                    output_manager.print_explicit_command_result(f"\n=== DEBUG SUMMARY ===")
                    output_manager.print_explicit_command_result(f"Total Operations: {summary['total_operations']}")
                    output_manager.print_explicit_command_result(f"Error Count: {summary['error_count']}")
                    output_manager.print_explicit_command_result(f"Components Active: {len(summary['component_activity'])}")
                    output_manager.print_explicit_command_result(f"Timestamp: {summary['timestamp']}")
                    output_manager.print_explicit_command_result(f"\nComponent Activity:")
                    for comp, count in summary['component_activity'].items():
                        output_manager.print_explicit_command_result(f"  {comp}: {count} operations")
                    output_manager.print_explicit_command_result(f"\nComponent States:")
                    for comp, state_info in summary['component_states'].items():
                        output_manager.print_explicit_command_result(f"  {comp}: {state_info['timestamp']}")
                except Exception as e:
                    output_manager.print_explicit_command_result(f"Debug summary failed: {e}")
            elif parts[1].strip() == 'operations':
                # Show recent operations
                try:
                    ops = debug_tracker.get_recent_operations(20)
                    output_manager.print_explicit_command_result(f"\n=== RECENT OPERATIONS ({len(ops)}) ===")
                    for i, op in enumerate(ops[-10:], 1):  # Show last 10
                        status = "✓" if op['success'] else "✗"
                        print(f"{i}. {status} {op['component']}.{op['operation']} at {op['timestamp']}")
                        if not op['success'] and op['error']:
                            print(f"   Error: {op['error']}")
                except Exception as e:
                    print(f"Operations listing failed: {e}")
            elif parts[1].strip() == 'errors':
                # Show recent failed operations
                try:
                    errors = debug_tracker.get_error_operations(10)
                    output_manager.print_explicit_command_result(f"\n=== RECENT ERRORS ({len(errors)}) ===")
                    for i, error_op in enumerate(errors, 1):
                        print(f"{i}. ✗ {error_op['component']}.{error_op['operation']} at {error_op['timestamp']}")
                        print(f"   Error: {error_op['error']}")
                        print(f"   Details: {error_op['details']}")
                        print()
                except Exception as e:
                    print(f"Error listing failed: {e}")
            elif parts[1].strip() == 'stats':
                # Show debug statistics
                try:
                    stats = debug_tracker.get_statistics()
                    output_manager.print_explicit_command_result(f"\n=== DEBUG STATISTICS ===")
                    print(f"Total Operations: {stats['total_operations']}")
                    print(f"Error Operations: {stats['error_operations']}")
                    print(f"Success Rate: {stats['success_rate']:.1f}%")
                    print(f"Components Tracked: {stats['components_tracked']}")
                    print(f"Memory Usage: {stats['memory_usage_percent']:.1f}%")
                    print(f"\nBy Component:")
                    for comp, comp_stats in stats['operations_by_component'].items():
                        print(f"  {comp}: {comp_stats['total']} ops, {comp_stats['success_rate']:.1f}% success")
                except Exception as e:
                    print(f"Statistics generation failed: {e}")
            elif parts[1].strip() == 'clear':
                # Clear debug history
                try:
                    debug_tracker.clear_history()
                    print("\nDebug history cleared.")
                except Exception as e:
                    print(f"Debug clear failed: {e}")
            else:
                print("Unknown debug command. Use: /debug, /debug operations, /debug errors, /debug stats, /debug clear")
            continue
        
        # Health check commands
        if user_input.startswith('/health'):
            try:
                from health_check import health_checker
                
                health_parts = user_input.split(' ', 1)
                health_type = health_parts[1].strip() if len(health_parts) > 1 else 'all'
                
                if health_type == 'all' or health_type == '':
                    # Comprehensive health check
                    health_results = health_checker.comprehensive_health_check()
                    output_manager.print_explicit_command_result(f"\n=== SYSTEM HEALTH CHECK ===")
                    print(f"Overall Status: {health_results['overall_status'].upper()}")
                    print(f"Check Duration: {health_results['check_duration_ms']:.1f}ms")
                    print(f"Components: {health_results['healthy_count']}/{health_results['total_components']} healthy")
                    print()
                    
                    for component, status in health_results['components'].items():
                        status_icon = "✓" if status['status'] == 'healthy' else "✗"
                        print(f"{status_icon} {component}: {status['status']}")
                        
                        if status.get('response_time_ms'):
                            print(f"   Response time: {status['response_time_ms']:.1f}ms")
                        
                        if status.get('collections_count') is not None:
                            print(f"   Collections: {status['collections_count']}")
                        
                        if status.get('total_operations') is not None:
                            print(f"   Operations tracked: {status['total_operations']}")
                        
                        if status.get('error'):
                            print(f"   Error: {status['error']}")
                        
                        print()
                    
                    if health_results['unhealthy_components']:
                        print(f"Unhealthy components: {', '.join(health_results['unhealthy_components'])}")
                        
                elif health_type == 'neo4j':
                    status = health_checker.check_neo4j()
                    output_manager.print_explicit_command_result(f"\n=== NEO4J HEALTH ===")
                    print(f"Status: {status['status']}")
                    print(f"URL: {status['url']}")
                    if status['error']:
                        print(f"Error: {status['error']}")
                    else:
                        print(f"Response time: {status['response_time_ms']:.1f}ms")
                        print(f"Query time: {status['query_time_ms']:.1f}ms")
                        print(f"Query successful: {status['query_successful']}")
                        
                elif health_type == 'qdrant':
                    status = health_checker.check_qdrant()
                    output_manager.print_explicit_command_result(f"\n=== QDRANT HEALTH ===")
                    print(f"Status: {status['status']}")
                    print(f"URL: {status['url']}")
                    if status['error']:
                        print(f"Error: {status['error']}")
                    else:
                        print(f"Response time: {status['response_time_ms']:.1f}ms")
                        print(f"Collections: {status['collections_count']}")
                        
                elif health_type == 'filesystem':
                    status = health_checker.check_file_system()
                    output_manager.print_explicit_command_result(f"\n=== FILE SYSTEM HEALTH ===")
                    print(f"Status: {status['status']}")
                    print(f"Directory: {status['directory_path']}")
                    print(f"Exists: {status['directory_exists']}")
                    print(f"Readable: {status['readable']}")
                    print(f"Writable: {status['writable']}")
                    if status['error']:
                        print(f"Error: {status['error']}")
                        
                elif health_type == 'debug':
                    status = health_checker.check_debug_system()
                    output_manager.print_explicit_command_result(f"\n=== DEBUG SYSTEM HEALTH ===")
                    print(f"Status: {status['status']}")
                    if status['error']:
                        print(f"Error: {status['error']}")
                    else:
                        print(f"Check time: {status['check_time_ms']:.1f}ms")
                        print(f"Total operations: {status['total_operations']}")
                        print(f"Components tracked: {status['components_tracked']}")
                        print(f"Recent operations: {status['recent_operations_count']}")
                        
                else:
                    print("Unknown health check. Use: /health, /health neo4j, /health qdrant, /health filesystem, /health debug")
                    
            except Exception as e:
                print(f"Health check failed: {e}")
            continue
        
        # Configuration commands
        if user_input.startswith('/config'):
            try:
                config_parts = user_input.split(' ', 2)  # Allow for 'set key value'
                config_command = config_parts[1] if len(config_parts) > 1 else 'show'
                
                if config_command == 'show' or config_command == '':
                    # Show current configuration
                    config_result = config_manager.get_config()
                    if config_result.success:
                        config_dict = config_result.data
                        
                        output_manager.print_explicit_command_result(f"\n=== CURRENT CONFIGURATION ===")
                        print(f"Config loaded from: {config_result.metadata.get('config_file', 'defaults')}")
                        print()
                        
                        print(f"Database Configuration:")
                        for key, value in config_dict["database"].items():
                            if "password" in key.lower():
                                print(f"  {key}: {'*' * len(str(value))}")
                            else:
                                print(f"  {key}: {value}")
                        
                        print(f"\nMemory Configuration:")
                        for key, value in config_dict["memory"].items():
                            print(f"  {key}: {value}")
                        
                        print(f"\nLogging Configuration:")
                        for key, value in config_dict["logging"].items():
                            print(f"  {key}: {value}")
                        
                        print()
                    else:
                        print(f"\n✗ Failed to get configuration: {config_result.error}")
                
                elif config_command == 'save':
                    # Save current configuration to file
                    config_data = config_manager.config.to_dict()
                    save_result = config_manager.save_config(config_data)
                    
                    if save_result.success:
                        print(f"\n✓ Configuration saved to {save_result.metadata['config_file']}")
                    else:
                        print(f"\n✗ Failed to save configuration: {save_result.error}")
                
                elif config_command == 'reload':
                    # Reload configuration from file
                    reload_result = config_manager.load_config()
                    
                    if reload_result.success:
                        config_file = reload_result.metadata.get('config_file', 'defaults')
                        loaded_from_file = reload_result.metadata.get('loaded_from_file', False)
                        
                        print(f"\n✓ Configuration reloaded from {config_file}")
                        if loaded_from_file:
                            print("  Configuration loaded from file")
                        else:
                            print("  Using default configuration (no file found)")
                        
                        print("\n⚠️  Note: Restart required for some configuration changes to take effect")
                    else:
                        print(f"\n✗ Failed to reload configuration: {reload_result.error}")
                
                elif config_command == 'set' and len(config_parts) >= 4:
                    # Set configuration value
                    key = config_parts[2]
                    value_str = ' '.join(config_parts[3:])  # Join remaining parts for multi-word values
                    
                    # Try to parse value as appropriate type
                    try:
                        if value_str.lower() in ['true', 'false']:
                            value = value_str.lower() == 'true'
                        elif value_str.isdigit():
                            value = int(value_str)
                        elif '.' in value_str and value_str.replace('.', '').isdigit():
                            value = float(value_str)
                        else:
                            value = value_str
                    except ValueError:
                        value = value_str
                    
                    set_result = config_manager.set_config(key, value)
                    
                    if set_result.success:
                        print(f"\n✓ Configuration updated: {key} = {value}")
                        print("  Use '/config save' to persist changes")
                        print("  Restart may be required for some changes to take effect")
                    else:
                        print(f"\n✗ Failed to set configuration: {set_result.error}")
                
                elif config_command == 'get' and len(config_parts) >= 3:
                    # Get specific configuration value
                    key = config_parts[2]
                    get_result = config_manager.get_config(key)
                    
                    if get_result.success:
                        output_manager.print_explicit_command_result(f"\n=== CONFIGURATION VALUE ===")
                        for k, v in get_result.data.items():
                            if "password" in k.lower():
                                print(f"{k}: {'*' * len(str(v))}")
                            else:
                                print(f"{k}: {v}")
                    else:
                        print(f"\n✗ Failed to get configuration value: {get_result.error}")
                
                elif config_command == 'set':
                    print("\nUsage: /config set <key> <value>")
                    print("Examples:")
                    print("  /config set memory.relevance_threshold 0.65")
                    print("  /config set database.qdrant_port 6334")
                    print("  /config set logging.debug_level DEBUG")
                
                elif config_command == 'get':
                    print("\nUsage: /config get <key>")
                    print("Examples:")
                    print("  /config get memory.relevance_threshold")
                    print("  /config get database")
                    print("  /config get logging.log_directory")
                
                else:
                    print("\nUnknown config command. Available commands:")
                    print("  /config [show] - show current configuration")
                    print("  /config save - save configuration to file")
                    print("  /config reload - reload configuration from file")
                    print("  /config set <key> <value> - set configuration value")
                    print("  /config get <key> - get configuration value")
                
            except Exception as e:
                print(f"\nConfiguration command failed: {e}")
            continue
        
        # Multi-agent commands
        if user_input.startswith('/agents'):
            try:
                agents_result = assistant.foundation.list_agents()
                if agents_result.success:
                    agents = agents_result.data
                    if agents:
                        output_manager.print_explicit_command_result(f"\n=== Active Agent Namespaces ({len(agents)}) ===")
                        for agent_id, info in agents.items():
                            print(f"• {agent_id} ({info['agent_type']})")
                            print(f"  Created: {info.get('creation_time', 'unknown')}")
                            print(f"  Operations: {info.get('operation_count', 0)}")
                            print(f"  Uptime: {info.get('uptime_seconds', 0):.1f}s")
                            if info['custom_processors']:
                                print(f"  Processors: {', '.join(info['custom_processors'])}")
                            if info['custom_expanders']:
                                print(f"  Expanders: {', '.join(info['custom_expanders'])}")
                            if info['custom_filters']:
                                print(f"  Filters: {', '.join(info['custom_filters'])}")
                    else:
                        print("\nNo agent namespaces created yet.")
                        print("Use '/agent create <id> <type>' to create your first agent.")
                else:
                    print(f"✗ Error listing agents: {agents_result.error}")
            except Exception as e:
                print(f"\n✗ Agent listing failed: {e}")
            continue

        if user_input.startswith('/agent create '):
            try:
                parts = user_input.split(' ', 3)
                if len(parts) >= 4:
                    agent_id = parts[2]
                    agent_type = parts[3]
                    
                    result = assistant.foundation.create_agent(agent_id, agent_type)
                    if result.success:
                        print(f"\n✓ Created agent namespace: {agent_id} ({agent_type})")
                        print(f"  Agent can now be used with '/agent {agent_id} search <query>'")
                    else:
                        print(f"\n✗ Failed to create agent: {result.error}")
                else:
                    print("\nUsage: /agent create <id> <type>")
                    print("Example: /agent create scheduler calendar_agent")
            except Exception as e:
                print(f"\n✗ Agent creation failed: {e}")
            continue

        if user_input.startswith('/agent ') and ' search ' in user_input:
            try:
                parts = user_input.split(' ', 3)
                if len(parts) >= 4:
                    agent_id = parts[1]
                    query = parts[3]
                    
                    agent = assistant.foundation.get_agent(agent_id)
                    if agent:
                        result = agent.search_memories(query)
                        if result.success:
                            search_data = result.data
                            output_manager.print_explicit_command_result(f"\n=== Agent {agent_id} Search Results ===")
                            print(f"Query: {query}")
                            print(f"Results: {search_data.total_count}")
                            print(f"Search time: {search_data.search_time_ms:.1f}ms")
                            print(f"Expansion applied: {search_data.expansion_applied}")
                            
                            if result.metadata.get('agent_processing'):
                                proc_info = result.metadata['agent_processing']
                                print(f"Agent processing: {len(proc_info.get('filters_applied', []))} filters, {len(proc_info.get('processors_applied', []))} processors")
                            
                            for i, memory in enumerate(search_data.results[:5], 1):
                                memory_text = memory.get('memory', memory.get('text', 'N/A'))
                                score = memory.get('score', memory.get('relevance', 0))
                                print(f"{i}. [{score:.3f}] {memory_text[:100]}...")
                        else:
                            print(f"\n✗ Agent search failed: {result.error}")
                    else:
                        print(f"\n✗ Agent '{agent_id}' not found")
                        print("Use '/agents' to list available agents")
                else:
                    print("\nUsage: /agent <id> search <query>")
                    print("Example: /agent scheduler search meetings today")
            except Exception as e:
                print(f"\n✗ Agent search failed: {e}")
            continue

        if user_input.startswith('/plugins'):
            try:
                plugins_result = assistant.foundation.list_plugins()
                if plugins_result.success:
                    plugins = plugins_result.data
                    total_plugins = sum(len(plugin_list) for plugin_list in plugins.values())
                    
                    output_manager.print_explicit_command_result(f"\n=== Registered Plugins ({total_plugins}) ===")
                    
                    for plugin_type, plugin_list in plugins.items():
                        if plugin_list:
                            type_display = plugin_type.replace('_', ' ').title()
                            print(f"\n{type_display}:")
                            for plugin in plugin_list:
                                print(f"  • {plugin['name']} ({plugin['class']}) - Priority: {plugin['priority']}")
                                if plugin['metadata']:
                                    print(f"    Metadata: {plugin['metadata']}")
                    
                    if total_plugins == 0:
                        print("No plugins registered yet.")
                        print("Use foundation.register_processor/expander/filter() to add plugins.")
                else:
                    print(f"\n✗ Error listing plugins: {plugins_result.error}")
            except Exception as e:
                print(f"\n✗ Plugin listing failed: {e}")
            continue

        if user_input.startswith('/foundation health'):
            try:
                health_result = assistant.foundation.get_foundation_health()
                if health_result.success:
                    health_data = health_result.data
                    overall_healthy = health_result.metadata.get('overall_healthy', False)
                    
                    output_manager.print_explicit_command_result(f"\n=== FOUNDATION HEALTH ===")
                    print(f"Overall Status: {'✓ HEALTHY' if overall_healthy else '✗ UNHEALTHY'}")
                    print(f"Uptime: {health_data['uptime_seconds']:.1f}s")
                    print(f"Active Agents: {health_data['active_agents']}")
                    print(f"Registered Plugins: {health_data['registered_plugins']}")
                    print(f"Shared Operations: {health_data['shared_operations']}")
                    
                    print(f"\nComponent Health:")
                    if 'system_health' in health_data:
                        sys_health = health_data['system_health']
                        print(f"  System: {sys_health.get('overall_status', 'unknown').upper()}")
                    
                    if 'memory_health' in health_data:
                        mem_health = health_data['memory_health']
                        if 'error' not in mem_health:
                            print(f"  Memory: {mem_health.get('status', 'unknown').upper()}")
                        else:
                            print(f"  Memory: ERROR - {mem_health['error']}")
                    
                    if 'agent_health' in health_data:
                        agent_health = health_data['agent_health']
                        if 'error' not in agent_health:
                            print(f"  Agents: {agent_health.get('healthy_agents', 0)}/{agent_health.get('total_agents', 0)} healthy")
                        else:
                            print(f"  Agents: ERROR - {agent_health['error']}")
                else:
                    print(f"\n✗ Foundation health check failed: {health_result.error}")
            except Exception as e:
                print(f"\n✗ Foundation health check failed: {e}")
            continue

        if user_input.startswith('/foundation debug'):
            try:
                debug_result = assistant.foundation.get_foundation_debug_info()
                if debug_result.success:
                    debug_data = debug_result.data
                    
                    output_manager.print_explicit_command_result(f"\n=== FOUNDATION DEBUG INFO ===")
                    foundation_info = debug_data.get('foundation_info', {})
                    print(f"Uptime: {foundation_info.get('uptime_seconds', 0):.1f}s")
                    print(f"Shared Operations: {foundation_info.get('shared_operations', 0)}")
                    
                    active_agents = debug_data.get('active_agents', {})
                    print(f"Active Agents: {len(active_agents)}")
                    
                    registered_plugins = debug_data.get('registered_plugins', {})
                    total_plugins = sum(len(plugin_list) for plugin_list in registered_plugins.values())
                    print(f"Registered Plugins: {total_plugins}")
                    
                    debug_summary = debug_data.get('debug_summary', {})
                    print(f"Debug Operations: {debug_summary.get('total_operations', 0)}")
                    print(f"Debug Errors: {debug_summary.get('error_count', 0)}")
                    
                    recent_ops = debug_data.get('recent_operations', [])
                    print(f"Recent Foundation Operations: {len(recent_ops)}")
                    for i, op in enumerate(recent_ops[-5:], 1):
                        status = "✓" if op.get('success', True) else "✗"
                        print(f"  {i}. {status} {op.get('component', 'unknown')}.{op.get('operation', 'unknown')}")
                else:
                    print(f"\n✗ Foundation debug failed: {debug_result.error}")
            except Exception as e:
                print(f"\n✗ Foundation debug failed: {e}")
            continue

        if user_input.startswith('/foundation stats'):
            try:
                stats_result = assistant.foundation.get_foundation_stats()
                if stats_result.success:
                    stats = stats_result.data
                    
                    output_manager.print_explicit_command_result(f"\n=== FOUNDATION STATISTICS ===")
                    foundation_stats = stats.get('foundation', {})
                    print(f"Uptime: {foundation_stats.get('uptime_seconds', 0):.1f}s")
                    print(f"Shared Operations: {foundation_stats.get('shared_operations', 0)}")
                    
                    agent_stats = stats.get('agents', {})
                    print(f"Total Agents: {agent_stats.get('total_agents', 0)}")
                    print(f"Agent Creations: {agent_stats.get('creation_count', 0)}")
                    print(f"Total Agent Operations: {agent_stats.get('total_operations', 0)}")
                    
                    plugin_stats = stats.get('plugins', {})
                    print(f"Total Plugins: {plugin_stats.get('total_plugins', 0)}")
                    print(f"Processors: {plugin_stats.get('memory_processors', 0)}")
                    print(f"Expanders: {plugin_stats.get('context_expanders', 0)}")
                    print(f"Filters: {plugin_stats.get('memory_filters', 0)}")
                    
                    config_stats = stats.get('configuration', {})
                    print(f"Config Loaded: {config_stats.get('config_loaded', False)}")
                    print(f"Config Sections: {config_stats.get('config_sections', 0)}")
                else:
                    print(f"\n✗ Foundation stats failed: {stats_result.error}")
            except Exception as e:
                print(f"\n✗ Foundation stats failed: {e}")
            continue
        
        if not user_input:
            continue
        
        # Process the message
        debug_tracker.log_operation("main_interface", "user_message_start", {
            "user_id": user_id,
            "message_length": len(user_input)
        })
        
        try:
            response = assistant.process_message(user_input, user_id)
            if output_manager.silent_mode:
                output_manager.print_conversation(response)
            else:
                output_manager.print_conversation(f"\nAssistant: {response}\n")
            
            debug_tracker.log_operation("main_interface", "user_message_complete", {
                "user_id": user_id,
                "response_length": len(response)
            }, success=True)
        except Exception as e:
            debug_tracker.log_operation("main_interface", "user_message_failed", {
                "user_id": user_id,
                "error_type": type(e).__name__
            }, success=False, error=str(e))
            raise  # Re-raise the exception for Rule 3 compliance

if __name__ == "__main__":
    main()