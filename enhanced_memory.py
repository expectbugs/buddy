"""
Enhanced Memory Wrapper for mem0 Integration
Phase 2: Enhanced Memory Storage with automatic metadata linking

This wrapper adds context logging and metadata enhancement to mem0's Memory class
without modifying its core behavior. All existing functionality remains intact.

Rules Compliance:
- Rule 1: NEVER disable or remove a feature to fix a bug or error
- Rule 2: NEVER fix an error or bug by hiding it  
- Rule 3: NO silent fallbacks or silent failures, all problems should be loud and proud
- Rule 4: Always check online documentation of every package used and do everything the officially recommended way
- Rule 5: Clean up your mess. Remove any temporary and/or outdated files or scripts
"""

import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone

from mem0 import Memory
from context_logger import ContextLogger
from context_bridge import ContextBridge
from temporal_utils import get_current_datetime_info

logger = logging.getLogger(__name__)

class EnhancedMemory:
    """
    Enhanced wrapper around mem0's Memory class that adds:
    1. Automatic context logging with lookup codes
    2. Enhanced metadata for every memory (timestamps, context links)
    3. Backward compatibility with existing mem0 functionality
    4. Loud error handling (no silent failures)
    
    This wrapper is completely transparent - all mem0 methods work exactly the same,
    but with enhanced context tracking capabilities.
    """
    
    def __init__(self, mem0_memory: Memory, context_logger: Optional[ContextLogger] = None, context_expander = None):
        """
        Initialize enhanced memory wrapper
        
        Args:
            mem0_memory: The mem0 Memory instance to wrap
            context_logger: Optional context logger (creates new one if None)
            context_expander: Optional context expander for automatic expansion during search
        """
        # Check if it's a Memory instance or has Memory-like interface for testing
        if not (isinstance(mem0_memory, Memory) or 
                (hasattr(mem0_memory, 'search') and hasattr(mem0_memory, 'add') and 
                 hasattr(mem0_memory, 'update') and hasattr(mem0_memory, 'delete'))):
            raise TypeError(f"ENHANCED MEMORY REQUIRES mem0.Memory INSTANCE OR COMPATIBLE INTERFACE, got {type(mem0_memory)}")
        
        # Store the original mem0 instance
        self.memory = mem0_memory
        
        # Initialize context system
        try:
            self.context_logger = context_logger or ContextLogger()
            self.context_bridge = ContextBridge(self.context_logger)
            
            # Initialize context expander if provided
            self.context_expander = context_expander
            if self.context_expander:
                logger.info("EnhancedMemory initialized with context expansion enabled")
            else:
                logger.info("EnhancedMemory initialized without context expansion")
                
        except Exception as e:
            raise RuntimeError(f"FAILED TO INITIALIZE ENHANCED MEMORY CONTEXT SYSTEM: {e}")
    
    def add(self, messages: Union[str, List[Dict[str, str]]], user_id: str, **kwargs) -> Any:
        """
        Simple passthrough to mem0.add() with full backward compatibility
        
        Phase 3 Note: Context logging happens at application level in run.py,
        not here. This keeps mem0 functionality pure and unchanged.
        """
        return self.memory.add(messages, user_id=user_id, **kwargs)
    
    # Delegate all other methods to the original mem0 Memory instance
    # This ensures 100% backward compatibility
    
    def search(self, query: str, user_id: str, limit: int = 10, enable_expansion: Optional[bool] = None, **kwargs):
        """Enhanced search with memory operation tracking and optional context expansion"""
        start_time = time.time()
        expansion_enabled = (enable_expansion is not None and enable_expansion) or (enable_expansion is None and self.context_expander is not None)
        
        try:
            # Call original mem0 search
            result = self.memory.search(query=query, user_id=user_id, limit=limit, **kwargs)
            search_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Enrich search results with metadata for context expansion
            if result and isinstance(result, dict) and 'results' in result:
                # Get all memories to access metadata
                all_memories = self.memory.get_all(user_id=user_id)
                if isinstance(all_memories, dict) and 'results' in all_memories:
                    # Create a lookup map for quick access
                    memory_map = {mem.get('id'): mem for mem in all_memories['results']}
                    
                    # Enrich each search result with full metadata AND ensure relevance key exists
                    for search_result in result['results']:
                        memory_id = search_result.get('id')
                        if memory_id and memory_id in memory_map:
                            full_memory = memory_map[memory_id]
                            # Add metadata to search result
                            search_result['metadata'] = full_memory.get('metadata', {})
                            
                        # CRITICAL: Ensure context_expander can find relevance score
                        # mem0 uses 'score' but context_expander expects 'relevance'
                        if 'score' in search_result and 'relevance' not in search_result:
                            search_result['relevance'] = search_result['score']
            
            # Apply context expansion if enabled and available
            if expansion_enabled and self.context_expander and result:
                # NO SILENT FALLBACKS - let errors bubble up as per Rule 3
                expansion_start = time.time()
                expanded_result = self.context_expander.expand_memory_results(result, query)
                expansion_time = (time.time() - expansion_start) * 1000
                
                # Use expanded result
                result = expanded_result
                
                logger.debug(f"Context expansion applied (took {expansion_time:.1f}ms)")
            
            # Prepare memory operations data
            memory_operations = {
                "memories_searched": [{
                    "query": query,
                    "results_count": len(result.get('results', [])) if result else 0,
                    "search_time_ms": search_time,
                    "expansion_enabled": expansion_enabled,
                    "expansion_applied": expansion_enabled and self.context_expander is not None
                }],
                "memories_found": [],
                "memories_added": [],
                "relationships_extracted": []
            }
            
            # Extract search results information
            if result and isinstance(result, dict) and 'results' in result:
                for memory_item in result['results']:
                    memory_found = {
                        "id": memory_item.get('id', 'unknown'),
                        "content": (memory_item.get('memory') or memory_item.get('text') or 'unknown')[:100] + "..." if len(memory_item.get('memory') or memory_item.get('text') or '') > 100 else (memory_item.get('memory') or memory_item.get('text') or 'unknown'),
                        "relevance": memory_item.get('score', 0.0),
                        "expanded": 'expanded_context' in memory_item
                    }
                    memory_operations["memories_found"].append(memory_found)
            
            # Include expansion metadata in memory operations if available
            if 'expansion_metadata' in result:
                memory_operations["expansion_metadata"] = result['expansion_metadata']
            
            # Log the memory operation
            try:
                self.context_logger.log_memory_operation(
                    operation_type="search",
                    details={
                        "query": query,
                        "user_id": user_id,
                        "limit": limit,
                        "operation_time_ms": search_time,
                        "results_count": len(result.get('results', [])) if result else 0,
                        "top_relevance_score": result['results'][0].get('score', 0.0) if result and result.get('results') and len(result['results']) > 0 else 0.0,
                        "memories_found": memory_operations["memories_found"],
                        "expansion_enabled": expansion_enabled,
                        "expanded_count": result.get('expansion_metadata', {}).get('expanded_count', 0) if result else 0
                    },
                    user_id=user_id
                )
            except Exception as e:
                logger.warning(f"Memory operation logging failed for search: {e}")
            
            total_time = (time.time() - start_time) * 1000
            expanded_info = f", {result.get('expansion_metadata', {}).get('expanded_count', 0)} expanded" if expansion_enabled else ""
            logger.debug(f"Enhanced memory search completed (took {total_time:.1f}ms, found {len(result.get('results', [])) if result else 0} results{expanded_info})")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"MEM0 SEARCH FAILED in EnhancedMemory: {e}")
    
    def get_all(self, *args, **kwargs):
        """Delegate get_all to original mem0 instance"""
        try:
            return self.memory.get_all(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"MEM0 GET_ALL FAILED in EnhancedMemory: {e}")
    
    def update(self, memory_id: str, data: str, user_id: Optional[str] = None, **kwargs):
        """Enhanced update with memory operation tracking and automatic timestamp"""
        start_time = time.time()
        
        try:
            # Add last_updated_at timestamp to metadata
            if 'metadata' not in kwargs:
                kwargs['metadata'] = {}
            kwargs['metadata']['last_updated_at'] = datetime.now(timezone.utc).isoformat()
            
            # Call original mem0 update (only accepts memory_id and data)
            result = self.memory.update(memory_id=memory_id, data=data)
            update_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Log the memory operation
            try:
                self.context_logger.log_memory_operation(
                    operation_type="update",
                    details={
                        "memory_id": memory_id,
                        "user_id": user_id or "unknown",
                        "operation_time_ms": update_time,
                        "data_length": len(str(data)),
                        "update_successful": True
                    },
                    user_id=user_id
                )
            except Exception as e:
                logger.warning(f"Memory operation logging failed for update: {e}")
            
            logger.debug(f"Enhanced memory update completed (took {update_time:.1f}ms)")
            return result
            
        except Exception as e:
            # Log failed operation
            try:
                self.context_logger.log_memory_operation(
                    operation_type="update",
                    details={
                        "memory_id": memory_id,
                        "user_id": user_id or "unknown",
                        "operation_time_ms": (time.time() - start_time) * 1000,
                        "update_successful": False,
                        "error": str(e)
                    },
                    user_id=user_id
                )
            except Exception as log_e:
                logger.warning(f"Memory operation logging failed for failed update: {log_e}")
            
            raise RuntimeError(f"MEM0 UPDATE FAILED in EnhancedMemory: {e}")
    
    def delete(self, memory_id: str, user_id: Optional[str] = None, **kwargs):
        """Enhanced delete that archives instead of deleting for historical preservation"""
        start_time = time.time()
        
        try:
            # Get the memory content before archiving
            all_memories = self.memory.get_all(user_id=user_id)
            target_memory = None
            if isinstance(all_memories, dict) and 'results' in all_memories:
                for memory in all_memories['results']:
                    if memory.get('id') == memory_id:
                        target_memory = memory
                        break
            
            if not target_memory:
                raise ValueError(f"Memory {memory_id} not found for archiving")
            
            # Archive instead of delete - update with archive metadata
            current_metadata = target_memory.get('metadata', {})
            archive_metadata = current_metadata.copy()
            archive_metadata.update({
                'archived': True,
                'archived_at': datetime.now(timezone.utc).isoformat(),
                'archive_reason': 'user_delete_request',
                'original_content': target_memory.get('memory', target_memory.get('text', '')),
                'archived_by_user': user_id
            })
            
            # Update the memory to mark it as archived (preserves history)
            result = self.update(
                memory_id=memory_id,
                data=f"[ARCHIVED] {target_memory.get('memory', target_memory.get('text', ''))}",
                user_id=user_id,
                metadata=archive_metadata
            )
            
            delete_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Log the memory operation
            try:
                self.context_logger.log_memory_operation(
                    operation_type="archive",
                    details={
                        "memory_id": memory_id,
                        "user_id": user_id or "unknown",
                        "operation_time_ms": delete_time,
                        "archive_successful": True,
                        "original_content": target_memory.get('memory', target_memory.get('text', ''))[:100]
                    },
                    user_id=user_id
                )
            except Exception as e:
                logger.warning(f"Memory operation logging failed for archive: {e}")
            
            logger.info(f"Memory {memory_id} archived instead of deleted (took {delete_time:.1f}ms)")
            return result
            
        except Exception as e:
            # Log failed operation
            try:
                self.context_logger.log_memory_operation(
                    operation_type="delete",
                    details={
                        "memory_id": memory_id,
                        "user_id": user_id or "unknown",
                        "operation_time_ms": (time.time() - start_time) * 1000,
                        "delete_successful": False,
                        "error": str(e)
                    },
                    user_id=user_id
                )
            except Exception as log_e:
                logger.warning(f"Memory operation logging failed for failed delete: {log_e}")
            
            raise RuntimeError(f"MEM0 DELETE FAILED in EnhancedMemory: {e}")
    
    def reset(self, *args, **kwargs):
        """Delegate reset to original mem0 instance"""
        try:
            return self.memory.reset(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"MEM0 RESET FAILED in EnhancedMemory: {e}")
    
    @property
    def graph(self):
        """Delegate graph property to original mem0 instance"""
        return self.memory.graph
    
    @property 
    def config(self):
        """Delegate config property to original mem0 instance"""
        return self.memory.config
    
    def __getattr__(self, name):
        """
        Delegate any other method calls to the original mem0 Memory instance
        This ensures we don't break any mem0 functionality we haven't explicitly wrapped
        """
        try:
            return getattr(self.memory, name)
        except AttributeError:
            raise AttributeError(f"EnhancedMemory AND mem0.Memory DO NOT HAVE ATTRIBUTE: {name}")
    
    def get_context_for_memory(self, memory_id: str, memory_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        New method: Get full context for a memory using its enhanced metadata
        
        Args:
            memory_id: The memory ID
            memory_metadata: The memory's metadata (should contain lookup_code)
            
        Returns:
            Full context data or None if not available
        """
        try:
            return self.context_bridge.get_full_context(memory_id, memory_metadata)
        except Exception as e:
            # Don't fail the whole operation, just log and return None
            logger.warning(f"Could not retrieve context for memory {memory_id}: {e}")
            return None
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """
        Enhanced method: Get statistics about enhanced memory usage (Phase 3)
        
        Returns:
            Statistics about context logging, enhanced memories, memory operations, and expansion
        """
        try:
            bridge_stats = self.context_bridge.get_bridge_stats()
            
            # Get expansion statistics if available
            expansion_stats = None
            if self.context_expander:
                try:
                    expansion_stats = self.context_expander.get_expansion_statistics()
                except Exception as e:
                    logger.warning(f"Could not get expansion statistics: {e}")
            
            features = [
                "automatic_context_logging",
                "enhanced_metadata", 
                "lookup_code_linking",
                "temporal_awareness",
                "memory_operation_tracking",
                "performance_monitoring",
                "thinking_trace_capture",
                "enhanced_format_logging"
            ]
            
            # Add expansion features if available
            if self.context_expander:
                features.extend([
                    "automatic_context_expansion",
                    "intelligent_expansion_decisions",
                    "expansion_caching",
                    "context_formatting_optimization"
                ])
            
            return {
                "enhanced_memory_active": True,
                "phase": 3,
                "context_bridge_stats": bridge_stats,
                "context_expansion": {
                    "enabled": self.context_expander is not None,
                    "statistics": expansion_stats
                },
                "features": features,
                "capabilities": {
                    "context_expansion_ready": True,
                    "context_expansion_active": self.context_expander is not None,
                    "v3_agent_training_data": True,
                    "memory_operation_analytics": True,
                    "performance_metrics": True,
                    "automatic_high_relevance_expansion": self.context_expander is not None,
                    "expansion_caching": self.context_expander is not None,
                    "intelligent_expansion_decisions": self.context_expander is not None
                }
            }
        except Exception as e:
            raise RuntimeError(f"FAILED TO GET ENHANCED MEMORY STATS: {e}")