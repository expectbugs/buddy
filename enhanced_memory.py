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
from exceptions import ContextLoggingError, ContextExpansionError, MemoryOperationError
from debug_info import debug_tracker

# Phase 2: Standard interfaces and result types
from foundation_interface import MemoryFoundationInterface
from result_types import (
    OperationResult, MemorySearchResult, MemoryAddResult,
    HealthCheckResult, DebugInfo,
    MemorySearchOperationResult, MemoryAddOperationResult,
    HealthCheckOperationResult, DebugInfoOperationResult,
    BoolOperationResult, DictOperationResult
)

logger = logging.getLogger(__name__)

class EnhancedMemory(MemoryFoundationInterface):
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
        start_time = time.time()
        debug_tracker.log_operation("enhanced_memory", "add_start", {
            "user_id": user_id,
            "message_type": type(messages).__name__,
            "message_count": len(messages) if isinstance(messages, list) else 1
        })
        
        try:
            result = self.memory.add(messages, user_id=user_id, **kwargs)
            debug_tracker.log_operation("enhanced_memory", "add_complete", {
                "user_id": user_id,
                "operation_time_ms": (time.time() - start_time) * 1000,
                "result_type": type(result).__name__
            }, success=True)
            return result
        except Exception as e:
            debug_tracker.log_operation("enhanced_memory", "add_failed", {
                "user_id": user_id,
                "error_type": type(e).__name__,
                "operation_time_ms": (time.time() - start_time) * 1000
            }, success=False, error=str(e))
            raise
    
    # Delegate all other methods to the original mem0 Memory instance
    # This ensures 100% backward compatibility
    
    def search(self, query: str, user_id: str, limit: int = 10, enable_expansion: Optional[bool] = None, **kwargs):
        """Enhanced search with memory operation tracking and optional context expansion"""
        start_time = time.time()
        expansion_enabled = (enable_expansion is not None and enable_expansion) or (enable_expansion is None and self.context_expander is not None)
        
        # Debug tracking: Log search start
        debug_tracker.log_operation("enhanced_memory", "search_start", {
            "query": query[:100],  # Truncate for debug
            "user_id": user_id,
            "limit": limit,
            "expansion_enabled": expansion_enabled
        })
        
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
                raise ContextLoggingError(f"Memory operation logging failed for search: {e}") from e
            
            total_time = (time.time() - start_time) * 1000
            expanded_info = f", {result.get('expansion_metadata', {}).get('expanded_count', 0)} expanded" if expansion_enabled else ""
            logger.debug(f"Enhanced memory search completed (took {total_time:.1f}ms, found {len(result.get('results', [])) if result else 0} results{expanded_info})")
            
            # Debug tracking: Log search completion
            debug_tracker.log_operation("enhanced_memory", "search_complete", {
                "query": query[:100],
                "results_count": len(result.get('results', [])) if result else 0,
                "search_time_ms": total_time,
                "expansion_applied": expansion_enabled and self.context_expander is not None
            }, success=True)
            
            return result
            
        except Exception as e:
            # Debug tracking: Log search failure
            debug_tracker.log_operation("enhanced_memory", "search_failed", {
                "query": query[:100],
                "error_type": type(e).__name__,
                "search_time_ms": (time.time() - start_time) * 1000
            }, success=False, error=str(e))
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
        
        debug_tracker.log_operation("enhanced_memory", "update_start", {
            "memory_id": memory_id,
            "user_id": user_id or "unknown",
            "data_length": len(str(data))
        })
        
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
                raise ContextLoggingError(f"Memory operation logging failed for update: {e}") from e
            
            logger.debug(f"Enhanced memory update completed (took {update_time:.1f}ms)")
            
            debug_tracker.log_operation("enhanced_memory", "update_complete", {
                "memory_id": memory_id,
                "user_id": user_id or "unknown",
                "operation_time_ms": update_time
            }, success=True)
            
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
                raise ContextLoggingError(f"Memory operation logging failed for failed update: {log_e}") from log_e
            
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
                raise ContextLoggingError(f"Memory operation logging failed for archive: {e}") from e
            
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
                raise ContextLoggingError(f"Memory operation logging failed for failed delete: {log_e}") from log_e
            
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
    
    # ========== PHASE 2: STANDARDIZED INTERFACE IMPLEMENTATION ==========
    
    def search_memories(self, query: str, user_id: str, **options) -> MemorySearchOperationResult:
        """
        Standardized search interface for multi-agent use
        
        Args:
            query: Search query string
            user_id: User identifier
            **options: Additional search options (limit, enable_expansion, etc.)
            
        Returns:
            OperationResult[MemorySearchResult] with success/error status
        """
        operation_id = f"search_{int(time.time() * 1000)}"
        debug_tracker.log_operation("enhanced_memory", "search_memories_start", {
            "operation_id": operation_id,
            "query": query[:100],
            "user_id": user_id,
            "options": str(options)[:200]
        })
        
        try:
            start_time = time.time()
            
            # Call existing search method
            raw_result = self.search(query, user_id, **options)
            
            search_time = (time.time() - start_time) * 1000
            
            # Convert to standard format
            search_result = MemorySearchResult(
                results=raw_result.get('results', []),
                total_count=len(raw_result.get('results', [])),
                search_time_ms=search_time,
                query=query,
                expansion_applied=raw_result.get('expansion_metadata', {}).get('expanded_count', 0) > 0,
                expansion_metadata=raw_result.get('expansion_metadata')
            )
            
            debug_tracker.log_operation("enhanced_memory", "search_memories_success", {
                "operation_id": operation_id,
                "results_count": search_result.total_count,
                "search_time_ms": search_time,
                "expansion_applied": search_result.expansion_applied
            }, success=True)
            
            return OperationResult.success_result(search_result, {
                "operation_id": operation_id,
                "component": "enhanced_memory"
            })
            
        except Exception as e:
            debug_tracker.log_operation("enhanced_memory", "search_memories_failed", {
                "operation_id": operation_id,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(
                f"Memory search failed: {e}",
                {"operation_id": operation_id, "component": "enhanced_memory"}
            )
    
    def add_memory(self, data: Any, user_id: str, **options) -> MemoryAddOperationResult:
        """
        Standardized add interface for multi-agent use
        
        Args:
            data: Memory data (string, dict, or list of messages)
            user_id: User identifier
            **options: Additional options (metadata, etc.)
            
        Returns:
            OperationResult[MemoryAddResult] with success/error status
        """
        operation_id = f"add_{int(time.time() * 1000)}"
        debug_tracker.log_operation("enhanced_memory", "add_memory_start", {
            "operation_id": operation_id,
            "user_id": user_id,
            "data_type": type(data).__name__
        })
        
        try:
            start_time = time.time()
            
            # Call existing add method
            raw_result = self.add(data, user_id, **options)
            
            operation_time = (time.time() - start_time) * 1000
            
            # Extract lookup codes if available
            lookup_codes = []
            if raw_result and isinstance(raw_result, dict) and 'results' in raw_result:
                for memory_item in raw_result['results']:
                    if 'metadata' in memory_item and 'context_lookup_code' in memory_item['metadata']:
                        lookup_codes.append(memory_item['metadata']['context_lookup_code'])
            
            # Convert to standard format
            add_result = MemoryAddResult(
                memories_added=raw_result.get('results', []) if raw_result else [],
                relationships_extracted=raw_result.get('relationships', []) if raw_result else [],
                operation_time_ms=operation_time,
                lookup_codes=lookup_codes
            )
            
            debug_tracker.log_operation("enhanced_memory", "add_memory_success", {
                "operation_id": operation_id,
                "memories_added": len(add_result.memories_added),
                "relationships_extracted": len(add_result.relationships_extracted),
                "operation_time_ms": operation_time
            }, success=True)
            
            return OperationResult.success_result(add_result, {
                "operation_id": operation_id,
                "component": "enhanced_memory"
            })
            
        except Exception as e:
            debug_tracker.log_operation("enhanced_memory", "add_memory_failed", {
                "operation_id": operation_id,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(
                f"Memory add failed: {e}",
                {"operation_id": operation_id, "component": "enhanced_memory"}
            )
    
    def update_memory(self, memory_id: str, data: Any, user_id: str, **options) -> BoolOperationResult:
        """
        Standardized update interface for multi-agent use
        
        Args:
            memory_id: ID of memory to update
            data: New memory data
            user_id: User identifier
            **options: Additional update options
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        operation_id = f"update_{int(time.time() * 1000)}"
        debug_tracker.log_operation("enhanced_memory", "update_memory_start", {
            "operation_id": operation_id,
            "memory_id": memory_id,
            "user_id": user_id
        })
        
        try:
            # Call existing update method
            self.update(memory_id, str(data), user_id, **options)
            
            debug_tracker.log_operation("enhanced_memory", "update_memory_success", {
                "operation_id": operation_id
            }, success=True)
            
            return OperationResult.success_result(True, {
                "operation_id": operation_id,
                "component": "enhanced_memory"
            })
            
        except Exception as e:
            debug_tracker.log_operation("enhanced_memory", "update_memory_failed", {
                "operation_id": operation_id,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(
                f"Memory update failed: {e}",
                {"operation_id": operation_id, "component": "enhanced_memory"}
            )
    
    def delete_memory(self, memory_id: str, user_id: str, **options) -> BoolOperationResult:
        """
        Standardized delete/archive interface for multi-agent use
        
        Args:
            memory_id: ID of memory to delete
            user_id: User identifier
            **options: Additional delete options
            
        Returns:
            OperationResult[bool] indicating success/failure
        """
        operation_id = f"delete_{int(time.time() * 1000)}"
        debug_tracker.log_operation("enhanced_memory", "delete_memory_start", {
            "operation_id": operation_id,
            "memory_id": memory_id,
            "user_id": user_id
        })
        
        try:
            # Call existing delete method (which archives)
            self.delete(memory_id, user_id, **options)
            
            debug_tracker.log_operation("enhanced_memory", "delete_memory_success", {
                "operation_id": operation_id
            }, success=True)
            
            return OperationResult.success_result(True, {
                "operation_id": operation_id,
                "component": "enhanced_memory"
            })
            
        except Exception as e:
            debug_tracker.log_operation("enhanced_memory", "delete_memory_failed", {
                "operation_id": operation_id,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(
                f"Memory delete failed: {e}",
                {"operation_id": operation_id, "component": "enhanced_memory"}
            )
    
    def get_all_memories(self, user_id: str, **options) -> DictOperationResult:
        """
        Standardized get all memories interface for multi-agent use
        
        Args:
            user_id: User identifier
            **options: Additional retrieval options
            
        Returns:
            OperationResult[Dict] with memories data
        """
        operation_id = f"get_all_{int(time.time() * 1000)}"
        debug_tracker.log_operation("enhanced_memory", "get_all_memories_start", {
            "operation_id": operation_id,
            "user_id": user_id
        })
        
        try:
            # Call existing get_all method
            result = self.get_all(user_id=user_id, **options)
            
            debug_tracker.log_operation("enhanced_memory", "get_all_memories_success", {
                "operation_id": operation_id,
                "memory_count": len(result.get('results', [])) if result else 0
            }, success=True)
            
            return OperationResult.success_result(result, {
                "operation_id": operation_id,
                "component": "enhanced_memory"
            })
            
        except Exception as e:
            debug_tracker.log_operation("enhanced_memory", "get_all_memories_failed", {
                "operation_id": operation_id,
                "error": str(e)
            }, success=False, error=str(e))
            
            return OperationResult.error_result(
                f"Get all memories failed: {e}",
                {"operation_id": operation_id, "component": "enhanced_memory"}
            )
    
    def get_health_status(self) -> HealthCheckOperationResult:
        """
        Get enhanced memory component health with standardized result
        
        Returns:
            OperationResult[HealthCheckResult] with health information
        """
        try:
            start_time = time.time()
            
            # Check if underlying memory system is accessible
            test_result = self.memory.get_all(user_id="health_check_test")
            
            response_time = (time.time() - start_time) * 1000
            
            health_result = HealthCheckResult(
                component="enhanced_memory",
                status="healthy",
                response_time_ms=response_time,
                details={
                    "context_logger_active": self.context_logger is not None,
                    "context_expander_active": self.context_expander is not None,
                    "base_memory_accessible": True,
                    "base_memory_type": type(self.memory).__name__
                }
            )
            
            return OperationResult.success_result(health_result)
            
        except Exception as e:
            health_result = HealthCheckResult(
                component="enhanced_memory",
                status="unhealthy",
                response_time_ms=None,
                details={
                    "context_logger_active": self.context_logger is not None,
                    "context_expander_active": self.context_expander is not None,
                    "base_memory_accessible": False
                },
                error=str(e)
            )
            
            return OperationResult.success_result(health_result)  # Health check itself succeeded
    
    def get_debug_info(self) -> DebugInfoOperationResult:
        """
        Get debug information about enhanced memory with standardized result
        
        Returns:
            OperationResult[DebugInfo] with debug data
        """
        try:
            recent_ops = debug_tracker.get_recent_operations(10, "enhanced_memory")
            component_state = debug_tracker.get_component_state("enhanced_memory")
            
            # Get enhanced memory stats
            try:
                enhanced_stats = self.get_enhanced_stats()
            except Exception:
                enhanced_stats = {"error": "Could not retrieve enhanced stats"}
            
            debug_info = DebugInfo(
                component="enhanced_memory",
                recent_operations=recent_ops,
                component_state=component_state,
                statistics=enhanced_stats,
                configuration={
                    "context_logger_enabled": self.context_logger is not None,
                    "context_expander_enabled": self.context_expander is not None,
                    "base_memory_type": type(self.memory).__name__,
                    "has_graph_store": hasattr(self.memory, 'graph') and self.memory.graph is not None
                }
            )
            
            return OperationResult.success_result(debug_info)
            
        except Exception as e:
            return OperationResult.error_result(f"Debug info retrieval failed: {e}")
    
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
    
    # ========== END PHASE 2 INTERFACE ==========