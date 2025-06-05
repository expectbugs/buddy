"""
Context Bridge - Connects mem0 memories with full context logs
Enables retrieval of full conversation context for any memory
NO silent failures - all errors are loud and proud
"""

import json
import logging
from typing import Dict, Any, Optional, List
from context_logger import ContextLogger

logger = logging.getLogger(__name__)

class ContextBridge:
    """
    Bridges mem0 memories with full context logs
    Enables context expansion and retrieval
    """
    
    def __init__(self, context_logger: Optional[ContextLogger] = None):
        self.context_logger = context_logger or ContextLogger()
        logger.info("ContextBridge initialized")
    
    def get_full_context(self, memory_id: str, memory_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve full context for a memory using its metadata
        FAILS LOUDLY if lookup_code missing or context not found
        """
        if not memory_id:
            raise ValueError("MEMORY ID CANNOT BE EMPTY")
        
        if not memory_metadata:
            raise ValueError(f"MEMORY METADATA MISSING for memory {memory_id}")
        
        lookup_code = memory_metadata.get('lookup_code')
        if not lookup_code:
            raise KeyError(f"NO LOOKUP CODE in metadata for memory {memory_id}")
        
        try:
            context = self.context_logger.retrieve_by_lookup_code(lookup_code)
        except Exception as e:
            raise RuntimeError(f"FAILED TO RETRIEVE CONTEXT for memory {memory_id}, lookup_code {lookup_code}: {e}")
        
        return {
            "memory_id": memory_id,
            "lookup_code": lookup_code,
            "context": context,
            "metadata": memory_metadata
        }
    
    def get_context_window(self, 
                          memory_id: str, 
                          memory_metadata: Dict[str, Any],
                          before_lines: int = 2,
                          after_lines: int = 2) -> Dict[str, Any]:
        """
        Get context window around a memory (lines before/after)
        FAILS LOUDLY if context not available or invalid parameters
        """
        if before_lines < 0 or after_lines < 0:
            raise ValueError(f"CONTEXT WINDOW PARAMETERS MUST BE NON-NEGATIVE: before={before_lines}, after={after_lines}")
        
        # Get the base context first
        base_context = self.get_full_context(memory_id, memory_metadata)
        
        # Get session logs to find surrounding entries
        session_id = base_context["context"].get("session_id")
        if not session_id:
            raise ValueError(f"NO SESSION ID in context for memory {memory_id}")
        
        try:
            session_logs = self.context_logger.get_session_logs(session_id)
        except Exception as e:
            raise RuntimeError(f"FAILED TO GET SESSION LOGS for memory {memory_id}: {e}")
        
        # Find the target entry in session logs
        target_interaction = base_context["context"]["interaction_number"]
        target_entry = None
        target_index = -1
        
        for i, entry in enumerate(session_logs):
            if entry.get("interaction_number") == target_interaction:
                target_entry = entry
                target_index = i
                break
        
        if target_entry is None:
            raise KeyError(f"TARGET INTERACTION {target_interaction} NOT FOUND in session logs for memory {memory_id}")
        
        # Get surrounding entries
        start_index = max(0, target_index - before_lines)
        end_index = min(len(session_logs), target_index + after_lines + 1)
        
        context_window = session_logs[start_index:end_index]
        
        return {
            "memory_id": memory_id,
            "lookup_code": base_context["lookup_code"],
            "target_entry": target_entry,
            "context_window": context_window,
            "window_info": {
                "before_lines": before_lines,
                "after_lines": after_lines,
                "actual_before": target_index - start_index,
                "actual_after": end_index - target_index - 1,
                "total_entries": len(context_window)
            },
            "metadata": memory_metadata
        }
    
    def should_expand_context(self, memory_score: float, expansion_threshold: float = 0.85) -> bool:
        """
        Determine if a memory should have its context expanded
        Based on relevance score threshold
        """
        if expansion_threshold < 0 or expansion_threshold > 1:
            raise ValueError(f"EXPANSION THRESHOLD MUST BE BETWEEN 0 AND 1: {expansion_threshold}")
        
        return memory_score >= expansion_threshold
    
    def expand_memory_result(self, 
                           memory_result: Dict[str, Any], 
                           include_context_window: bool = False,
                           window_size: int = 2) -> Dict[str, Any]:
        """
        Expand a memory search result with full context
        FAILS LOUDLY if expansion fails
        """
        if not memory_result:
            raise ValueError("MEMORY RESULT CANNOT BE EMPTY")
        
        memory_id = memory_result.get('id')
        if not memory_id:
            raise KeyError("MEMORY RESULT MISSING 'id' field")
        
        metadata = memory_result.get('metadata', {})
        if not metadata:
            raise KeyError(f"MEMORY RESULT MISSING metadata for memory {memory_id}")
        
        expanded_result = memory_result.copy()
        
        try:
            if include_context_window:
                context_data = self.get_context_window(
                    memory_id, 
                    metadata, 
                    before_lines=window_size, 
                    after_lines=window_size
                )
                expanded_result['expanded_context'] = context_data
            else:
                context_data = self.get_full_context(memory_id, metadata)
                expanded_result['expanded_context'] = context_data
        except Exception as e:
            raise RuntimeError(f"FAILED TO EXPAND MEMORY {memory_id}: {e}")
        
        return expanded_result
    
    def search_context_logs(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search directly in context logs (bypass mem0)
        FAILS LOUDLY on search errors
        """
        try:
            return self.context_logger.search_logs(query, limit)
        except Exception as e:
            raise RuntimeError(f"CONTEXT LOG SEARCH FAILED for query '{query}': {e}")
    
    def get_bridge_stats(self) -> Dict[str, Any]:
        """
        Get statistics about context logging and bridging
        FAILS LOUDLY if stats retrieval fails
        """
        try:
            logger_stats = self.context_logger.get_stats()
        except Exception as e:
            raise RuntimeError(f"FAILED TO GET CONTEXT LOGGER STATS: {e}")
        
        return {
            "context_logger_stats": logger_stats,
            "bridge_info": {
                "bridge_active": True,
                "logger_session_id": self.context_logger.session_id
            }
        }
    
    def validate_memory_metadata(self, memory_metadata: Dict[str, Any]) -> bool:
        """
        Validate that memory metadata contains required fields for context retrieval
        Returns True if valid, raises exception if invalid
        """
        if not memory_metadata:
            raise ValueError("MEMORY METADATA IS EMPTY")
        
        required_fields = ['lookup_code', 'timestamp_utc', 'unix_timestamp']
        missing_fields = []
        
        for field in required_fields:
            if field not in memory_metadata:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"MEMORY METADATA MISSING REQUIRED FIELDS: {missing_fields}")
        
        # Validate lookup_code format
        lookup_code = memory_metadata['lookup_code']
        if not lookup_code or not lookup_code.startswith('CTX-'):
            raise ValueError(f"INVALID LOOKUP CODE FORMAT: {lookup_code}")
        
        return True