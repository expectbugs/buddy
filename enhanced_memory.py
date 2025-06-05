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
    
    def __init__(self, mem0_memory: Memory, context_logger: Optional[ContextLogger] = None):
        """
        Initialize enhanced memory wrapper
        
        Args:
            mem0_memory: The mem0 Memory instance to wrap
            context_logger: Optional context logger (creates new one if None)
        """
        if not isinstance(mem0_memory, Memory):
            raise TypeError(f"ENHANCED MEMORY REQUIRES mem0.Memory INSTANCE, got {type(mem0_memory)}")
        
        # Store the original mem0 instance
        self.memory = mem0_memory
        
        # Initialize context system
        try:
            self.context_logger = context_logger or ContextLogger()
            self.context_bridge = ContextBridge(self.context_logger)
            logger.info("EnhancedMemory initialized successfully")
        except Exception as e:
            raise RuntimeError(f"FAILED TO INITIALIZE ENHANCED MEMORY CONTEXT SYSTEM: {e}")
    
    def add(self, messages: Union[str, List[Dict[str, str]]], user_id: str, **kwargs) -> Any:
        """
        Enhanced add method that automatically includes context metadata
        
        This method:
        1. Calls the original mem0 add() method with all original functionality
        2. Adds enhanced metadata with context linking
        3. Maintains full backward compatibility
        4. Fails loudly if anything goes wrong
        
        Args:
            messages: Same as mem0.Memory.add() - can be string or list of message dicts
            user_id: Same as mem0.Memory.add() - user identifier
            **kwargs: All other mem0.Memory.add() arguments pass through unchanged
            
        Returns:
            Same return value as mem0.Memory.add()
        """
        # Extract user input and assistant response for context logging
        user_input = ""
        assistant_response = ""
        
        if isinstance(messages, str):
            user_input = messages
        elif isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get('role') == 'user':
                        user_input = msg.get('content', '')
                    elif msg.get('role') == 'assistant':
                        assistant_response = msg.get('content', '')
        
        # Generate lookup code for context linking (but only if we have both user input and response)
        lookup_code = None
        if user_input and assistant_response:  # Only log if we have both parts of a conversation
            try:
                lookup_code = self.context_logger.log_interaction(
                    user_input=user_input,
                    assistant_response=assistant_response,
                    reasoning=None,
                    metadata={
                        "user_id": user_id,
                        "enhanced_memory": True,
                        "phase": "2",
                        "datetime_info": get_current_datetime_info()
                    }
                )
                logger.debug(f"Context logged for enhanced memory: {lookup_code}")
            except Exception as e:
                # Fail loudly per Rule 3
                raise RuntimeError(f"CONTEXT LOGGING FAILED in EnhancedMemory.add(): {e}")
        else:
            logger.debug("Skipping context logging - not a full conversation (missing user input or assistant response)")
        
        # Enhance metadata with context linking and timestamps
        enhanced_metadata = kwargs.get('metadata', {}).copy() if 'metadata' in kwargs else {}
        
        # Add our enhancements
        if lookup_code:
            enhanced_metadata['lookup_code'] = lookup_code
            
        enhanced_metadata.update({
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'unix_timestamp': time.time(),
            'context_type': 'enhanced',
            'enhanced_memory_version': '2.0'
        })
        
        # Update kwargs with enhanced metadata
        kwargs['metadata'] = enhanced_metadata
        
        # Call original mem0 add method with enhanced metadata
        try:
            result = self.memory.add(messages, user_id=user_id, **kwargs)
            logger.debug(f"Enhanced memory added successfully with metadata: {len(enhanced_metadata)} fields")
            return result
        except Exception as e:
            # Fail loudly per Rule 3 - don't hide mem0 errors
            raise RuntimeError(f"MEM0 ADD FAILED in EnhancedMemory: {e}")
    
    # Delegate all other methods to the original mem0 Memory instance
    # This ensures 100% backward compatibility
    
    def search(self, *args, **kwargs):
        """Delegate search to original mem0 instance"""
        try:
            return self.memory.search(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"MEM0 SEARCH FAILED in EnhancedMemory: {e}")
    
    def get_all(self, *args, **kwargs):
        """Delegate get_all to original mem0 instance"""
        try:
            return self.memory.get_all(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"MEM0 GET_ALL FAILED in EnhancedMemory: {e}")
    
    def update(self, *args, **kwargs):
        """Delegate update to original mem0 instance"""
        try:
            return self.memory.update(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"MEM0 UPDATE FAILED in EnhancedMemory: {e}")
    
    def delete(self, *args, **kwargs):
        """Delegate delete to original mem0 instance"""
        try:
            return self.memory.delete(*args, **kwargs)
        except Exception as e:
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
        New method: Get statistics about enhanced memory usage
        
        Returns:
            Statistics about context logging and enhanced memories
        """
        try:
            bridge_stats = self.context_bridge.get_bridge_stats()
            return {
                "enhanced_memory_active": True,
                "phase": "2",
                "context_bridge_stats": bridge_stats,
                "features": [
                    "automatic_context_logging",
                    "enhanced_metadata", 
                    "lookup_code_linking",
                    "temporal_awareness"
                ]
            }
        except Exception as e:
            raise RuntimeError(f"FAILED TO GET ENHANCED MEMORY STATS: {e}")