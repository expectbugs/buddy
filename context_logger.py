"""
Full Context Logger for mem0 Integration
Stores complete conversation context alongside mem0 memories
NO silent failures - all errors are loud and proud
"""

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class ContextLogger:
    """
    Stores complete conversation context with unique lookup codes
    Works alongside mem0 without interfering with its operations
    """
    
    def __init__(self, log_dir: str = "/var/log/buddy/full_context"):
        self.log_dir = Path(log_dir)
        
        # Create directory - fail loudly if we can't
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"FAILED TO CREATE LOG DIRECTORY {log_dir}: {e}")
        
        self.session_id = str(uuid.uuid4())
        self.interaction_count = 0
        self.session_start_time = datetime.now(timezone.utc).isoformat()
        
        # Create/verify index file - fail loudly if problems
        self.index_file = self.log_dir / "lookup_index.json"
        try:
            if not self.index_file.exists():
                with open(self.index_file, 'w') as f:
                    json.dump({}, f)
            else:
                # Verify we can read it
                with open(self.index_file, 'r') as f:
                    json.load(f)
        except Exception as e:
            raise RuntimeError(f"FAILED TO INITIALIZE LOOKUP INDEX {self.index_file}: {e}")
        
        logger.info(f"ContextLogger initialized: session={self.session_id}, log_dir={self.log_dir}")
    
    def generate_lookup_code(self) -> str:
        """Generate unique lookup code for context retrieval"""
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        return f"CTX-{timestamp}-{uuid.uuid4().hex[:8]}"
    
    def log_interaction(self, 
                       user_input: str,
                       assistant_response: str,
                       reasoning: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       memory_operations: Optional[Dict[str, Any]] = None,
                       thinking_trace: Optional[str] = None,
                       user_id: Optional[str] = None,
                       response_time_ms: Optional[float] = None,
                       enhanced_format: bool = True,
                       lookup_code: Optional[str] = None) -> str:
        """
        Log complete interaction with lookup code
        Supports both basic format (backward compatible) and enhanced format (Phase 3)
        Returns: lookup code for this entry
        FAILS LOUDLY if logging fails
        """
        if not user_input or not assistant_response:
            raise ValueError("CONTEXT LOGGING REQUIRES NON-EMPTY user_input AND assistant_response")
        
        self.interaction_count += 1
        if lookup_code is None:
            lookup_code = self.generate_lookup_code()
        else:
            # LOUD VALIDATION: If provided lookup code, we MUST use it exactly
            if not lookup_code.startswith("CTX-"):
                raise ValueError(f"PROVIDED LOOKUP CODE IS INVALID: {lookup_code}")
        current_time = datetime.now(timezone.utc)
        current_timestamp = time.time()
        
        if enhanced_format:
            # Enhanced Phase 3 format
            entry = {
                "lookup_code": lookup_code,
                "timestamp": current_time.isoformat(),
                "unix_timestamp": current_timestamp,
                "format_version": "3.0",
                
                "session_info": {
                    "user_id": user_id or "unknown",
                    "conversation_turn": self.interaction_count,
                    "session_id": self.session_id,
                    "session_start": getattr(self, 'session_start_time', current_time.isoformat())
                },
                
                "interaction": {
                    "user_input": user_input,
                    "user_input_with_temporal": user_input,  # Will be enhanced by caller if temporal utils used
                    "assistant_response": assistant_response,
                    "thinking_trace": thinking_trace,
                    "response_time_ms": response_time_ms or 0,
                    "reasoning": reasoning  # Keep for backward compatibility
                },
                
                "memory_operations": memory_operations or {
                    "memories_searched": [],
                    "memories_found": [],
                    "memories_added": [],
                    "relationships_extracted": []
                },
                
                "context_metadata": {
                    "topics": [],
                    "entities": [],
                    "importance_score": 0.5,
                    "expansion_eligible": False,
                    "expansion_reasons": [],
                    "conversation_type": "general"
                },
                
                "system_metadata": {
                    "enhanced_memory_version": "3.0",
                    "phase": 3,
                    "model_used": "unknown",
                    "legacy_metadata": metadata or {}
                },
                
                "character_counts": {
                    "user_input": len(user_input),
                    "assistant_response": len(assistant_response),
                    "thinking_trace": len(thinking_trace) if thinking_trace else 0,
                    "reasoning": len(reasoning) if reasoning else 0
                }
            }
        else:
            # Basic format for backward compatibility
            entry = {
                "lookup_code": lookup_code,
                "timestamp": current_time.isoformat(),
                "unix_timestamp": current_timestamp,
                "session_id": self.session_id,
                "interaction_number": self.interaction_count,
                "user_input": user_input,
                "assistant_response": assistant_response,
                "reasoning": reasoning,
                "metadata": metadata or {},
                "format_version": "2.0",
                "character_counts": {
                    "user_input": len(user_input),
                    "assistant_response": len(assistant_response),
                    "reasoning": len(reasoning) if reasoning else 0
                }
            }
        
        # Write to daily log file - fail loudly if problems
        log_file = self.log_dir / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            raise RuntimeError(f"FAILED TO WRITE CONTEXT LOG to {log_file}: {e}")
        
        # Update lookup index - fail loudly if problems
        try:
            self._update_lookup_index(lookup_code, log_file, entry['timestamp'])
        except Exception as e:
            raise RuntimeError(f"FAILED TO UPDATE LOOKUP INDEX for {lookup_code}: {e}")
        
        logger.debug(f"Context logged: {lookup_code}")  # Changed to debug level
        return lookup_code
    
    def log_memory_operation(self, 
                           operation_type: str, 
                           details: Dict[str, Any],
                           user_id: Optional[str] = None) -> str:
        """
        Log individual memory operations for Phase 3 agent training data
        FAILS LOUDLY if logging fails
        
        Args:
            operation_type: Type of operation (search, add, update, delete, relationship_extract)
            details: Operation-specific details
            user_id: User identifier
            
        Returns:
            lookup code for this operation log
        """
        if not operation_type or not details:
            raise ValueError("MEMORY OPERATION LOGGING REQUIRES operation_type AND details")
        
        lookup_code = self.generate_lookup_code()
        current_time = datetime.now(timezone.utc)
        
        entry = {
            "lookup_code": lookup_code,
            "timestamp": current_time.isoformat(),
            "unix_timestamp": time.time(),
            "format_version": "3.0",
            "entry_type": "memory_operation",
            
            "session_info": {
                "user_id": user_id or "unknown",
                "session_id": self.session_id,
                "session_start": self.session_start_time
            },
            
            "operation": {
                "type": operation_type,
                "details": details,
                "timestamp": current_time.isoformat()
            },
            
            "system_metadata": {
                "enhanced_memory_version": "3.0",
                "phase": 3,
                "log_type": "memory_operation"
            }
        }
        
        # Write to daily log file - fail loudly if problems
        log_file = self.log_dir / f"memory_ops_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            raise RuntimeError(f"FAILED TO WRITE MEMORY OPERATION LOG to {log_file}: {e}")
        
        # Update lookup index - fail loudly if problems
        try:
            self._update_lookup_index(lookup_code, log_file, entry['timestamp'])
        except Exception as e:
            raise RuntimeError(f"FAILED TO UPDATE LOOKUP INDEX for memory operation {lookup_code}: {e}")
        
        logger.debug(f"Memory operation logged: {operation_type} - {lookup_code}")
        return lookup_code
    
    def update_memory_operations_in_log(self, 
                                      lookup_code: str,
                                      memory_operations: Dict[str, Any]) -> bool:
        """
        Update memory operations in an existing log entry
        Used to add memory operation data after the fact
        FAILS LOUDLY if update fails
        
        Args:
            lookup_code: The lookup code of the entry to update
            memory_operations: Memory operations data to add
            
        Returns:
            True if successful
        """
        if not lookup_code or not memory_operations:
            raise ValueError("UPDATE REQUIRES lookup_code AND memory_operations")
        
        # This is a simplified implementation - in a real system you might want
        # to implement more sophisticated log updating mechanisms
        logger.warning(f"Memory operations update requested for {lookup_code} - implementing as separate log entry")
        
        # For now, log as a separate memory operation entry
        self.log_memory_operation(
            operation_type="log_update",
            details={
                "original_lookup_code": lookup_code,
                "memory_operations": memory_operations,
                "update_reason": "post_interaction_memory_data"
            }
        )
        
        return True
    
    def _update_lookup_index(self, lookup_code: str, log_file: Path, timestamp: str):
        """Maintain fast lookup index - FAIL LOUDLY on errors"""
        try:
            # Read existing index
            with open(self.index_file, 'r') as f:
                index = json.load(f)
        except Exception as e:
            raise RuntimeError(f"FAILED TO READ LOOKUP INDEX {self.index_file}: {e}")
        
        # Add new entry
        index[lookup_code] = {
            "file": str(log_file.name),  # Just filename, not full path
            "timestamp": timestamp,
            "session_id": self.session_id
        }
        
        # Write updated index
        try:
            with open(self.index_file, 'w') as f:
                json.dump(index, f, indent=2)
        except Exception as e:
            raise RuntimeError(f"FAILED TO WRITE LOOKUP INDEX {self.index_file}: {e}")
    
    def retrieve_by_lookup_code(self, lookup_code: str) -> Dict[str, Any]:
        """
        Fast retrieval of full context by lookup code
        FAILS LOUDLY if lookup code not found or retrieval fails
        """
        if not lookup_code or not lookup_code.startswith("CTX-"):
            raise ValueError(f"INVALID LOOKUP CODE: {lookup_code}")
        
        try:
            with open(self.index_file, 'r') as f:
                index = json.load(f)
        except Exception as e:
            raise RuntimeError(f"FAILED TO READ LOOKUP INDEX: {e}")
        
        if lookup_code not in index:
            raise KeyError(f"LOOKUP CODE NOT FOUND: {lookup_code}")
        
        # Get log file
        log_filename = index[lookup_code]["file"]
        log_file = self.log_dir / log_filename
        
        if not log_file.exists():
            raise FileNotFoundError(f"LOG FILE NOT FOUND: {log_file}")
        
        # Search for entry in log file - handle both old and new formats
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("lookup_code") == lookup_code:
                            # Normalize format for backward compatibility
                            return self._normalize_log_entry(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            raise RuntimeError(f"FAILED TO READ LOG FILE {log_file}: {e}")
        
        raise KeyError(f"ENTRY NOT FOUND IN LOG FILE for {lookup_code}")
    
    def _normalize_log_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize log entries for backward compatibility
        Converts between different format versions
        """
        format_version = entry.get("format_version", "1.0")
        
        if format_version == "3.0":
            # Enhanced format - return as-is but ensure backward compatibility fields exist
            normalized = entry.copy()
            
            # Extract backward compatibility fields from enhanced structure
            if "interaction" in entry:
                normalized["user_input"] = entry["interaction"].get("user_input", "")
                normalized["assistant_response"] = entry["interaction"].get("assistant_response", "")
                normalized["reasoning"] = entry["interaction"].get("reasoning")
            
            if "session_info" in entry:
                normalized["session_id"] = entry["session_info"].get("session_id", "")
                normalized["interaction_number"] = entry["session_info"].get("conversation_turn", 0)
            
            # Ensure metadata field exists for backward compatibility
            if "metadata" not in normalized:
                normalized["metadata"] = entry.get("system_metadata", {}).get("legacy_metadata", {})
                
            return normalized
            
        elif format_version == "2.0":
            # Basic format - add format_version if missing and return
            if "format_version" not in entry:
                entry["format_version"] = "2.0"
            return entry
            
        else:
            # Very old format - add format_version and return
            entry["format_version"] = "1.0"
            return entry
    
    def search_logs(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search through full context logs
        FAILS LOUDLY on search errors
        """
        if not query or len(query.strip()) == 0:
            raise ValueError("SEARCH QUERY CANNOT BE EMPTY")
        
        if limit <= 0:
            raise ValueError(f"SEARCH LIMIT MUST BE POSITIVE: {limit}")
        
        results = []
        query_lower = query.lower()
        
        # Get all log files sorted by date (newest first)
        try:
            log_files = sorted(self.log_dir.glob("*.jsonl"), reverse=True)
        except Exception as e:
            raise RuntimeError(f"FAILED TO LIST LOG FILES in {self.log_dir}: {e}")
        
        if not log_files:
            raise FileNotFoundError(f"NO LOG FILES FOUND in {self.log_dir}")
        
        for log_file in log_files:
            if len(results) >= limit:
                break
                
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if len(results) >= limit:
                            break
                            
                        try:
                            entry = json.loads(line)
                            normalized_entry = self._normalize_log_entry(entry)
                            
                            # Search in user input, assistant response, and metadata
                            searchable_text = (
                                normalized_entry.get("user_input", "") + " " +
                                normalized_entry.get("assistant_response", "") + " " +
                                json.dumps(normalized_entry.get("metadata", {}))
                            ).lower()
                            
                            if query_lower in searchable_text:
                                results.append(normalized_entry)
                                
                        except json.JSONDecodeError:
                            # Skip malformed lines but don't fail the whole search
                            continue
                            
            except Exception as e:
                raise RuntimeError(f"FAILED TO SEARCH LOG FILE {log_file}: {e}")
        
        return results
    
    def get_session_logs(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all logs for a specific session
        FAILS LOUDLY on errors
        """
        target_session = session_id or self.session_id
        if not target_session:
            raise ValueError("SESSION ID CANNOT BE EMPTY")
        
        logs = []
        
        try:
            log_files = sorted(self.log_dir.glob("*.jsonl"))
        except Exception as e:
            raise RuntimeError(f"FAILED TO LIST LOG FILES: {e}")
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            normalized_entry = self._normalize_log_entry(entry)
                            if normalized_entry.get("session_id") == target_session:
                                logs.append(normalized_entry)
                        except json.JSONDecodeError:
                            # Skip malformed lines
                            continue
            except Exception as e:
                raise RuntimeError(f"FAILED TO READ LOG FILE {log_file}: {e}")
                
        return sorted(logs, key=lambda x: x.get("unix_timestamp", 0))
    
    def get_all_logs(self) -> List[Dict[str, Any]]:
        """
        Get ALL logs across all sessions - for timeline view
        FAILS LOUDLY on errors per Rule 3
        """
        logs = []
        
        try:
            log_files = sorted(self.log_dir.glob("*.jsonl"))
        except Exception as e:
            raise RuntimeError(f"FAILED TO LIST LOG FILES: {e}")
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            normalized_entry = self._normalize_log_entry(entry)
                            # Include all entries, not filtered by session
                            logs.append(normalized_entry)
                        except json.JSONDecodeError:
                            # Skip malformed lines
                            continue
            except Exception as e:
                raise RuntimeError(f"FAILED TO READ LOG FILE {log_file}: {e}")
                
        return sorted(logs, key=lambda x: x.get("unix_timestamp", 0))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics - enhanced for Phase 3
        FAILS LOUDLY if can't read logs
        """
        total_entries = 0
        total_characters = 0
        sessions = set()
        format_versions = {}
        memory_operations_count = 0
        enhanced_entries = 0
        
        try:
            log_files = list(self.log_dir.glob("*.jsonl"))
            memory_log_files = list(self.log_dir.glob("memory_ops_*.jsonl"))
        except Exception as e:
            raise RuntimeError(f"FAILED TO LIST LOG FILES for stats: {e}")
        
        # Process regular log files
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            normalized_entry = self._normalize_log_entry(entry)
                            total_entries += 1
                            sessions.add(normalized_entry.get("session_id"))
                            
                            # Track format versions
                            version = normalized_entry.get("format_version", "1.0")
                            format_versions[version] = format_versions.get(version, 0) + 1
                            
                            if version == "3.0":
                                enhanced_entries += 1
                            
                            counts = normalized_entry.get("character_counts", {})
                            total_characters += sum(counts.values())
                            
                        except json.JSONDecodeError:
                            # Skip malformed lines but don't fail stats
                            continue
            except Exception as e:
                raise RuntimeError(f"FAILED TO READ LOG FILE {log_file} for stats: {e}")
        
        # Process memory operation log files
        for log_file in memory_log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            memory_operations_count += 1
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.warning(f"Could not read memory ops log {log_file}: {e}")
        
        return {
            "total_entries": total_entries,
            "total_sessions": len(sessions),
            "total_characters": total_characters,
            "current_session_id": self.session_id,
            "log_files": len(log_files),
            "memory_operation_files": len(memory_log_files),
            "memory_operations_count": memory_operations_count,
            "log_directory": str(self.log_dir),
            "format_versions": format_versions,
            "enhanced_entries": enhanced_entries,
            "enhanced_format_percentage": (enhanced_entries / total_entries * 100) if total_entries > 0 else 0,
            "phase": 3,
            "features": [
                "enhanced_logging",
                "memory_operation_tracking", 
                "backward_compatibility",
                "format_normalization"
            ]
        }