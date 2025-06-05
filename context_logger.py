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
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Log complete interaction with lookup code
        Returns: lookup code for this entry
        FAILS LOUDLY if logging fails
        """
        if not user_input or not assistant_response:
            raise ValueError("CONTEXT LOGGING REQUIRES NON-EMPTY user_input AND assistant_response")
        
        self.interaction_count += 1
        lookup_code = self.generate_lookup_code()
        
        entry = {
            "lookup_code": lookup_code,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "unix_timestamp": time.time(),
            "session_id": self.session_id,
            "interaction_number": self.interaction_count,
            "user_input": user_input,
            "assistant_response": assistant_response,
            "reasoning": reasoning,
            "metadata": metadata or {},
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
        
        # Search for entry in log file
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("lookup_code") == lookup_code:
                            return entry
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            raise RuntimeError(f"FAILED TO READ LOG FILE {log_file}: {e}")
        
        raise KeyError(f"ENTRY NOT FOUND IN LOG FILE for {lookup_code}")
    
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
                            # Search in user input, assistant response, and metadata
                            searchable_text = (
                                entry.get("user_input", "") + " " +
                                entry.get("assistant_response", "") + " " +
                                json.dumps(entry.get("metadata", {}))
                            ).lower()
                            
                            if query_lower in searchable_text:
                                results.append(entry)
                                
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
                            if entry.get("session_id") == target_session:
                                logs.append(entry)
                        except json.JSONDecodeError:
                            # Skip malformed lines
                            continue
            except Exception as e:
                raise RuntimeError(f"FAILED TO READ LOG FILE {log_file}: {e}")
                
        return sorted(logs, key=lambda x: x.get("unix_timestamp", 0))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get logging statistics
        FAILS LOUDLY if can't read logs
        """
        total_entries = 0
        total_characters = 0
        sessions = set()
        
        try:
            log_files = list(self.log_dir.glob("*.jsonl"))
        except Exception as e:
            raise RuntimeError(f"FAILED TO LIST LOG FILES for stats: {e}")
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            total_entries += 1
                            sessions.add(entry.get("session_id"))
                            
                            counts = entry.get("character_counts", {})
                            total_characters += sum(counts.values())
                            
                        except json.JSONDecodeError:
                            # Skip malformed lines but don't fail stats
                            continue
            except Exception as e:
                raise RuntimeError(f"FAILED TO READ LOG FILE {log_file} for stats: {e}")
        
        return {
            "total_entries": total_entries,
            "total_sessions": len(sessions),
            "total_characters": total_characters,
            "current_session_id": self.session_id,
            "log_files": len(log_files),
            "log_directory": str(self.log_dir)
        }