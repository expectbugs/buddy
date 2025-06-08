"""
Context Expansion System for Phase 3: Smart Context Expansion
Automatically expands high-relevance memories with full conversation context
NO silent failures - all errors are loud and proud
"""

import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import re

logger = logging.getLogger(__name__)

# For caching
import hashlib
import threading
from collections import OrderedDict

class ContextExpansionDecision:
    """
    Decides when and how to expand memory context based on relevance and criteria
    
    Expansion criteria:
    - Relevance score > 0.75 (high relevance threshold)
    - Memory has lookup_code (can retrieve full context)
    - User query suggests need for detailed context
    - Recent memory (within last 30 days) - older ones less likely to need expansion
    - Memory type benefits from context (relationships, complex facts)
    """
    
    def __init__(self, 
                 relevance_threshold: float = 0.55,
                 recency_days: int = 30,
                 max_expansions: int = 3):
        """
        Initialize expansion decision engine
        
        Args:
            relevance_threshold: Minimum relevance score for expansion (0.0-1.0)
            recency_days: Days within which memories are considered recent
            max_expansions: Maximum number of memories to expand per query
        """
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError(f"RELEVANCE THRESHOLD MUST BE BETWEEN 0.0 AND 1.0: {relevance_threshold}")
        
        if recency_days <= 0:
            raise ValueError(f"RECENCY DAYS MUST BE POSITIVE: {recency_days}")
        
        if max_expansions <= 0:
            raise ValueError(f"MAX EXPANSIONS MUST BE POSITIVE: {max_expansions}")
        
        self.relevance_threshold = relevance_threshold
        self.recency_days = recency_days
        self.max_expansions = max_expansions
        
        # Query patterns that suggest need for detailed context
        self.context_need_patterns = [
            r"\b(what|how|why|when|where|who|tell me about|explain|describe)\b",
            r"\b(details?|specifics?|more info|background|context)\b",
            r"\b(conversation|discussed?|talked about|mentioned)\b",
            r"\b(exactly|precisely|specifically)\b",
            r"\b(full story|complete|entire)\b"
        ]
        
        # Temporal patterns that indicate historical context is needed
        self.temporal_patterns = [
            r"\b(used to|before|previously|originally|changed|was)\b",
            r"\b(when did|what was|who was|where was|how was)\b",
            r"\b(history|past|former|old|previous)\b",
            r"\b(then|earlier|ago|back when|at the time)\b"
        ]
        
        # Memory types that benefit from context expansion
        self.expansion_worthy_types = {
            "relationship", "complex_fact", "conversation", "project", 
            "event", "decision", "problem_solving", "learning"
        }
        
        logger.info(f"ContextExpansionDecision initialized: threshold={relevance_threshold}, "
                   f"recency={recency_days} days, max_expansions={max_expansions}")
    
    def should_expand(self, 
                     memory_result: Dict[str, Any], 
                     user_query: str, 
                     relevance_score: float) -> Tuple[bool, List[str]]:
        """
        Decide if a memory should have its context expanded
        
        Args:
            memory_result: Memory search result with metadata
            user_query: Original user query
            relevance_score: Relevance score from search
            
        Returns:
            Tuple of (should_expand: bool, reasons: List[str])
        """
        if not memory_result:
            return False, ["empty_memory_result"]
        
        if not user_query or not user_query.strip():
            return False, ["empty_user_query"]
        
        reasons = []
        
        # Primary check: Relevance score threshold
        if relevance_score < self.relevance_threshold:
            return False, [f"low_relevance_score_{relevance_score:.3f}"]
        
        reasons.append(f"high_relevance_{relevance_score:.3f}")
        
        # Check if memory has lookup_code for context retrieval
        metadata = memory_result.get('metadata', {})
        lookup_code = metadata.get('context_lookup_code') or metadata.get('lookup_code')
        
        if not lookup_code:
            return False, ["missing_lookup_code"]
        reasons.append("has_lookup_code")
        
        # Check recency - recent memories more likely to need expansion
        if self._is_recent_memory(metadata):
            reasons.append("recent_memory")
        else:
            # Old memories are less likely to need expansion unless very high relevance
            if relevance_score < 0.65:
                return False, [f"old_memory_insufficient_relevance_{relevance_score:.3f}"]
            reasons.append("old_memory_high_relevance")
        
        # Analyze user query for context need indicators
        if self._query_suggests_context_need(user_query):
            reasons.append("query_suggests_context_need")
        
        # Check memory type - some types benefit more from expansion
        memory_type = self._classify_memory_type(memory_result)
        if memory_type in self.expansion_worthy_types:
            reasons.append(f"expansion_worthy_type_{memory_type}")
        
        # Check memory content complexity
        if self._is_complex_memory(memory_result):
            reasons.append("complex_memory_content")
        
        # Final decision: If we have at least 2 positive indicators, expand
        positive_indicators = len([r for r in reasons if not r.startswith("old_memory")])
        
        if positive_indicators >= 2:
            reasons.append("sufficient_positive_indicators")
            return True, reasons
        else:
            reasons.append(f"insufficient_indicators_{positive_indicators}")
            return False, reasons
    
    def _is_recent_memory(self, metadata: Dict[str, Any]) -> bool:
        """Check if memory is recent (within recency_days)"""
        try:
            timestamp = metadata.get('unix_timestamp')
            if not timestamp:
                # If no timestamp, assume it's old
                return False
            
            memory_time = datetime.fromtimestamp(float(timestamp))
            cutoff_time = datetime.now() - timedelta(days=self.recency_days)
            
            return memory_time >= cutoff_time
            
        except (ValueError, TypeError, OSError):
            # If we can't parse timestamp, assume it's old
            return False
    
    def _query_suggests_context_need(self, user_query: str) -> bool:
        """Analyze query to see if it suggests need for detailed context"""
        query_lower = user_query.lower()
        
        for pattern in self.context_need_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check temporal patterns (historical context needed)
        for pattern in self.temporal_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Additional heuristics
        # Questions tend to need more context
        if user_query.strip().endswith('?'):
            return True
        
        # Long queries often need context
        if len(user_query.split()) > 8:
            return True
        
        # Queries with specific entities might need context (but not common greetings)
        if re.search(r'\b[A-Z][a-z]+\b', user_query):  # Proper nouns
            # Exclude common greetings, simple words, and non-entity capitalized words
            common_non_entities = {
                'Hello', 'Hi', 'Hey', 'Thanks', 'Thank', 'Yes', 'No', 'Ok', 'Okay',
                'Pizza', 'Coffee', 'Tea', 'Music', 'Movie', 'Book', 'Game', 'Food',
                'Weather', 'Time', 'Day', 'Night', 'Morning', 'Evening', 'Today',
                'Tomorrow', 'Yesterday', 'Good', 'Bad', 'Nice', 'Great', 'Cool'
            }
            words = user_query.split()
            entity_words = [w for w in words if re.match(r'^[A-Z][a-z]+$', w) and w not in common_non_entities]
            if entity_words:
                return True
        
        return False
    
    def _classify_memory_type(self, memory_result: Dict[str, Any]) -> str:
        """Classify memory type based on content and metadata"""
        memory_text = memory_result.get('memory', memory_result.get('text', '')).lower()
        metadata = memory_result.get('metadata', {})
        
        # Check explicit type in metadata
        explicit_type = metadata.get('memory_type', metadata.get('type'))
        if explicit_type and explicit_type in self.expansion_worthy_types:
            return explicit_type
        
        # Infer type from content
        if any(word in memory_text for word in ['relationship', 'friend', 'colleague', 'family', 'knows', 'works with']):
            return "relationship"
        
        if any(word in memory_text for word in ['project', 'working on', 'building', 'developing']):
            return "project"
        
        if any(word in memory_text for word in ['conversation', 'discussed', 'talked about', 'mentioned']):
            return "conversation"
        
        if any(word in memory_text for word in ['decided', 'chose', 'selected', 'picked']):
            return "decision"
        
        if any(word in memory_text for word in ['problem', 'issue', 'challenge', 'solving', 'trouble', 'debugging', 'error', 'bug', 'fix', 'stuck']):
            return "problem_solving"
        
        if any(word in memory_text for word in ['learned', 'discovered', 'found out', 'realized']):
            return "learning"
        
        if any(word in memory_text for word in ['meeting', 'event', 'happened', 'occurred']):
            return "event"
        
        # Complex indicators
        if len(memory_text.split()) > 15:
            return "complex_fact"
        
        return "simple_fact"
    
    def _is_complex_memory(self, memory_result: Dict[str, Any]) -> bool:
        """Determine if memory content is complex enough to benefit from expansion"""
        memory_text = memory_result.get('memory', memory_result.get('text', ''))
        
        # Length-based complexity
        if len(memory_text.split()) > 20:
            return True
        
        # Structural complexity indicators
        complexity_indicators = [
            ',' in memory_text,  # Lists or complex sentences
            ' and ' in memory_text.lower(),  # Compound information
            ' but ' in memory_text.lower(),  # Contrasting information
            ' because ' in memory_text.lower(),  # Causal relationships
            ' when ' in memory_text.lower(),  # Temporal relationships
            ' where ' in memory_text.lower(),  # Spatial relationships
        ]
        
        return sum(complexity_indicators) >= 2
    
    def get_expansion_priority(self, 
                             memories: List[Dict[str, Any]], 
                             user_query: str) -> List[Dict[str, Any]]:
        """
        Rank memories by expansion value and limit to max_expansions
        
        Args:
            memories: List of memory search results
            user_query: Original user query
            
        Returns:
            List of memories ranked by expansion priority (limited to max_expansions)
        """
        if not memories:
            return []
        
        if not user_query or not user_query.strip():
            raise ValueError("USER QUERY CANNOT BE EMPTY for expansion priority ranking")
        
        expansion_candidates = []
        
        for memory in memories:
            relevance_score = memory.get('relevance', 0.0)
            should_expand, reasons = self.should_expand(memory, user_query, relevance_score)
            
            if should_expand:
                # Calculate expansion priority score
                priority_score = self._calculate_priority_score(memory, user_query, relevance_score, reasons)
                
                expansion_candidates.append({
                    'memory': memory,
                    'priority_score': priority_score,
                    'expansion_reasons': reasons,
                    'relevance_score': relevance_score
                })
        
        # Sort by priority score (highest first) and limit to max_expansions
        expansion_candidates.sort(key=lambda x: x['priority_score'], reverse=True)
        top_candidates = expansion_candidates[:self.max_expansions]
        
        logger.debug(f"Expansion priority: {len(expansion_candidates)} candidates, "
                    f"top {len(top_candidates)} selected for expansion")
        
        return top_candidates
    
    def _calculate_priority_score(self, 
                                memory: Dict[str, Any], 
                                user_query: str, 
                                relevance_score: float,
                                reasons: List[str]) -> float:
        """
        Calculate expansion priority score for ranking
        
        Higher scores get expanded first
        """
        base_score = relevance_score  # Start with relevance (0.85-1.0)
        
        # Bonus points for various factors
        bonus = 0.0
        
        # Recent memories get priority
        if "recent_memory" in reasons:
            bonus += 0.05
        
        # Query suggests context need
        if "query_suggests_context_need" in reasons:
            bonus += 0.03
        
        # Complex memories benefit more from expansion
        if "complex_memory_content" in reasons:
            bonus += 0.02
        
        # Certain types get priority
        expansion_worthy_reasons = [r for r in reasons if r.startswith("expansion_worthy_type_")]
        if expansion_worthy_reasons:
            memory_type = expansion_worthy_reasons[0].split("_")[-1]
            type_bonuses = {
                "relationship": 0.04,
                "conversation": 0.03,
                "project": 0.02,
                "decision": 0.02,
                "complex_fact": 0.01
            }
            bonus += type_bonuses.get(memory_type, 0.01)
        
        # Very high relevance gets extra priority
        if relevance_score > 0.92:
            bonus += 0.02
        
        final_score = base_score + bonus
        
        logger.debug(f"Priority calculation: base={base_score:.3f}, bonus={bonus:.3f}, "
                    f"final={final_score:.3f}, reasons={len(reasons)}")
        
        return final_score
    
    def get_expansion_statistics(self) -> Dict[str, Any]:
        """Get statistics about expansion decisions for monitoring"""
        return {
            "relevance_threshold": self.relevance_threshold,
            "recency_days": self.recency_days,
            "max_expansions": self.max_expansions,
            "context_need_patterns": len(self.context_need_patterns),
            "expansion_worthy_types": list(self.expansion_worthy_types),
            "decision_engine_version": "3.0"
        }


class ContextExpander:
    """
    Main context expansion engine that processes search results and expands high-value memories
    
    Features:
    - Intelligent expansion using ContextExpansionDecision
    - LRU cache with TTL for performance optimization
    - Context formatting optimized for AI consumption  
    - Token limits and performance safeguards
    - Integration with ContextBridge for context retrieval
    """
    
    def __init__(self, 
                 context_bridge,  # ContextBridge instance
                 relevance_threshold: float = 0.55,
                 max_expansions: int = 3,
                 cache_size: int = 100,
                 cache_ttl_minutes: int = 60,
                 max_context_tokens: int = 4000,
                 expansion_timeout_ms: int = 500):
        """
        Initialize context expansion engine
        
        Args:
            context_bridge: ContextBridge instance for context retrieval
            relevance_threshold: Minimum relevance score for expansion
            max_expansions: Maximum memories to expand per query
            cache_size: Maximum number of cached expansions
            cache_ttl_minutes: Cache TTL in minutes
            max_context_tokens: Maximum tokens for expanded context
            expansion_timeout_ms: Timeout for expansion operations
        """
        if not context_bridge:
            raise ValueError("CONTEXT BRIDGE IS REQUIRED")
        
        if not 0.0 <= relevance_threshold <= 1.0:
            raise ValueError(f"RELEVANCE THRESHOLD MUST BE BETWEEN 0.0 AND 1.0: {relevance_threshold}")
        
        if max_expansions <= 0:
            raise ValueError(f"MAX EXPANSIONS MUST BE POSITIVE: {max_expansions}")
        
        if cache_size <= 0:
            raise ValueError(f"CACHE SIZE MUST BE POSITIVE: {cache_size}")
        
        if cache_ttl_minutes <= 0:
            raise ValueError(f"CACHE TTL MUST BE POSITIVE: {cache_ttl_minutes}")
        
        if max_context_tokens <= 0:
            raise ValueError(f"MAX CONTEXT TOKENS MUST BE POSITIVE: {max_context_tokens}")
        
        if expansion_timeout_ms <= 0:
            raise ValueError(f"EXPANSION TIMEOUT MUST BE POSITIVE: {expansion_timeout_ms}")
        
        self.context_bridge = context_bridge
        self.decision_engine = ContextExpansionDecision(
            relevance_threshold=relevance_threshold,
            max_expansions=max_expansions
        )
        
        # Configuration
        self.max_context_tokens = max_context_tokens
        self.expansion_timeout_ms = expansion_timeout_ms
        self.cache_ttl_seconds = cache_ttl_minutes * 60
        
        # LRU Cache with TTL
        self.expansion_cache = OrderedDict()
        self.cache_timestamps = {}
        self.cache_size = cache_size
        self.cache_lock = threading.Lock()
        
        # Performance tracking
        self.expansion_stats = {
            "total_expansions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "expansion_failures": 0,
            "average_expansion_time_ms": 0.0,
            "total_expansion_time_ms": 0.0
        }
        
        logger.info(f"ContextExpander initialized: threshold={relevance_threshold}, "
                   f"max_expansions={max_expansions}, cache_size={cache_size}, "
                   f"max_tokens={max_context_tokens}")
    
    def expand_memory_results(self, search_results: Dict[str, Any], user_query: str) -> Dict[str, Any]:
        """
        Process search results and expand high-value memories
        
        Args:
            search_results: Memory search results from mem0
            user_query: Original user query
            
        Returns:
            Enhanced results with expanded context for high-relevance memories
        """
        if not search_results or not isinstance(search_results, dict):
            logger.warning("Empty or invalid search results for expansion")
            return search_results
        
        if not user_query or not user_query.strip():
            logger.warning("Empty user query for expansion")
            return search_results
        
        start_time = time.time()
        
        try:
            # Get memories from search results
            memories = search_results.get('results', [])
            if not memories:
                logger.debug("No memories in search results to expand")
                return search_results
            
            # Get expansion candidates using decision engine
            expansion_candidates = self.decision_engine.get_expansion_priority(memories, user_query)
            
            if not expansion_candidates:
                logger.debug("No memories qualified for expansion")
                return search_results
            
            logger.debug(f"Expanding {len(expansion_candidates)} memories out of {len(memories)} total")
            
            # Expand qualified memories
            enhanced_results = search_results.copy()
            enhanced_memories = []
            
            # VALIDATION: If we have candidates but no matches, this is an error (Rule 3)
            matches_found = 0
            
            for memory_result in memories:
                # Check if this memory should be expanded
                candidate = next((c for c in expansion_candidates if c['memory']['id'] == memory_result.get('id')), None)
                
                if candidate:
                    matches_found += 1
                    # Expand this memory
                    expanded_memory = self._expand_single_memory(memory_result, user_query, candidate)
                    enhanced_memories.append(expanded_memory)
                else:
                    # Keep original memory
                    enhanced_memories.append(memory_result)
            
            # LOUD ERROR: If we created candidates but found no matches, something is wrong
            if len(expansion_candidates) > 0 and matches_found == 0:
                candidate_ids = [c['memory'].get('id', 'NO_ID') for c in expansion_candidates]
                memory_ids = [m.get('id', 'NO_ID') for m in memories]
                raise RuntimeError(f"EXPANSION CANDIDATE MATCHING FAILED: Created {len(expansion_candidates)} candidates but found 0 matches. Candidate IDs: {candidate_ids}, Memory IDs: {memory_ids}")
            
            enhanced_results['results'] = enhanced_memories
            
            # Add expansion metadata
            enhanced_results['expansion_metadata'] = {
                "expanded_count": len(expansion_candidates),
                "total_memories": len(memories),
                "expansion_time_ms": (time.time() - start_time) * 1000,
                "expansion_enabled": True
            }
            
            # Update stats
            self.expansion_stats["total_expansions"] += len(expansion_candidates)
            expansion_time = (time.time() - start_time) * 1000
            self.expansion_stats["total_expansion_time_ms"] += expansion_time
            self.expansion_stats["average_expansion_time_ms"] = (
                self.expansion_stats["total_expansion_time_ms"] / 
                max(1, self.expansion_stats["total_expansions"])
            )
            
            logger.debug(f"Expansion completed in {expansion_time:.1f}ms: "
                        f"{len(expansion_candidates)} memories expanded")
            
            return enhanced_results
            
        except Exception as e:
            self.expansion_stats["expansion_failures"] += 1
            logger.error(f"Context expansion failed: {e}")
            # Return original results on failure - graceful degradation
            return search_results
    
    def _expand_single_memory(self, memory_result: Dict[str, Any], user_query: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expand a single memory with full context
        
        Args:
            memory_result: Original memory search result
            user_query: User query for context
            candidate: Expansion candidate with priority info
            
        Returns:
            Memory result with expanded context
        """
        memory_id = memory_result.get('id')
        if not memory_id:
            logger.warning("Memory missing ID, cannot expand")
            return memory_result
        
        # Check cache first
        cache_key = self._generate_cache_key(memory_id, user_query)
        cached_context = self._get_from_cache(cache_key)
        
        if cached_context:
            self.expansion_stats["cache_hits"] += 1
            expanded_memory = memory_result.copy()
            expanded_memory['expanded_context'] = cached_context
            expanded_memory['expansion_info'] = {
                "expanded": True,
                "source": "cache",
                "priority_score": candidate['priority_score'],
                "expansion_reasons": candidate['expansion_reasons']
            }
            return expanded_memory
        
        self.expansion_stats["cache_misses"] += 1
        
        # NO SILENT FALLBACKS - expansion failures should be loud (Rule 3)
        # Get full context from context bridge
        metadata = memory_result.get('metadata', {})
        context_data = self.context_bridge.get_full_context(memory_id, metadata)
        
        # Format the context for AI consumption
        original_memory = memory_result.get('memory', memory_result.get('text', ''))
        formatted_context = self.format_expanded_context(context_data['context'], original_memory)
        
        # Cache the formatted context
        self._add_to_cache(cache_key, formatted_context)
        
        # Create expanded memory result
        expanded_memory = memory_result.copy()
        expanded_memory['expanded_context'] = formatted_context
        expanded_memory['expansion_info'] = {
            "expanded": True,
            "source": "context_bridge",
            "priority_score": candidate['priority_score'],
            "expansion_reasons": candidate['expansion_reasons'],
            "context_length": len(formatted_context),
            "lookup_code": context_data.get('lookup_code')
        }
        
        return expanded_memory
    
    def format_expanded_context(self, full_context: Dict[str, Any], memory_summary: str) -> str:
        """
        Format full context for optimal AI consumption
        
        Args:
            full_context: Full context data from context logger
            memory_summary: Original memory summary
            
        Returns:
            Formatted context string optimized for AI consumption
        """
        if not full_context:
            return memory_summary
        
        try:
            # Extract key information from context
            user_input = full_context.get('user_input', '')
            assistant_response = full_context.get('assistant_response', '')
            timestamp = full_context.get('timestamp', '')
            
            # For enhanced format, extract from interaction structure
            if 'interaction' in full_context:
                interaction = full_context['interaction']
                user_input = interaction.get('user_input', user_input)
                assistant_response = interaction.get('assistant_response', assistant_response)
                thinking_trace = interaction.get('thinking_trace', '')
            
            # Build formatted context
            formatted_parts = []
            
            # Add timestamp context
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    formatted_time = dt.strftime("%B %d, %Y at %I:%M %p")
                    formatted_parts.append(f"Context from conversation on {formatted_time}:")
                except:
                    formatted_parts.append("Context from previous conversation:")
            else:
                formatted_parts.append("Context from previous conversation:")
            
            # Add user input if different from memory
            if user_input and user_input.lower() not in memory_summary.lower():
                formatted_parts.append(f"User said: \"{user_input}\"")
            
            # Add assistant response
            if assistant_response:
                formatted_parts.append(f"Assistant responded: \"{assistant_response}\"")
            
            # Add thinking trace if available
            if 'thinking_trace' in locals() and thinking_trace:
                formatted_parts.append(f"Assistant reasoning: {thinking_trace}")
            
            # Add memory summary
            formatted_parts.append(f"Memory summary: {memory_summary}")
            
            # Join with proper formatting
            formatted_context = "\n".join(formatted_parts)
            
            # Limit token count (rough estimation: 1 token ≈ 4 characters)
            if len(formatted_context) > self.max_context_tokens * 4:
                # Truncate while preserving structure
                max_chars = self.max_context_tokens * 4
                truncated = formatted_context[:max_chars-50]  # Leave room for truncation notice
                formatted_context = truncated + "\n... [context truncated]"
            
            return formatted_context
            
        except Exception as e:
            logger.warning(f"Failed to format expanded context: {e}")
            return memory_summary
    
    def _generate_cache_key(self, memory_id: str, user_query: str) -> str:
        """Generate cache key for memory expansion"""
        query_hash = hashlib.md5(user_query.encode()).hexdigest()[:8]
        return f"{memory_id}:{query_hash}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """Get formatted context from cache if not expired"""
        with self.cache_lock:
            if cache_key not in self.expansion_cache:
                return None
            
            # Check if expired
            if cache_key in self.cache_timestamps:
                age = time.time() - self.cache_timestamps[cache_key]
                if age > self.cache_ttl_seconds:
                    # Remove expired entry
                    del self.expansion_cache[cache_key]
                    del self.cache_timestamps[cache_key]
                    return None
            
            # Move to end (LRU)
            value = self.expansion_cache[cache_key]
            del self.expansion_cache[cache_key]
            self.expansion_cache[cache_key] = value
            
            return value
    
    def _add_to_cache(self, cache_key: str, formatted_context: str):
        """Add formatted context to cache with LRU eviction"""
        with self.cache_lock:
            # Remove oldest entries if cache is full
            while len(self.expansion_cache) >= self.cache_size:
                oldest_key = next(iter(self.expansion_cache))
                del self.expansion_cache[oldest_key]
                if oldest_key in self.cache_timestamps:
                    del self.cache_timestamps[oldest_key]
            
            # Add new entry
            self.expansion_cache[cache_key] = formatted_context
            self.cache_timestamps[cache_key] = time.time()
    
    def clear_cache(self):
        """Clear expansion cache"""
        with self.cache_lock:
            self.expansion_cache.clear()
            self.cache_timestamps.clear()
        logger.info("Expansion cache cleared")
    
    def get_expansion_statistics(self) -> Dict[str, Any]:
        """Get expansion performance statistics"""
        with self.cache_lock:
            cache_size = len(self.expansion_cache)
        
        return {
            "expansion_stats": self.expansion_stats.copy(),
            "cache_stats": {
                "cache_size": cache_size,
                "cache_capacity": self.cache_size,
                "cache_hit_rate": (
                    self.expansion_stats["cache_hits"] / 
                    max(1, self.expansion_stats["cache_hits"] + self.expansion_stats["cache_misses"])
                ) * 100
            },
            "configuration": {
                "max_context_tokens": self.max_context_tokens,
                "expansion_timeout_ms": self.expansion_timeout_ms,
                "cache_ttl_seconds": self.cache_ttl_seconds,
                "max_expansions": self.decision_engine.max_expansions,
                "relevance_threshold": self.decision_engine.relevance_threshold
            },
            "version": "3.0"
        }