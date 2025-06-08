# Pattern-Based Auto-Expansion Implementation Plan

## Executive Summary
Add verbatim/detail request pattern detection to the existing context expansion system without breaking any current functionality. When users ask for exact words, literal phrases, or specific details, automatically trigger context expansion regardless of relevance scores.

## Current System Analysis

### Existing Context Expansion Flow
```
User Query → Memory Search → ContextExpansionDecision.should_expand()
├─ Check relevance threshold (0.55/0.65)
├─ Check lookup code exists  
├─ Check temporal patterns
├─ Check query context patterns
└─ Require 2+ positive indicators → Expand or Skip
```

### The Problem with "Fnord"
- User: "Hello, my name is Adam! Fnord."
- mem0 extracts: "Name is Adam" (correctly filters out Fnord)
- Query: "What came after 'My name is Adam'?"
- Current result: No expansion (low relevance + no verbatim patterns)
- Desired result: Auto-expand to show full conversation

## Solution Architecture

### Core Principle: Additive Pattern Matching
Add a new pattern category that **forces expansion** when verbatim/literal content is requested, bypassing normal relevance requirements while preserving all existing functionality.

## Detailed Implementation Plan

### Phase 1: Pattern Definition (Zero Breaking Changes)

**File: `context_expander.py`**
**Location: `ContextExpansionDecision.__init__()`**

```python
# Add after existing pattern definitions (around line 77)
# Verbatim/literal content request patterns
self.verbatim_request_patterns = [
    r'\b(what did i say after|exact words|specifically said|verbatim)\b',
    r'\b(what came after|what followed|next word|word after)\b',
    r'\b(quote|exact phrase|word for word|literally said)\b',
    r'\b(exact|precisely|specifically)\b.*\b(said|told|mentioned)\b',
    r'\b(full|complete|entire)\b.*\b(sentence|phrase|statement)\b',
    r'\b(actual words|actual phrase|literally)\b',
    r'\b(original|unfiltered)\b.*\b(conversation|message|input)\b'
]

logger.info(f"ContextExpansionDecision initialized with verbatim patterns: {len(self.verbatim_request_patterns)}")
```

### Phase 2: Decision Logic Enhancement (Backward Compatible)

**File: `context_expander.py`**
**Location: `ContextExpansionDecision.should_expand()` method (around line 83)**

```python
def should_expand(self, memory_result: Dict[str, Any], user_query: str, relevance_score: float) -> Tuple[bool, List[str]]:
    """
    Decide if a memory should have its context expanded
    NOW INCLUDES: Verbatim request override for literal content queries
    """
    if not memory_result:
        return False, ["empty_memory_result"]
    
    if not user_query or not user_query.strip():
        return False, ["empty_user_query"]
    
    reasons = []
    
    # NEW: Check for verbatim/literal content requests FIRST
    verbatim_requested = self._query_requests_verbatim_content(user_query)
    if verbatim_requested:
        reasons.append("verbatim_content_requested")
        # Still need lookup code for context retrieval
        metadata = memory_result.get('metadata', {})
        lookup_code = metadata.get('context_lookup_code') or metadata.get('lookup_code')
        if not lookup_code:
            return False, ["verbatim_requested_but_no_lookup_code"]
        reasons.append("has_lookup_code")
        
        # For verbatim requests, bypass normal relevance thresholds
        reasons.append("verbatim_override")
        logger.debug(f"Verbatim expansion triggered for query: {user_query[:50]}...")
        return True, reasons
    
    # EXISTING: All current logic remains unchanged
    # Primary check: Relevance score threshold
    if relevance_score < self.relevance_threshold:
        return False, [f"low_relevance_score_{relevance_score:.3f}"]
    
    reasons.append(f"high_relevance_{relevance_score:.3f}")
    
    # ... [rest of existing logic unchanged]
```

### Phase 3: Pattern Detection Method (New Helper)

**File: `context_expander.py`**
**Location: Add after `_query_suggests_context_need()` method (around line 219)**

```python
def _query_requests_verbatim_content(self, user_query: str) -> bool:
    """
    Detect if user is requesting exact/literal/verbatim content
    
    These queries should trigger expansion regardless of relevance scores
    because the user explicitly wants the full/exact conversation content.
    """
    query_lower = user_query.lower()
    
    for pattern in self.verbatim_request_patterns:
        if re.search(pattern, query_lower):
            logger.debug(f"Verbatim pattern matched: {pattern}")
            return True
    
    # Additional heuristics for verbatim requests
    # Questions about "after X" or "before X" often want literal content
    if re.search(r'\b(after|before)\b.*["\'].*["\']', user_query):
        logger.debug("Quote-based verbatim pattern detected")
        return True
    
    # Questions about specific words/phrases
    if re.search(r'\bwhat.*word.*\b(after|before|said|mentioned)\b', query_lower):
        logger.debug("Word-specific verbatim pattern detected")
        return True
    
    return False
```

### Phase 4: Enhanced Context Formatting (Quality Improvement)

**File: `context_expander.py`**
**Location: `ContextExpander.format_expanded_context()` method (around line 635)**

```python
def format_expanded_context(self, full_context: Dict[str, Any], memory_summary: str, expansion_reasons: List[str] = None) -> str:
    """
    Format full context for optimal AI consumption
    NOW INCLUDES: Special formatting for verbatim requests
    """
    if not full_context:
        return memory_summary
    
    try:
        # Check if this is a verbatim content request
        is_verbatim_request = expansion_reasons and any(
            reason.startswith("verbatim") for reason in expansion_reasons
        )
        
        # Extract key information from context
        user_input = full_context.get('user_input', '')
        assistant_response = full_context.get('assistant_response', '')
        timestamp = full_context.get('timestamp', '')
        
        # ... [existing extraction logic unchanged]
        
        formatted_parts = []
        
        if is_verbatim_request:
            # Special formatting for verbatim requests - emphasize exact content
            formatted_parts.append("📝 EXACT CONVERSATION CONTENT:")
            formatted_parts.append("=" * 40)
        
        # Add timestamp context
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime("%B %d, %Y at %I:%M %p")
                formatted_parts.append(f"Conversation on {formatted_time}:")
            except:
                formatted_parts.append("Previous conversation:")
        else:
            formatted_parts.append("Previous conversation:")
        
        # Add user input with special emphasis for verbatim requests
        if user_input:
            if is_verbatim_request:
                formatted_parts.append(f"USER SAID EXACTLY: \"{user_input}\"")
            else:
                formatted_parts.append(f"User said: \"{user_input}\"")
        
        # ... [rest of existing formatting logic unchanged]
        
        return formatted_context
        
    except Exception as e:
        logger.warning(f"Failed to format expanded context: {e}")
        return memory_summary
```

### Phase 5: Integration Safety (Zero Risk)

**File: `context_expander.py`**
**Location: `ContextExpander._expand_single_memory()` method (around line 574)**

```python
# Modify the call to format_expanded_context to pass expansion reasons
formatted_context = self.format_expanded_context(
    context_data['context'], 
    original_memory,
    candidate.get('expansion_reasons', [])  # Pass reasons for verbatim detection
)
```

**File: `context_expander.py`**
**Location: Configuration section (around line 32)**

```python
# Add feature flag for verbatim expansion (around line 35)
def __init__(self, 
             relevance_threshold: float = 0.55,
             recency_days: int = 30,
             max_expansions: int = 3,
             enable_verbatim_expansion: bool = True):  # NEW: Feature flag
    
    # ... existing validation ...
    
    self.enable_verbatim_expansion = enable_verbatim_expansion
    
    # Only initialize verbatim patterns if feature enabled
    if self.enable_verbatim_expansion:
        self.verbatim_request_patterns = [...]
    else:
        self.verbatim_request_patterns = []
```

## Testing Strategy

### Phase T1: Unit Testing (Safe)

**Test Verbatim Pattern Detection:**
```python
def test_verbatim_patterns():
    decision_engine = ContextExpansionDecision()
    
    # Should trigger
    assert decision_engine._query_requests_verbatim_content("What did I say after my name?")
    assert decision_engine._query_requests_verbatim_content("Quote my exact words")
    assert decision_engine._query_requests_verbatim_content("What came after 'hello'?")
    
    # Should not trigger  
    assert not decision_engine._query_requests_verbatim_content("Tell me about my friend")
    assert not decision_engine._query_requests_verbatim_content("What do you know about me?")
```

### Phase T2: Integration Testing (Controlled)

**Test with "Fnord" Scenario:**
1. Create conversation: "Hello, my name is Adam! Fnord."
2. Query: "What did I say after 'My name is Adam'?"
3. Verify: Expansion triggers and shows full conversation
4. Verify: Normal expansions still work unchanged

### Phase T3: Regression Testing (Safety Net)

**Verify No Breaking Changes:**
1. Test all existing expansion scenarios still work
2. Test relevance thresholds unchanged for normal queries
3. Test temporal patterns still work
4. Test cache performance unaffected

## Risk Mitigation

### 1. Feature Flag Protection
```python
# Environment variable override
ENABLE_VERBATIM_EXPANSION = os.getenv("ENABLE_VERBATIM_EXPANSION", "true").lower() == "true"
```

### 2. Logging and Monitoring
```python
# Track verbatim expansion usage
if verbatim_requested:
    logger.info(f"VERBATIM EXPANSION: Query='{user_query[:50]}...', Memory='{memory_id[:8]}'")
```

### 3. Graceful Degradation
- If verbatim patterns fail, fall back to normal expansion logic
- If context retrieval fails, return original memory
- All existing error handling preserved

### 4. Performance Protection
- Verbatim check happens early (before expensive operations)
- Pattern regex compilation cached at initialization
- No additional API calls required

## Expected Results

### Before Implementation
```
User: "What did I say after 'My name is Adam'?"
System: Searches "Name is Adam" memory → Low relevance → No expansion
AI: "I don't have access to past conversations or specific words you've said."
```

### After Implementation
```
User: "What did I say after 'My name is Adam'?"
System: Detects verbatim pattern → Forces expansion → Retrieves full context
AI: "After saying 'My name is Adam', you said 'Fnord'."
```

## Implementation Timeline

- **Day 1**: Phase 1-2 (Pattern detection + decision logic)
- **Day 2**: Phase 3-4 (Helper method + enhanced formatting)  
- **Day 3**: Phase 5 + Testing (Integration + safety)
- **Day 4**: Testing + Documentation + Deployment

## Success Metrics

1. **Verbatim queries trigger expansion**: 95% success rate
2. **No regression in existing functionality**: 100% compatibility
3. **Performance impact**: <100ms additional processing time
4. **User satisfaction**: Accurate responses to literal content requests

## Implementation Notes

### Key Benefits
- **Zero Breaking Changes**: All existing functionality preserved
- **Intelligent Override**: Bypasses relevance thresholds when appropriate
- **Enhanced User Experience**: Answers literal content questions accurately
- **Maintainable**: Clean separation of verbatim vs normal expansion logic

### Design Decisions
- **Pattern-based Detection**: More reliable than AI interpretation
- **Early Override**: Verbatim check happens before expensive operations
- **Feature Flag**: Can be disabled if issues arise
- **Enhanced Formatting**: Special display for verbatim content requests

This plan provides comprehensive verbatim request handling while maintaining 100% backward compatibility and zero risk to existing functionality.