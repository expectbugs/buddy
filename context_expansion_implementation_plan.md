# Context Expansion Implementation Plan for mem0 Integration

## Overview
This plan outlines a careful, step-by-step approach to add full-context logging and smart context expansion to our existing mem0-based system (v1.0) without breaking current functionality.

## Core Principles
1. **Non-Breaking**: All changes must be additive, not modifying existing behavior
2. **Graceful Degradation**: System continues to work even if new features fail
3. **Backward Compatible**: Existing memories remain functional
4. **Minimal Dependencies**: Reuse mem0's existing infrastructure where possible

## Phase 1: Foundation Layer (No Breaking Changes)

### 1.1 Add Context Logger Module
Create `context_logger.py` that runs alongside mem0:
- Implement FullContextLogger class (port from old system)
- Store logs in `/var/log/buddy/full_context/` (separate from mem0)
- Generate lookup codes (CTX-{timestamp}-{uuid})
- Log every interaction with full context
- Create lookup index for fast retrieval

### 1.2 Extend Memory Metadata
Without modifying mem0's core behavior, add metadata to memories:
- When calling `memory.add()`, include in metadata:
  - `lookup_code`: Link to full context log
  - `timestamp_utc`: ISO format timestamp
  - `unix_timestamp`: For fast sorting
  - `context_type`: "full", "summary", or "standard"
  - `line_numbers`: If from logs (start_line, end_line)

### 1.3 Create Context Bridge
Build `context_bridge.py` to connect mem0 with context logs:
- `get_full_context(memory_id)`: Retrieve full context for any memory
- `get_context_window(memory_id, before=2, after=2)`: Get surrounding context
- Works by reading lookup_code from memory metadata
- Falls back gracefully if no context available

## Phase 2: Enhanced Memory Storage (Backward Compatible)

### 2.1 Wrap Memory Operations
Create `enhanced_memory.py` that wraps mem0's Memory class:
```python
class EnhancedMemory:
    def __init__(self, mem0_memory):
        self.memory = mem0_memory
        self.context_logger = FullContextLogger()
    
    def add(self, messages, user_id, metadata=None):
        # Log full context first
        lookup_code = self.context_logger.log_interaction(messages)
        
        # Enhance metadata
        enhanced_metadata = {
            **(metadata or {}),
            'lookup_code': lookup_code,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'unix_timestamp': time.time()
        }
        
        # Call original mem0 add
        return self.memory.add(messages, user_id, enhanced_metadata)
```

### 2.2 Update run.py Gradually
Modify `run.py` to use EnhancedMemory:
- Replace `Memory()` with `EnhancedMemory(Memory())`
- All existing functionality remains unchanged
- New memories automatically get context logging

## Phase 3: Smart Context Expansion (Additive Features)

### 3.1 Create Context Expander
Build `context_expander.py`:
- `should_expand(memory, relevance_score)`: Determine if expansion needed
- `expand_memory(memory)`: Replace summary with full context
- `add_surrounding_context(memory, window_size)`: Add before/after lines
- Works with memory metadata, doesn't modify mem0 internals

### 3.2 Enhance Search Results
Modify search handling in `run.py`:
```python
# After getting search results from mem0
results = self.memory.search(query, user_id)

# Enhance high-relevance results
for result in results:
    if result['score'] > 0.85 and 'lookup_code' in result['metadata']:
        # Expand context without modifying the memory itself
        result['expanded_context'] = self.context_expander.expand(result)
```

### 3.3 Add Context Commands
New user commands that don't interfere with existing ones:
- `/context <memory_id>`: Show full context for a memory
- `/expand <query>`: Search with automatic context expansion
- `/timeline`: Show memories with full timestamps

## Phase 4: Integration Testing & Rollout

### 4.1 Parallel Testing
- Run new system alongside old for comparison
- Verify no regression in existing functionality
- Test context expansion accuracy

### 4.2 Migration Tools
- Script to add lookup codes to existing memories (optional)
- Backfill context logs from any available sources
- Maintain compatibility with memories lacking context

### 4.3 Performance Optimization
- Implement caching for frequently accessed contexts
- Async logging to prevent blocking
- Efficient lookup index management

## Implementation Order

1. **Week 1**: Implement Phase 1 (Foundation)
   - Start with context logger
   - Test logging without touching mem0
   - Verify no performance impact

2. **Week 2**: Implement Phase 2 (Enhanced Storage)
   - Wrap memory operations
   - Test metadata enhancement
   - Ensure backward compatibility

3. **Week 3**: Implement Phase 3 (Context Expansion)
   - Build expansion logic
   - Test with real conversations
   - Measure relevance improvements

4. **Week 4**: Testing & Polish
   - Full integration testing
   - Performance optimization
   - Documentation updates

## Risk Mitigation

1. **Feature Flags**: Add environment variables to disable new features
   - `ENABLE_CONTEXT_LOGGING=false`
   - `ENABLE_CONTEXT_EXPANSION=false`

2. **Graceful Failures**: All new features fail silently
   - Missing context logs don't break memory storage
   - Failed expansions return original memory

3. **Monitoring**: Add metrics for new features
   - Context log write success rate
   - Expansion request rate
   - Performance impact measurements

## Success Criteria

1. Zero breaking changes to v1.0 functionality
2. All existing tests continue to pass
3. Context expansion improves relevance for 80%+ of queries
4. Performance impact < 5% for standard operations
5. Clean separation of concerns between mem0 and context system

## Next Steps

1. Review and approve this plan
2. Create feature branch `feature/context-expansion`
3. Implement Phase 1 with comprehensive tests
4. Iterate based on results before proceeding to Phase 2