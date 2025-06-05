# Mem0 Memory Extraction and Filtering Research

## Overview
This research examines how mem0 and mem0g handle memory extraction and filtering, focusing on understanding their implementation to address overly aggressive filtering issues.

## 1. Mem0 Memory Extraction Logic

### Core Extraction Process
Mem0 uses a multi-step LLM-based extraction process:

1. **Input Processing**: Parses conversation messages
2. **Fact Extraction**: Uses LLM with specific prompts to extract key facts
3. **Memory Search**: Searches existing memories for related content
4. **Action Determination**: Decides whether to add/update/delete memories

### Extraction Implementation
```python
# From mem0/memory/main.py
system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)
# Uses FACT_RETRIEVAL_PROMPT from mem0/configs/prompts.py
```

### Memory Addition Flow
```python
# Memories are added after each conversation turn
messages.append({"role": "assistant", "content": assistant_response})
memory.add(messages, user_id=user_id)
```

## 2. Mem0 Fact Retrieval Criteria

### What Mem0 Considers "Memory-Worthy"
Based on the FACT_RETRIEVAL_PROMPT, mem0 extracts:

1. **Personal Preferences**: Likes, dislikes, preferences in food, products, activities
2. **Personal Details**: Names, relationships, important dates
3. **Plans and Intentions**: Upcoming events, trips, goals
4. **Activity Preferences**: Dining, travel, hobby preferences
5. **Health and Wellness**: Dietary restrictions, fitness routines
6. **Professional Information**: Job titles, work habits, career goals
7. **Miscellaneous Details**: Favorite books, movies, brands

### Extraction Guidelines
- Extract facts from user and assistant messages only
- Ignore system messages
- Detect and record facts in original language
- Return facts in JSON format with "facts" key

### Example Transformations
```
Input: "Hi, my name is John. I am a software engineer."
Facts: ["Name is John", "Is a Software engineer"]

Input: "Yesterday, I had a meeting with John at 3pm. We discussed the new project."
Facts: ["Had a meeting with John at 3pm", "Discussed the new project"]
```

## 3. Mem0 Filtering and Configuration

### Session-Based Filtering
Mem0 requires one of these identifiers for memory scoping:
- `user_id`
- `agent_id` 
- `run_id`

Optional additional filters:
- `actor_id`

### Configuration Options
1. **Custom Extraction Prompts**: Override default FACT_RETRIEVAL_PROMPT
2. **Inference Control**: Use `infer=False` to disable automatic inference
3. **Memory Type Specification**: E.g., procedural memory
4. **Embedding Model Configuration**: Configurable embedding model and vector store
5. **Similarity Thresholds**: For memory similarity searches

### Extraction Sensitivity Controls
- Custom prompts can modify extraction criteria
- Threshold settings for memory similarity
- Ability to add raw memories without automatic processing

## 4. Mem0g Differences

### Key Distinctions from Mem0
Mem0g appears to be a fork of mem0 with these claimed improvements:
- **Performance**: 91% faster responses, 90% lower token usage
- **Accuracy**: +26% accuracy over OpenAI Memory
- **Multi-level Memory**: Enhanced user, session, and agent state tracking

### Memory Retrieval Approach
```python
# Dynamic memory retrieval during conversation
relevant_memories = memory.search(query=message, user_id=user_id, limit=3)
memories_str = "\n".join(f"- {entry['memory']}" for entry in relevant_memories["results"])
```

### Architecture
- Supports multiple LLMs (default: GPT-4o-mini)
- Cross-platform SDKs
- Fully managed service option

## 5. Known Issues and Limitations

### Performance Issues
- Memory addition can take 20+ seconds (Issue #2813)
- Async/sync coordination problems (Issue #2892)
- Memory deletion complications with vector databases (Issue #2891)

### Stability Concerns
- Memory extraction sensitive to system updates (Issue #2880)
- Potential breaking changes in memory behavior across versions

## 6. Recommendations for Reducing Aggressive Filtering

### 1. Custom Prompt Modification
Override the default FACT_RETRIEVAL_PROMPT with more permissive criteria:
```python
# Instead of strict categorization, use broader extraction
custom_prompt = """Extract any potentially useful information from the conversation,
including casual mentions, preferences, and contextual details that might be relevant
for future interactions."""
```

### 2. Configuration Adjustments
- Lower similarity thresholds for memory searches
- Increase memory retention limits
- Use more inclusive filtering criteria

### 3. Inference Control
- Use `infer=False` for specific messages that should be stored as-is
- Implement custom logic for determining when to extract vs store raw

### 4. Multi-Level Approach
- Implement different extraction sensitivity for different memory types
- Use session-specific vs user-specific extraction criteria
- Consider temporal relevance in extraction decisions

## 7. Implementation Insights

### Key Learnings
1. **LLM-Dependent**: Extraction quality heavily depends on the LLM and prompt design
2. **Context-Aware**: Uses existing memories to inform new extraction decisions
3. **Configurable**: Multiple configuration points for customizing behavior
4. **Performance Trade-offs**: More thorough extraction comes with latency costs

### Best Practices
1. Monitor extraction performance and adjust prompts accordingly
2. Use custom prompts for domain-specific applications
3. Implement fallback mechanisms for extraction failures
4. Consider using multiple extraction strategies for different content types

## Conclusion

Mem0's memory extraction is primarily controlled through:
1. The FACT_RETRIEVAL_PROMPT design
2. LLM model choice and configuration
3. Similarity thresholds and filtering criteria
4. Session scoping and context management

To reduce overly aggressive filtering, focus on customizing the extraction prompts, adjusting similarity thresholds, and implementing more nuanced extraction logic that considers context and user intent.