# V3 Specialized Memory Agents Plan

## Vision
Transition from single general-purpose LLM to three fine-tuned specialized memory agents, each optimized for specific memory operations.

## Current State (v1.2)
- Single LLM (GPT-4o-mini) handles all operations:
  - User chat responses
  - Memory extraction for Qdrant
  - Relationship extraction for Neo4j
  - Memory search decisions
- Full context logging captures all memory operations
- mem0 default prompts for fact/relationship extraction

## Proposed V3 Architecture: Three Specialized Agents

### Agent 1: Semantic Memory Agent (Qdrant Specialist)
**Purpose**: Extract and summarize factual information for vector storage

**Specialization**:
- Semantic fact extraction from conversations
- Intelligent summarization for embedding optimization
- Priority scoring for memory importance
- Deduplication detection
- Temporal context integration

**Training Data Sources**:
- Full context logs showing successful fact extractions
- Examples of good vs. poor memory candidates
- Summarization quality examples
- User feedback on memory relevance

**Fine-tuning Target**: Small model (7B-13B parameters) trained specifically on factual extraction patterns

### Agent 2: Relational Memory Agent (Neo4j Specialist)
**Purpose**: Extract and structure relationship data for graph storage

**Specialization**:
- Entity relationship detection
- Hierarchical relationship mapping (boss/subordinate, family, etc.)
- Relationship type classification
- Entity disambiguation and normalization
- Multi-hop relationship inference

**Training Data Sources**:
- Full context logs showing relationship extractions
- Examples of complex relationship parsing ("Josh's boss is Dave" → "Dave manages Josh")
- Relationship type taxonomies
- Graph structure optimization examples

**Fine-tuning Target**: Small model (7B-13B parameters) trained on relationship pattern recognition

### Agent 3: Context Intelligence Agent (Search & Expansion Specialist)
**Purpose**: Decide which memories to retrieve and when to expand context

**Specialization**:
- Query interpretation for memory search
- Relevance scoring for memory retrieval
- Context expansion decision making (when to load full context vs. summaries)
- Cross-modal memory correlation (vector + graph)
- Conversation flow analysis

**Training Data Sources**:
- Full context logs showing search decisions and results
- Examples of successful vs. unsuccessful memory retrievals
- Context expansion effectiveness data
- User satisfaction patterns with memory relevance

**Fine-tuning Target**: Small model (7B-13B parameters) trained on search and retrieval optimization

## Training Data Pipeline

### Data Collection (Already in Progress)
- **Full Context Logs**: `/var/log/buddy/full_context/` contains detailed interaction logs
- **Memory Operations**: Logs show memory extraction, storage, and retrieval decisions
- **User Interactions**: Complete conversation history with memory handling metadata

### Data Processing for Fine-tuning
1. **Extract Training Examples**: Parse context logs for specific agent training data
2. **Label Quality**: Identify successful vs. unsuccessful memory operations
3. **Create Specialized Datasets**: Separate data streams for each agent type
4. **Validation Sets**: Hold out data for agent performance testing

### Training Approach
1. **Base Model Selection**: Start with efficient base models (Mistral-7B, Llama2-7B, etc.)
2. **Specialized Fine-tuning**: Train each agent on its specific task domain
3. **Coordination Training**: Train agents to work together effectively
4. **Performance Validation**: Test against current single-model approach

## Implementation Phases

### Phase 3A: Data Preparation
- Parse existing context logs for training data
- Create labeled datasets for each agent type
- Build data quality assessment tools
- Establish training/validation splits

### Phase 3B: Agent Development
- Fine-tune three specialized models
- Develop agent coordination protocols
- Create fallback mechanisms for agent failures
- Build performance monitoring tools

### Phase 3C: Integration & Testing
- Replace single LLM with three-agent system
- A/B testing against v1.2 system
- Performance optimization and tuning
- User experience validation

## Expected Benefits

### Performance Improvements
- **Faster Operations**: Smaller specialized models vs. large general model
- **Better Accuracy**: Domain-specific training for each memory operation type
- **Lower Costs**: Smaller models require less computational resources
- **Parallel Processing**: Multiple agents can work simultaneously

### System Capabilities
- **Specialized Expertise**: Each agent optimized for its specific domain
- **Improved Memory Quality**: Better extraction and storage decisions
- **Enhanced Search**: More intelligent memory retrieval
- **Scalable Architecture**: Easier to improve individual components

## Technical Requirements

### Infrastructure
- Model hosting for three fine-tuned models
- Agent coordination service
- Performance monitoring and logging
- Fallback to general LLM if agents fail

### Development Tools
- Fine-tuning pipeline for model training
- Data processing tools for context log analysis
- Agent testing and validation frameworks
- Performance comparison tools

## Success Metrics

### Quantitative
- Memory extraction accuracy improvement
- Search relevance score improvements
- Response time reduction
- Cost per operation reduction

### Qualitative
- User satisfaction with memory relevance
- Conversation flow quality
- System reliability and robustness
- Memory organization effectiveness

## Timeline

**Prerequisites**: Complete Phase 3 (Smart Context Expansion)
**Estimated Timeline**: 6-12 months of development
**Dependencies**: 
- Sufficient training data from context logs
- Compute resources for model fine-tuning
- Performance baseline establishment

---

## Notes

This represents a significant architectural evolution from general-purpose to specialized AI agents, leveraging the rich training data we're already collecting through our context logging system. The three-agent approach allows for domain expertise while maintaining system coherence through proper coordination protocols.