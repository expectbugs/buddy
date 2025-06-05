# Mem0 Relationship/Graph Data Research Summary

## Overview
This research investigated how to retrieve and display relationship/graph data from the mem0 library, which stores both vector-based memories and graph-based relationships in separate stores.

## Key Findings

### 1. Memory vs. Relationship Data Structure

**Vector Store (Memories):**
- Accessed via: `memory.get_all(user_id=user_id)`
- Returns: Dictionary with 'results' key containing memory objects
- Contains: Text-based memories and conversations
- Current count: 2 memories

**Graph Store (Relationships):**
- Accessed via: `memory.graph.get_all(filters={'user_id': user_id})`
- Returns: List of relationship objects
- Contains: Structured entity-relationship-entity data
- Current count: 32 relationships

### 2. Available Graph Methods

The `Memory.graph` object provides:
- `get_all(filters, limit=100)` - Get all relationships matching filters
- `search(query, filters, limit=100)` - Search for relationships containing query terms

**Important:** The `user_id` must be passed in the `filters` parameter, not as a direct argument.

### 3. Relationship Data Format

Each relationship object contains:
```python
{
    'source': 'entity1',           # Source entity
    'relationship': 'relation_type', # Relationship type
    'target': 'entity2',           # Target entity (or 'destination')
}
```

### 4. Current Relationship Types in System

The analysis revealed 22 different relationship types:
- **WORKING_WITH** - Collaboration relationships
- **HAS_BOSS** - Management hierarchy (josh → dave, adam_001 → josh)
- **IS_TRAINING** - Training relationships
- **WORKS_AT** - Employment/location relationships
- **HAS_FRIEND**, **HAS_COLLEAGUE** - Social relationships
- And 16 more types for various semantic relationships

### 5. Implementation

Added three new methods to `ImprovedMem0Assistant`:

1. **`show_all_relationships(user_id)`** - Displays all relationships grouped by type
2. **`search_relationships(query, user_id)`** - Searches for specific relationships
3. Enhanced main loop with new commands:
   - `relationships` - Show all relationships
   - `search relationships <query>` - Search for specific relationships

### 6. Key Differences from Basic Memories

| Aspect | Memories (`memory.get_all()`) | Relationships (`memory.graph.get_all()`) |
|--------|------------------------------|------------------------------------------|
| **Data Type** | Unstructured text | Structured entity-relationship-entity |
| **Storage** | Vector store (Qdrant) | Graph store (Neo4j) |
| **Count** | 2 items | 32 items |
| **Format** | Conversational text | Semantic triplets |
| **Purpose** | Context for conversations | Entity relationship mapping |

### 7. Log Analysis

The logs showing "Retrieved 26 relationships" (now 32) were referring to the graph store data, which was not being displayed in the original implementation. The relationship data was being stored and retrieved correctly, but only the basic memories were shown to users.

## Usage Examples

```python
# Get all relationships
relationships = memory.graph.get_all(filters={'user_id': 'adam_001'})

# Search for specific entity relationships  
dave_rels = memory.graph.search(query='Dave', filters={'user_id': 'adam_001'})

# Search for specific relationship types
boss_rels = memory.graph.search(query='boss', filters={'user_id': 'adam_001'})
```

## Enhanced Commands

The updated system now supports:
- `memories` - Show vector store memories (conversation context)
- `relationships` - Show graph store relationships (entity connections)
- `search relationships <query>` - Search specific relationship patterns

## Conclusion

The mem0 system was correctly storing both memories and relationships, but the original implementation only displayed the vector store memories. The graph store contains rich relationship data that provides structured information about entity connections, hierarchies, and semantic relationships that complement the conversational memories.

The relationship data shows a much richer picture of the stored knowledge, including proper management hierarchy (josh → dave), training relationships, and various semantic connections that were extracted from conversations but stored separately from the basic memory text.