#!/usr/bin/env python3
"""
Custom prompts for mem0 to handle our specific memory management needs
Based on mem0's official documentation patterns
"""

# Custom update memory prompt to handle our specific cases
UPDATE_MEMORY_PROMPT = """
You are a memory manager for a personal AI assistant. You will be given new facts and a list of existing memories. Your task is to decide how to update the memory list.

Guidelines:
1. When user says "forget X", mark ALL memories mentioning X for deletion
2. For boss relationships: "X's boss is Y" means Y manages X (not the user)
3. Avoid duplicate memories - merge similar facts
4. Keep user identity consistent (user_001 = Adam)
5. Clean entity names (remove special characters, normalize)

For each existing memory, classify the update as one of the following:
- ADD: New information not in existing memories
- UPDATE: Enhanced or corrected version of existing memory
- DELETE: Memory should be removed (contradicted, user requested forgetting, or duplicate)
- NONE: No change needed

Return a JSON list with format:
[
  {
    "id": "memory_id",
    "text": "updated memory text",
    "event": "UPDATE",
    "old_memory": "original text"
  }
]

Examples:
- If user says "forget Mike", DELETE all memories mentioning Mike
- If "Works with Alicia" exists and new fact is "User works with Alicia", keep the more specific one
- If "Boss is Josh" exists and new fact is "Josh's boss is Dave", UPDATE to clarify hierarchy
"""

# Custom fact extraction prompt for cleaner entities
FACT_EXTRACTION_PROMPT = """
Extract facts from the conversation. Focus on:
1. Personal information (name, workplace, relationships)
2. Work relationships (colleagues, bosses, reports)
3. Preferences and attributes

Rules:
- Use proper names, not system IDs (user_001 -> Adam)
- Normalize entities (the_shire -> Shire)
- Be specific about relationships (X works with Y, X manages Y)
- Ignore meta-information about the memory system itself

Return facts in this format:
{
  "facts": [
    "Adam works at the Shire",
    "Josh manages Adam",
    "Adam works with Alicia"
  ]
}
"""

# Configuration with custom prompts
def get_enhanced_config():
    """Get mem0 configuration with custom prompts for better memory handling"""
    return {
        "version": "v1.1",  # Required for graph support
        "graph_store": {
            "provider": "neo4j",
            "config": {
                "url": "bolt://localhost:7687",
                "username": "neo4j",
                "password": "password123"
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "mem0_enhanced"
            }
        },
        "custom_prompts": {
            "update_memory_prompt": UPDATE_MEMORY_PROMPT,
            "fact_extraction_prompt": FACT_EXTRACTION_PROMPT
        }
    }

# Memory operation helpers that follow mem0 patterns
def format_memory_for_update(memory_id: str, new_text: str, event: str, old_text: str = None):
    """Format memory update following mem0's expected structure"""
    update = {
        "id": memory_id,
        "text": new_text,
        "event": event
    }
    if old_text and event == "UPDATE":
        update["old_memory"] = old_text
    return update

def should_delete_memory(memory_text: str, forget_entities: list) -> bool:
    """Check if a memory should be deleted based on forget requests"""
    memory_lower = memory_text.lower()
    for entity in forget_entities:
        if entity.lower() in memory_lower:
            return True
    return False

def normalize_entity_in_memory(memory_text: str, entity_mappings: dict) -> str:
    """Normalize entities in memory text"""
    result = memory_text
    for old_entity, new_entity in entity_mappings.items():
        # Case-insensitive replacement
        import re
        pattern = re.compile(re.escape(old_entity), re.IGNORECASE)
        result = pattern.sub(new_entity, result)
    return result