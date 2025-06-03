#!/usr/bin/env python3
"""
Fixes for mem0 memory system issues:
1. Entity extraction and sanitization
2. Possessive relationship parsing
3. Memory deduplication
4. Data cleanup utilities
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ParsedRelationship:
    """Represents a parsed relationship with direction"""
    subject: str
    relation: str
    object: str
    confidence: float = 1.0

class MemorySystemFixes:
    """Collection of fixes for mem0 memory system issues"""
    
    def __init__(self):
        self.user_mappings = {
            "user_001": "adam",
            "user": "adam",
            "i": "adam",
            "me": "adam",
            "my": "adam"
        }
        
    def sanitize_entity(self, entity: str) -> str:
        """Clean and normalize entity names"""
        if not entity:
            return ""
            
        # Convert to lowercase for comparison
        entity_lower = entity.lower().strip()
        
        # Map user references to actual name
        if entity_lower in self.user_mappings:
            return self.user_mappings[entity_lower]
            
        # Remove system IDs
        if re.match(r'^user_\d+$', entity_lower):
            return "adam"
            
        # Remove special characters and clean up
        # Keep only alphanumeric, spaces, and basic punctuation
        cleaned = re.sub(r'[^a-zA-Z0-9\s\-\'.]', '', entity)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Handle "the_" prefix
        if cleaned.lower().startswith('the '):
            cleaned = cleaned[4:]
            
        return cleaned.lower()
    
    def parse_possessive_relationship(self, text: str) -> Optional[ParsedRelationship]:
        """Parse possessive relationships like 'X's boss is Y'"""
        patterns = [
            # X's boss is Y
            (r"(\w+)'s\s+boss\s+is\s+(\w+)", lambda m: ParsedRelationship(
                subject=self.sanitize_entity(m.group(2)),
                relation="manages",
                object=self.sanitize_entity(m.group(1))
            )),
            # X is Y's boss
            (r"(\w+)\s+is\s+(\w+)'s\s+boss", lambda m: ParsedRelationship(
                subject=self.sanitize_entity(m.group(1)),
                relation="manages",
                object=self.sanitize_entity(m.group(2))
            )),
            # X works for Y
            (r"(\w+)\s+works\s+for\s+(\w+)", lambda m: ParsedRelationship(
                subject=self.sanitize_entity(m.group(2)),
                relation="manages",
                object=self.sanitize_entity(m.group(1))
            )),
        ]
        
        text_lower = text.lower()
        for pattern, builder in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return builder(match)
                
        return None
    
    def extract_clean_relationships(self, text: str) -> List[ParsedRelationship]:
        """Extract relationships with proper entity sanitization"""
        relationships = []
        
        # First check for possessive relationships
        possessive = self.parse_possessive_relationship(text)
        if possessive:
            relationships.append(possessive)
            
        # Common relationship patterns
        patterns = [
            (r"(\w+)\s+works\s+at\s+(?:the\s+)?(\w+)", "works_at"),
            (r"(\w+)\s+works\s+with\s+(\w+)", "works_with"),
            (r"(\w+)\s+is\s+friends?\s+with\s+(\w+)", "friend_of"),
            (r"(\w+)\s+(?:is\s+)?training\s+(\w+)", "trains"),
            (r"(\w+)\s+got\s+fired", "was_fired"),
        ]
        
        text_lower = text.lower()
        for pattern, relation in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                subj = self.sanitize_entity(match.group(1))
                obj = self.sanitize_entity(match.group(2)) if match.lastindex >= 2 else None
                
                # Skip if entities are too short or invalid
                if len(subj) < 2 or (obj and len(obj) < 2):
                    continue
                    
                # Skip if entity is just numbers or special chars
                if re.match(r'^[\d_]+$', subj) or (obj and re.match(r'^[\d_]+$', obj)):
                    continue
                    
                if obj:
                    relationships.append(ParsedRelationship(
                        subject=subj,
                        relation=relation,
                        object=obj
                    ))
                    
        return relationships
    
    def calculate_memory_similarity(self, mem1: str, mem2: str) -> float:
        """Calculate similarity between two memory strings"""
        # Normalize for comparison
        norm1 = mem1.lower().strip()
        norm2 = mem2.lower().strip()
        
        # Exact match
        if norm1 == norm2:
            return 1.0
            
        # Use sequence matcher for fuzzy matching
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def deduplicate_memories(self, memories: List[Dict[str, Any]], threshold: float = 0.85) -> List[Dict[str, Any]]:
        """Remove duplicate memories based on similarity threshold"""
        if not memories:
            return []
            
        unique_memories = []
        seen_hashes = set()
        
        for memory in memories:
            memory_text = memory.get('memory', memory.get('text', ''))
            if not memory_text:
                continue
                
            # Create normalized version for comparison
            normalized = memory_text.lower().strip()
            normalized = re.sub(r'\s+', ' ', normalized)
            
            # Check for exact duplicates via hash
            mem_hash = hashlib.md5(normalized.encode()).hexdigest()
            if mem_hash in seen_hashes:
                continue
                
            # Check for semantic duplicates
            is_duplicate = False
            for unique_mem in unique_memories:
                unique_text = unique_mem.get('memory', unique_mem.get('text', ''))
                similarity = self.calculate_memory_similarity(memory_text, unique_text)
                
                if similarity >= threshold:
                    is_duplicate = True
                    # If the new memory is longer, replace the old one
                    if len(memory_text) > len(unique_text):
                        unique_memories.remove(unique_mem)
                        unique_memories.append(memory)
                        seen_hashes.add(mem_hash)
                    break
                    
            if not is_duplicate:
                unique_memories.append(memory)
                seen_hashes.add(mem_hash)
                
        return unique_memories
    
    def clean_relationships(self, relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean up malformed relationships"""
        cleaned = []
        
        for rel in relationships:
            source = self.sanitize_entity(rel.get('source', ''))
            relation = rel.get('relationship', rel.get('relation', ''))
            target = self.sanitize_entity(rel.get('target', rel.get('destination', '')))
            
            # Skip invalid relationships
            if not source or not relation or not target:
                continue
                
            # Skip relationships with malformed entities
            if len(source) < 2 or len(target) < 2:
                continue
                
            # Skip relationships with excessive length (likely malformed)
            if len(source) > 50 or len(target) > 50:
                logger.warning(f"Skipping malformed relationship: {source} -> {relation} -> {target}")
                continue
                
            # Skip self-relationships unless they make sense
            if source == target and relation not in ['manages', 'reports_to']:
                continue
                
            cleaned.append({
                'source': source,
                'relationship': relation,
                'target': target
            })
            
        # Remove duplicate relationships
        unique = []
        seen = set()
        for rel in cleaned:
            key = (rel['source'], rel['relationship'], rel['target'])
            if key not in seen:
                seen.add(key)
                unique.append(rel)
                
        return unique
    
    def merge_memory_facts(self, memories: List[str]) -> Dict[str, List[str]]:
        """Group related memories by entity"""
        entity_facts = {}
        
        for memory in memories:
            # Extract entities mentioned in the memory
            words = memory.lower().split()
            for word in words:
                clean_word = self.sanitize_entity(word)
                if clean_word and len(clean_word) > 2:
                    if clean_word not in entity_facts:
                        entity_facts[clean_word] = []
                    if memory not in entity_facts[clean_word]:
                        entity_facts[clean_word].append(memory)
                        
        return entity_facts


# Example usage and tests
if __name__ == "__main__":
    fixer = MemorySystemFixes()
    
    # Test possessive parsing
    print("Testing possessive parsing:")
    test_cases = [
        "Josh's boss is Dave",
        "Dave is Josh's boss",
        "Mike works for Dave"
    ]
    
    for test in test_cases:
        result = fixer.parse_possessive_relationship(test)
        if result:
            print(f"{test} => {result.subject} {result.relation} {result.object}")
    
    # Test entity sanitization
    print("\nTesting entity sanitization:")
    entities = ["user_001", "the_shire", "i_am!__with_every_prompt!", "USER", "My boss"]
    for entity in entities:
        print(f"{entity} => {fixer.sanitize_entity(entity)}")
    
    # Test relationship extraction
    print("\nTesting relationship extraction:")
    texts = [
        "Adam works at the Shire",
        "Joe works with the user",
        "I am training Mike",
        "Josh's boss is Dave"
    ]
    
    for text in texts:
        rels = fixer.extract_clean_relationships(text)
        for rel in rels:
            print(f"{text} => {rel.subject} {rel.relation} {rel.object}")