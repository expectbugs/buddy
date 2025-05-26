#!/usr/bin/env python3

"""
Verify memory integrity by inspecting stored data in both Qdrant and Neo4j
"""

from qdrant_client import QdrantClient
from neo4j import GraphDatabase
import json

def verify_qdrant_memories():
    """Inspect all memories stored in Qdrant"""
    print("\n" + "="*60)
    print("🔍 QDRANT MEMORY VERIFICATION")
    print("="*60)
    
    client = QdrantClient(host="localhost", port=6333)
    
    # Get collection info
    collections = client.get_collections()
    print("\n📚 Collections found:")
    for collection in collections.collections:
        print(f"  - {collection.name}")
        info = client.get_collection(collection.name)
        print(f"    Vectors: {info.vectors_count}, Points: {info.points_count}")
    
    # Inspect memory_test_collection
    collection_name = "memory_test_collection"
    print(f"\n📋 Detailed inspection of '{collection_name}':")
    
    # Scroll through all points
    offset = None
    all_points = []
    
    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset
        )
        
        all_points.extend(result[0])
        offset = result[1]
        
        if offset is None:
            break
    
    print(f"\nTotal memories found: {len(all_points)}")
    
    # Group by user and type
    users = {}
    for point in all_points:
        user_id = point.payload.get("user_id", "unknown")
        mem_type = point.payload.get("memory_type", "unknown")
        
        if user_id not in users:
            users[user_id] = {}
        if mem_type not in users[user_id]:
            users[user_id][mem_type] = []
        
        users[user_id][mem_type].append({
            "id": point.id,
            "text": point.payload.get("text", ""),
            "timestamp": point.payload.get("timestamp", 0),
            "updated_at": point.payload.get("updated_at", None)
        })
    
    # Display organized memories
    for user_id, memory_types in users.items():
        print(f"\n👤 User: {user_id}")
        for mem_type, memories in memory_types.items():
            print(f"  📁 {mem_type} ({len(memories)} entries):")
            for mem in memories:
                updated = f" [UPDATED at {mem['updated_at']}]" if mem['updated_at'] else ""
                print(f"    - {mem['text'][:80]}...{updated}")
    
    return len(all_points)

def verify_neo4j_memories():
    """Inspect all memories and relationships in Neo4j"""
    print("\n" + "="*60)
    print("🔍 NEO4J MEMORY VERIFICATION")
    print("="*60)
    
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "password123")
    )
    
    with driver.session() as session:
        # Count all memory nodes
        result = session.run("""
            MATCH (m:MemoryTest)
            RETURN count(m) as total_memories
        """)
        total_memories = result.single()["total_memories"]
        print(f"\nTotal memory nodes: {total_memories}")
        
        # Count relationships
        result = session.run("""
            MATCH ()-[r:RELATED_TO]->()
            RETURN count(r) as total_relationships
        """)
        total_relationships = result.single()["total_relationships"]
        print(f"Total relationships: {total_relationships}")
        
        # Get memory graph structure
        print("\n🕸️ Memory Graph Structure:")
        result = session.run("""
            MATCH (m:MemoryTest)
            OPTIONAL MATCH (m)-[r:RELATED_TO]-(related:MemoryTest)
            WITH m, collect(DISTINCT {
                id: related.id,
                text: related.text,
                type: related.memory_type
            }) as related_memories
            RETURN m.id as id, m.text as text, m.memory_type as type,
                   m.updated_at as updated, related_memories
            ORDER BY m.timestamp
        """)
        
        for record in result:
            updated = " [UPDATED]" if record["updated"] else ""
            print(f"\n  📌 {record['type']}: {record['text'][:60]}...{updated}")
            print(f"     ID: {record['id']}")
            
            if record["related_memories"] and record["related_memories"][0]['id']:
                print("     Related to:")
                for related in record["related_memories"]:
                    if related['id']:  # Filter out null entries
                        print(f"       → [{related['type']}] {related['text'][:50]}...")
        
        # Verify specific updates
        print("\n🔄 Update Verification:")
        result = session.run("""
            MATCH (m:MemoryTest)
            WHERE m.updated_at IS NOT NULL
            RETURN m.text as text, m.updated_at as updated_at
        """)
        
        for record in result:
            print(f"  ✅ Updated memory: {record['text']}")
            print(f"     Updated at: {record['updated_at']}")
    
    driver.close()
    return total_memories

def verify_consistency():
    """Verify consistency between Qdrant and Neo4j"""
    print("\n" + "="*60)
    print("🔍 CONSISTENCY VERIFICATION")
    print("="*60)
    
    qdrant_count = verify_qdrant_memories()
    neo4j_count = verify_neo4j_memories()
    
    print("\n" + "="*60)
    print("📊 FINAL VERIFICATION SUMMARY")
    print("="*60)
    
    print(f"\n✅ Qdrant memories: {qdrant_count}")
    print(f"✅ Neo4j memories: {neo4j_count}")
    print(f"✅ Consistency: {'PASSED' if qdrant_count == neo4j_count else 'FAILED'}")
    
    if qdrant_count == neo4j_count:
        print("\n🎉 Memory integrity verified! All memories are intact and consistent.")
    else:
        print("\n⚠️ Warning: Memory count mismatch detected!")

if __name__ == "__main__":
    verify_consistency()