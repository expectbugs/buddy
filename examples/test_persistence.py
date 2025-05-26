#!/usr/bin/env python3

import subprocess
import time
import sys
from qdrant_client import QdrantClient
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_memory_counts():
    """Check current memory counts in both stores"""
    # Check Qdrant
    qdrant = QdrantClient(host="localhost", port=6333)
    try:
        collection_info = qdrant.get_collection("hermes_local_memory")
        qdrant_count = collection_info.points_count
    except:
        qdrant_count = 0
    
    # Check Neo4j
    neo4j_username = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=(neo4j_username, neo4j_password))
    
    with driver.session() as session:
        result = session.run("MATCH (n:HermesMemory) RETURN count(n) as count")
        neo4j_count = result.single()["count"]
    
    driver.close()
    
    return qdrant_count, neo4j_count

def test_persistence():
    print("=== Testing Memory Persistence ===")
    
    # Check initial state
    qdrant_before, neo4j_before = check_memory_counts()
    print(f"Initial state - Qdrant: {qdrant_before}, Neo4j: {neo4j_before}")
    
    # First session - create some memories
    print("\n--- Session 1: Creating memories ---")
    proc = subprocess.Popen(
        ['python3', 'launch_hermes_fixed.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send test messages
    test_messages = [
        "My favorite color is blue",
        "I have a cat named Whiskers",
        "I'm learning Python programming",
        "/memory",
        "/exit"
    ]
    
    for msg in test_messages:
        print(f"Sending: {msg}")
        proc.stdin.write(msg + "\n")
        proc.stdin.flush()
        time.sleep(2)  # Give time for processing
    
    # Wait for process to end
    proc.communicate()
    
    # Check after first session
    qdrant_after1, neo4j_after1 = check_memory_counts()
    print(f"\nAfter session 1 - Qdrant: {qdrant_after1}, Neo4j: {neo4j_after1}")
    
    if qdrant_after1 > qdrant_before:
        print("✅ Memories were stored in Qdrant")
    else:
        print("❌ No memories stored in Qdrant")
    
    if neo4j_after1 > neo4j_before:
        print("✅ Memories were stored in Neo4j")
    else:
        print("❌ No memories stored in Neo4j")
    
    # Wait before second session
    print("\n--- Waiting 5 seconds before next session ---")
    time.sleep(5)
    
    # Second session - check if memories persist
    print("\n--- Session 2: Checking persistence ---")
    proc2 = subprocess.Popen(
        ['python3', 'launch_hermes_fixed.py'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Just check memories and exit
    proc2.stdin.write("/memory\n")
    proc2.stdin.flush()
    time.sleep(2)
    
    proc2.stdin.write("/exit\n")
    proc2.stdin.flush()
    
    output, _ = proc2.communicate()
    
    # Check final state
    qdrant_after2, neo4j_after2 = check_memory_counts()
    print(f"\nAfter session 2 - Qdrant: {qdrant_after2}, Neo4j: {neo4j_after2}")
    
    # Print session 2 output
    print("\n--- Session 2 Output ---")
    if "My favorite color is blue" in output or "Whiskers" in output:
        print("✅ Memories persisted! Found previous conversation in output.")
    else:
        print("❌ Memories did not persist.")
    
    # Final verdict
    if qdrant_after2 >= qdrant_after1 and neo4j_after2 >= neo4j_after1:
        print("\n✅ SUCCESS: Memory persistence is working!")
    else:
        print("\n❌ FAILED: Memories were lost between sessions")

if __name__ == "__main__":
    test_persistence()