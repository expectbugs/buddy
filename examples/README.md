# Hermes-Mem0 Examples

This directory contains various examples and test scripts for the Hermes-Mem0 system.

## Examples Overview

### Basic Tests
- `simple_mem0_test.py` - Basic connectivity test for Qdrant and Neo4j
- `test_mem0_system.py` - Test the full Mem0 memory system
- `test_complete_system.py` - Comprehensive system test

### Memory Demonstrations
- `demo_memory_persistence.py` - Shows memory persistence across sessions
- `test_all_memory_types.py` - Tests different memory storage types
- `verify_memory_integrity.py` - Verifies memory storage and retrieval

### Interactive Examples
- `interactive_test.py` - Interactive testing interface
- `test_real_usage.py` - Real-world usage scenarios
- `final_system_demo.py` - Complete system demonstration
- `final_verification.py` - Final system verification

## Running Examples

Make sure services are running:
```bash
sudo rc-service qdrant start
sudo rc-service neo4j start
```

Then run any example:
```bash
python examples/simple_mem0_test.py
```

## What Each Example Does

### simple_mem0_test.py
Tests basic connectivity to Qdrant and Neo4j without the full Mem0 system.

### demo_memory_persistence.py
Demonstrates how memories persist across different sessions, showing the system's ability to recall previous conversations.

### test_all_memory_types.py
Tests various memory operations including:
- Adding memories
- Searching memories
- Memory relationships
- Memory metadata

### interactive_test.py
Provides an interactive interface to test different memory operations manually.

## Creating Your Own Examples

When creating new examples:
1. Import required modules
2. Load environment variables with `load_dotenv()`
3. Initialize services
4. Demonstrate specific functionality
5. Clean up resources

Example template:
```python
#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from mem0 import Memory

# Load environment variables
load_dotenv()

def main():
    # Your example code here
    pass

if __name__ == "__main__":
    main()
```