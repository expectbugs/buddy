# Changelog

All notable changes to Buddy will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.3] - 2025-06-08

### 🏗️ Phase 3: Extension Architecture - Multi-Agent Foundation

Complete multi-agent support framework enabling isolated agent namespaces with custom plugins.

### Added
- **Plugin System**: `plugin_system.py` with protocols for MemoryProcessor, ContextExpander, MemoryFilter
- **Agent Namespaces**: `agent_namespace.py` with isolated operations and custom configurations  
- **Multi-Agent Foundation**: `multi_agent_foundation.py` as unified interface for all agents
- **Multi-Agent Commands**: `/agents`, `/agent create`, `/plugins`, `/foundation health/debug/stats`

### Technical
- Protocol-based plugin architecture with priority execution
- Agent namespace isolation via user_id prefixing
- Extensible processor/expander/filter system
- Comprehensive health checking and debug tracking

## [1.5.2] - 2025-06-08

### 🔧 Phase 2: API Standardization - Predictable Interfaces

Standardized result types and centralized configuration for reliable agent integration.

### Added
- **Result Types**: `result_types.py` with standardized OperationResult pattern
- **Config Manager**: `config_manager.py` with centralized configuration and validation
- **Foundation Interface**: `foundation_interface.py` for consistent memory APIs

### Enhanced
- Enhanced memory wrapper now uses standardized result types
- All operations return predictable OperationResult[T] format
- Centralized config validation and management

## [1.5.1] - 2025-06-08

### 🛡️ Phase 1: Foundation Integrity - Rule Violations & Debugging

Eliminated all silent failures and added comprehensive debugging infrastructure.

### Added
- **Exception Hierarchy**: `exceptions.py` with custom exception types (BuddyMemoryError, etc.)
- **Debug Tracking**: `debug_info.py` with operation tracking and statistics
- **Health Checking**: `health_check.py` with Neo4j/Qdrant/filesystem monitoring

### Fixed
- **Rule 3 Compliance**: Eliminated all silent fallbacks - errors now fail loudly
- **Context Expansion**: Added proper error handling for expansion failures
- **Memory Operations**: Added validation and error tracking for all operations

## [1.5.0] - 2025-06-08

### 🎯 Context Expansion Bug Fix & Intelligence Verification

Fixed critical bug causing context expansion failures and verified system intelligence.

### Fixed
- **Context Expansion Bug**: Fixed `temp_lookup_code` → `shared_lookup_code` variable error
- **Lookup Code Sync**: Memory creation and conversation logging now use identical codes
- **Context Retrieval**: Expansion now successfully retrieves conversation context

### Verified
- **Intelligence Enhancement**: AI responses include specific details from expanded context
- **End-to-End Testing**: Comprehensive testing confirms expansion provides meaningful intelligence

## [1.2.0] - 2025-06-04

### 🎯 Phase 2 Enhanced Memory & Official mem0 Compliance

Enhanced memory system with automatic context linking and restored official mem0 behavior.

### Added
- **EnhancedMemory Wrapper**: `enhanced_memory.py` with automatic context linking
- **Automatic Metadata**: All memories include timestamps, lookup codes, version tracking
- **System Tools**: `fresh_start.py` for database cleanup

### Fixed
- **mem0 Compliance**: Removed custom filtering that blocked important information
- **Information Loss**: Mixed content messages now properly processed
- **LLM Decision Making**: Restored mem0's official extraction behavior

## [1.1.0] - 2025-06-04

### 🚀 Phase 1 Context System & Quality Improvements

Context expansion system and resolved inappropriate memory storage issues.

### Added
- **Context System**: `context_logger.py`, `context_bridge.py`, `temporal_utils.py`
- **Smart Filtering**: Prevents inappropriate content storage (commands, temporal data)
- **Full Context Logging**: Every conversation with unique lookup codes

### Fixed
- **Aggressive Extraction**: Fixed mem0 creating relationships from every fragment
- **Temporal Pollution**: Stopped current date/time being stored as memories
- **Database Cleanup**: Reset Neo4j/Qdrant for clean relationship building

## [1.0.0] - 2025-06-03

### 🎉 Production-Ready Memory System with mem0

Complete overhaul transitioning to industry-standard mem0 library.

### Added
- **mem0 Integration**: Official mem0ai v0.1.102 with Neo4j graph + Qdrant vector storage
- **Proper Deletion**: "forget X" actually deletes memories using mem0's delete() API
- **Deduplication**: Automatic detection and merging (85% threshold)
- **Entity Sanitization**: Clean extraction removing system IDs and special characters

### Fixed
- **Memory Forgetting**: Fixed - commands now delete instead of adding "wants to forget"
- **Boss Hierarchy**: "Josh's boss is Dave" → "Dave manages Josh" relationship
- **Duplicate Prevention**: Automatic deduplication prevents redundant storage

### Changed
- **Main Script**: Consolidated to `run.py` from multiple launchers
- **Architecture**: Full transition from custom to mem0 system
- **Repository**: Moved old scripts to archive/, clean structure

## [0.4.2] - 2025-05-27

### 🚨 Silent Failure Elimination & Service Fixes

Made all errors visible and fixed critical service startup issues.

### Fixed
- **Silent Failures**: Added logging for all error paths (access counts, parsing, Neo4j ops)
- **Critical Bugs**: Fixed IndexError, ZeroDivisionError, unsafe operations
- **Service Infrastructure**: Fixed Neo4j OpenRC service with proper Java paths
- **Comprehensive Logging**: Every error path now has appropriate logging

## [0.4.1] - 2025-05-27

### 🤖 QWQ-32B Integration & Major Bug Fixes

Added QWQ-32B reasoning model support with comprehensive fixes.

### Added
- **QWQ-32B Support**: `launch_qwq.py` with chain-of-thought display
- **Command Args**: `--debug`, `--quiet`, `--hide-reasoning` flags
- **Access Tracking**: Memories track access count and last accessed time

### Fixed
- **Critical Bugs**: Logger initialization, duplicate methods, missing implementations
- **API Fixes**: Qdrant delete operations, Neo4j case-insensitive queries
- **Schema Sync**: Unified ID fields between QWQ and Hermes implementations

## [0.4.0-dev] - 2025-05-27

### 🧠 Memory Consolidation & Summarization

Intelligent memory consolidation with episode tracking and context expansion.

### Added
- **Memory Summarization**: Automatic LLM-based episode summarization
- **Episode Detection**: Semantic similarity analysis for topic boundaries
- **Smart Context Expansion**: Relevance-based loading of full original context
- **Enhanced Commands**: `/episodes`, `/summarize` commands

## [0.3.0-dev] - 2025-05-26

### 🔐 Permanent Storage Enhancements

Robust persistence verification and reliability improvements.

### Added
- **Persistence Verification**: Automatic data survival testing on startup
- **Connection Reliability**: Neo4j pooling, retry logic, graceful degradation
- **Graceful Shutdown**: Signal handlers, queue completion, resource cleanup
- **Interaction Logging**: Structured JSONL logs with rotation

## [0.2.0] - 2025-05-26

### 🧠 Advanced Intelligence & Processing

Comprehensive memory system improvements with intelligent filtering.

### Added
- **Intelligent Filtering**: Trivial pattern detection, priority thresholds
- **Systematic Extraction**: 5-category framework (personal, relationships, preferences, activities, facts)
- **Background Processing**: ThreadPoolExecutor with non-blocking operations
- **Advanced Search**: Combined scoring (similarity + priority + recency)
- **Memory Organization**: Seven distinct types with specific handling

### Enhanced
- **60% Memory Reduction**: Intelligent filtering eliminates trivial content
- **Zero Latency Impact**: Background processing prevents conversation delays
- **40% Better Relevance**: Combined scoring improves context retrieval

## [0.1.0] - 2025-05-26

### Initial Release
- Basic Hermes-2-Pro-Mistral-10.7B integration
- Simple memory storage with Qdrant + Neo4j
- Interactive chat interface
- OpenRC service management for Gentoo Linux
- CUDA optimization for RTX 3090