# Buddy v0.2.0 - Advanced AI Assistant with Intelligent Memory System

Buddy is an advanced AI assistant built on Hermes-2-Pro-Mistral-10.7B with a sophisticated Phase 1 memory system featuring intelligent filtering, background processing, and comprehensive fact extraction. Optimized for NVIDIA RTX 3090 on Gentoo Linux with OpenRC.

## Features

### 🧠 **Phase 1 Advanced Memory System**
- **Intelligent Memory Filtering**: Automatically filters out trivial information (greetings, small talk)
- **Priority-Based Storage**: Smart classification with priority scoring (0.0-1.0)
- **Comprehensive Fact Extraction**: Systematically extracts personal facts, relationships, and preferences
- **Background Memory Processing**: Non-blocking memory operations using ThreadPoolExecutor
- **Advanced Search & Ranking**: Combined scoring (similarity + priority + recency)
- **Memory Type Classification**: Organized by personal_fact, relationship, preference, project, technical

### 🚀 **System Architecture**
- **Gentoo/OpenRC Native**: Optimized for Gentoo Linux with OpenRC service management
- **No Containers**: All components installed directly via Portage and manual installation
- **Hybrid Memory System**: Combines vector search (Qdrant) and graph relationships (Neo4j)
- **RTX 3090 Optimized**: Configured specifically for 24GB VRAM with optimal performance settings
- **Q6_K Quantization**: High-quality quantization for near-perfect results
- **Persistent Memory**: Conversations and context are stored across sessions with integrity

## System Requirements

- **Gentoo Linux** with OpenRC (systemd not supported)
- NVIDIA RTX 3090 GPU (24GB VRAM) with CUDA support
- Java 11+ (OpenJDK from Portage)
- Python 3.11+ (from Portage)
- At least 32GB RAM
- NVMe SSD recommended
- Root/sudo access for service installation
- Portage package manager (emerge)

## Installation

### 1. Install All Dependencies and Services Natively

```bash
./install_mem0_dependencies.sh
```

This will:
- Sync Portage tree and install system dependencies via emerge
- Install mem0ai and Python packages via pip
- Download and install Qdrant natively with OpenRC init script
- Download and install Neo4j Community Edition with APOC plugin
- Create proper OpenRC init scripts for both services
- Add services to default runlevel for automatic startup

### 2. Set up llama.cpp

```bash
./setup_llama_cpp.sh
```

This will:
- Clone and build llama.cpp with CUDA support
- Create necessary directories
- Set up Python bindings installer

### 3. Install Python bindings (optional but recommended)

```bash
./install_llama_cpp_python.sh
```

### 4. Download the Model

Download the Hermes-2-Pro-Mistral-10.7B-Q6_K GGUF model and place it in:
```
~/models/Hermes-2-Pro-Mistral-10.7B-Q6_K/hermes-2-pro-mistral-10.7b.Q6_K.gguf
```

## Configuration

### Environment Variables (.env)

- `OPENAI_API_KEY`: Optional, for OpenAI embeddings (falls back to local if not set)
- `NEO4J_USERNAME/PASSWORD`: Neo4j credentials (default: neo4j/password123)
- `QDRANT_HOST/PORT`: Qdrant connection settings
- `CUDA_VISIBLE_DEVICES`: GPU selection

### Memory Configuration (mem0_config.yaml)

Customize:
- Model path and parameters
- Vector store settings
- Graph store configuration
- Embedder selection
- Performance tuning

## Usage

### Launch Buddy

```bash
python3 launch_hermes_improved.py
```

### Interactive Commands

- `/memory` - View stored memories organized by type
- `/stats` - Display memory statistics and analytics
- `/clear` - Clear all stored memories
- `/exit` - End the session

### Advanced Memory Features

Buddy automatically:
- **Extracts Facts**: Names, occupations, relationships, preferences
- **Filters Trivially**: Ignores greetings, small talk, typos
- **Prioritizes Information**: Higher scores for important facts
- **Processes in Background**: Never blocks conversation flow
- **Searches Intelligently**: Combines similarity, priority, and recency
- **Organizes by Type**: Personal facts, relationships, preferences, projects, technical details

## Performance Settings

Optimized for RTX 3090:
- **GPU Layers**: 99 (all layers offloaded)
- **Context Size**: 8192 tokens
- **Batch Size**: 512
- **Threads**: 8 CPU threads
- **Expected Performance**: ~40-50 tokens/second

## Architecture

### Hybrid Memory System

1. **Vector Store (Qdrant)**
   - Semantic search capabilities
   - Fast similarity matching
   - Scalable to millions of embeddings

2. **Graph Store (Neo4j)**
   - Relationship modeling
   - Complex query capabilities
   - Context understanding through connections

3. **Integration**
   - Mem0 orchestrates both stores
   - Intelligent memory distribution
   - Enhanced retrieval accuracy

## Service Management (OpenRC)

### Native Services

- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Neo4j Browser**: http://localhost:7474 (neo4j/password123)

### OpenRC Service Commands

```bash
# Manage services easily with interactive menu
./manage_services.sh

# Manual OpenRC service control
sudo rc-service qdrant start
sudo rc-service qdrant stop
sudo rc-service qdrant restart
sudo rc-service qdrant status

sudo rc-service neo4j start
sudo rc-service neo4j stop
sudo rc-service neo4j restart
sudo rc-service neo4j status

# Runlevel management (auto-start on boot)
sudo rc-update add qdrant default
sudo rc-update add neo4j default
sudo rc-update del qdrant default
sudo rc-update del neo4j default

# Check what's in runlevels
rc-update show
rc-update show default
```

### Service Locations

- **Qdrant**: `~/qdrant/` (binary and data)
- **Neo4j**: `~/neo4j/neo4j-community/` (installation and data)
- **OpenRC Scripts**: `/etc/init.d/qdrant` and `/etc/init.d/neo4j`

### Additional OpenRC Commands

```bash
# Check current runlevel
rc-status --runlevel

# View all running services
rc-status

# View services in specific runlevel
rc-status default
```

## Troubleshooting

### Model Not Found
Ensure the model is in the correct path as specified in `mem0_config.yaml`

### CUDA Errors
- Verify CUDA is installed: `nvidia-smi`
- Check CUDA version compatibility
- Ensure llama.cpp was built with `LLAMA_CUBLAS=1`

### Service Issues (OpenRC)
- Check service status: `sudo rc-service qdrant status` or `sudo rc-service neo4j status`
- View logs: Check `~/qdrant/` and `~/neo4j/neo4j-community/logs/` directories
- Use service manager: `./manage_services.sh`
- Test connections: `./verify_installation.sh`
- Check runlevel: `rc-update show default`

### Gentoo-Specific Issues
- **Portage sync failed**: Run `sudo emerge --sync` manually
- **Emerge conflicts**: Use `emerge --ask --verbose --update --deep --newuse @world`
- **Missing USE flags**: Add required flags to `/etc/portage/make.conf`
- **Java not found**: Ensure `virtual/jdk:11` is emerged and selected with `eselect java list`

### Port Conflicts
If ports 6333, 7474, or 7687 are in use:
- Check what's using the port: `sudo netstat -tlnp | grep 6333`
- Stop conflicting services: `sudo rc-service <service> stop`
- Edit service configs in `~/qdrant/config.yaml` or `~/neo4j/neo4j-community/conf/neo4j.conf`

### Performance Issues
- Monitor GPU usage: `nvidia-smi -l 1`
- Adjust batch size and context in config
- Ensure model fits in VRAM

## Advanced Usage

### Custom Embeddings
Edit `mem0_config.yaml` to use different embedding models:
- OpenAI embeddings (requires API key)
- Local HuggingFace models
- Custom embedding providers

### Memory Management
The system automatically:
- Stores user interactions
- Builds relationship graphs
- Retrieves relevant context
- Improves over time

## Resources

- [Mem0 Documentation](https://docs.mem0.ai/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [llama.cpp Repository](https://github.com/ggerganov/llama.cpp)