# Contributing to Hermes-Mem0

Thank you for your interest in contributing to the Hermes-Mem0 project! This guide will help you get started.

## Code of Conduct

Please be respectful and constructive in all interactions. We want this to be a welcoming community for everyone.

## How to Contribute

### Reporting Issues

1. Check if the issue already exists in the [Issues](https://github.com/your-username/hermes-mem0/issues) section
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (if it's a bug)
   - Your system information (OS, GPU, Python version)
   - Relevant error messages or logs

### Submitting Pull Requests

1. Fork the repository
2. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes:
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation as needed
4. Test your changes:
   ```bash
   # Run the main script
   python launch_hermes_fixed.py
   
   # Test examples
   cd examples
   python simple_mem0_test.py
   ```
5. Commit your changes:
   ```bash
   git commit -m "Add feature: brief description"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Create a Pull Request with:
   - Clear description of changes
   - Reference to any related issues
   - Screenshots/examples if applicable

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/hermes-mem0.git
   cd hermes-mem0
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up services:
   ```bash
   ./install_mem0_dependencies.sh
   ./setup_llama_cpp.sh
   ```

4. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

## Code Style Guidelines

- Use Python 3.8+ features appropriately
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Handle errors gracefully with try/except blocks

## Testing

- Test your changes with different models if possible
- Verify memory persistence works correctly
- Check service connections (Qdrant, Neo4j)
- Test error handling scenarios

## Areas for Contribution

### High Priority
- Performance optimizations
- Memory management improvements
- Better error handling and recovery
- Documentation improvements
- Multi-user support

### Feature Ideas
- Web UI for memory management
- Memory export/import functionality
- Advanced memory search capabilities
- Integration with other LLM frameworks
- Memory compression algorithms
- Automated memory summarization

### Documentation
- Tutorials for specific use cases
- Video guides
- API documentation
- Architecture diagrams

## Questions?

Feel free to:
- Open an issue for discussion
- Join our community chat (if available)
- Reach out to maintainers

Thank you for contributing!