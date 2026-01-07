# Knowledge Layer Pipeline

A comprehensive RAG (Retrieval-Augmented Generation) pipeline for knowledge management, built with clean code principles and professional structure.

## Overview

This project provides a complete pipeline for processing, embedding, searching, and querying knowledge bases using modern NLP techniques.

## Infrastructure

The system architecture and deployment infrastructure:

![Infrastructure Diagram](docs/infrastructure-diagram.png)

### Deployment Architecture

- **API Server**: FastAPI application running on port 8000
- **Kubernetes**: Container orchestration with 2 replicas for high availability
- **Storage**: Persistent volumes for vector indexes and metrics
- **Configuration**: ConfigMap-based configuration management
- **Health Checks**: Liveness and readiness probes for reliability
- **Scaling**: Horizontal pod autoscaling support (2 replicas default)

### Container Details

- **Base Image**: Python 3.11-slim
- **Multi-stage Build**: Optimized for size and security
- **Non-root User**: Runs as `raguser` for security
- **Resource Limits**: 2-4GB RAM, 1-2 CPU cores per pod

> **Note**: For creating a Miro diagram, see [MIRO_DIAGRAM_SPEC.md](docs/MIRO_DIAGRAM_SPEC.md) for detailed specifications.

### Pipeline Steps

1. **Text Cleaning** - Normalize and clean raw text files
2. **Chunking** - Split text into manageable chunks (sentence-aware)
3. **Refinement** (Optional) - LLM-powered semantic chunk refinement
4. **Entity Extraction** - Extract named entities (people, locations, etc.)
5. **Embedding** - Generate vector embeddings for semantic search
6. **Search** - Vector similarity search with metadata filtering
7. **Context Building** - Create optimized context blocks for LLMs
8. **Question-Answering** - Generate answers using LLaMA/LLMs

## Project Structure

```
modell/
├── src/                          # Source code
│   ├── knowledge_layer/          # Core data structures
│   ├── processing/               # Text cleaning & chunking
│   ├── embeddings/               # Embedding generation
│   ├── search/                   # Vector search & context building
│   ├── llm/                      # LLM refinement & QA
│   ├── entities/                 # Entity extraction
│   ├── pipeline/                 # Pipeline orchestration
│   ├── ui/                       # Graphical user interfaces
│   └── utils/                    # Utilities
│
├── data/                         # Data directories
│   ├── raw/                      # Raw input files
│   ├── processed/                # Processed chunks
│   └── output/                   # Final outputs
│
├── docs/                         # Documentation
├── tests/                        # Unit tests
├── examples/                     # Example scripts
├── config/                       # Configuration files
│
├── main.py                       # Main entry point
├── chat_gui.py                   # Chat GUI launcher
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── README_CHAT_GUI.md            # Chat GUI documentation
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd modell
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Optional: Install additional dependencies**
   - For entity extraction: `pip install spacy && python -m spacy download en_core_web_sm`
   - For LLM features: Already included in requirements.txt

## Quick Start

### Basic Pipeline

```bash
# Run complete pipeline
python main.py --run-all

# Run specific steps
python main.py --chunk --extract-entities --embed
```

### Search & Answer

```bash
# Search only
python main.py --search --query "What is this about?" --top-k 10

# Search with context building
python main.py --search --query "What is this about?" --build-context

# Full RAG: Search + Context + Answer
python main.py \
    --search \
    --query "What is this about?" \
    --top-k 10 \
    --build-context \
    --answer \
    --llm-provider ollama \
    --llm-model llama3
```

### Chat GUI (Graphical Interface)

For a user-friendly graphical interface, use the chat application:

```bash
# Launch the chat GUI
python chat_gui.py
```

The GUI provides:
- Simple chat interface for asking questions
- Real-time answers from your knowledge base
- Background processing with status updates
- Error handling and recovery

See [README_CHAT_GUI.md](README_CHAT_GUI.md) for detailed usage instructions.

## Usage Examples

### 1. Process Text Files

```bash
# Clean text files
python -m src.processing.cleaning --input-dir data/raw --output-dir data/processed

# Chunk text
python -m src.processing.chunking --input-dir data/processed --output-dir data/output/chunks
```

### 2. Generate Embeddings

```bash
python -m src.embeddings.generator \
    --input-file data/output/chunks/chunks.jsonl \
    --output-file data/output/embedded.jsonl \
    --model all-MiniLM-L6-v2
```

### 3. Search Knowledge Base

```bash
python -m src.search.searcher \
    --embeddings-file data/output/embedded.jsonl \
    --query "Your question here" \
    --top-k 5 \
    --build-context
```

### 4. Generate Answers

```bash
python -m src.llm.qa \
    --question "Your question" \
    --context "Context text here" \
    --provider ollama \
    --model llama3
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# LLM API Keys (if using OpenAI/Anthropic)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here

# Ollama Configuration (default)
OLLAMA_BASE_URL=http://localhost:11434
```

### Default Settings

- **Chunk Size**: 400 tokens
- **Overlap**: 60 tokens
- **Embedding Model**: all-MiniLM-L6-v2
- **LLM Provider**: ollama
- **LLM Model**: llama3

## Features

### Text Processing
- ✅ Sentence-aware chunking
- ✅ Token-based chunking with overlap
- ✅ Text cleaning and normalization

### Embeddings
- ✅ Multiple embedding models supported
- ✅ Batch processing
- ✅ Deterministic embeddings

### Search
- ✅ Cosine and dot-product similarity
- ✅ Metadata filtering
- ✅ Persistent vector store
- ✅ Top-k retrieval

### LLM Integration
- ✅ Ollama (local models)
- ✅ OpenAI API
- ✅ Anthropic Claude API
- ✅ Custom OpenAI-compatible APIs

### Context Building
- ✅ Token limit enforcement
- ✅ Redundancy reduction
- ✅ Flexible ordering strategies

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_processing.py
```

### Code Structure

The codebase follows clean code principles:

- **Separation of Concerns**: Each module has a single responsibility
- **DRY**: No code duplication
- **SOLID Principles**: Interfaces and abstractions where appropriate
- **Type Hints**: Full type annotations
- **Documentation**: Comprehensive docstrings

### Adding New Features

1. Create module in appropriate `src/` subdirectory
2. Add `__init__.py` with exports
3. Update imports in dependent modules
4. Add tests in `tests/`
5. Update documentation

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependency list

### Core Dependencies
- `sentence-transformers` - Embeddings
- `numpy` - Numerical operations
- `tiktoken` - Token counting
- `langchain` - LLM integration
- `python-dotenv` - Environment variables


## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Support

[Add support information here]
