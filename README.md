# Closed-Loop RAG System

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Coverage](https://img.shields.io/badge/Coverage-%3E80%25-brightgreen.svg)

## Overview

A production-ready RAG (Retrieval-Augmented Generation) system with automated evaluation using TruLens. This system enables you to ingest documents, query them with semantic search, and evaluate the quality of responses using heuristic-based metrics.

## Tech Stack

- **RAG Framework**: LlamaIndex
- **Vector Database**: Milvus (HNSW index)
- **LLM**: Ollama (Llama 3 8B)
- **Embedding Model**: nomic-embed-text (Ollama)
- **Evaluation**: SimulatedEvaluator (heuristic-based metrics)
- **API**: FastAPI
- **Deployment**: Docker Compose

## Features

- **Multi-format Document Ingestion**: Supports PDF, Word, Excel, PowerPoint, HTML, CSV, and more via MarkItDown
- **Semantic Search**: Vector-based retrieval with configurable top_k
- **Automated Evaluation**: Faithfulness, Relevance, Context Precision, Context Recall metrics
- **Docker-based Deployment**: All services orchestrated with Docker Compose
- **RESTful API**: Simple `/ingest` and `/query` endpoints
- **Full Test Coverage**: >80% test coverage with pytest

## Quick Start

1. **Clone and start services**:
   ```bash
   docker-compose up -d
   ```

2. **Wait for services to be ready** (approximately 20 seconds):
   ```bash
   sleep 20
   ```

3. **Download Ollama models**:
   ```bash
   docker-compose exec ollama ollama pull llama3:8b nomic-embed-text
   ```

4. **Ingest a document**:
   ```bash
   curl -X POST http://localhost:8000/ingest \
     -F "file=@sample.pdf"
   ```

5. **Query the system**:
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What is RAG?"}'
   ```

6. **View the TruLens dashboard**:
   ```
   http://localhost:8501
   ```

## Clone and Run from GitHub

Get started quickly by cloning the repository from GitHub:

```bash
# Clone the repository
git clone https://github.com/zabwie/Closed-Loop-RAG-System
cd rag-system

# Start all services
docker-compose up -d

# Wait for services to initialize (approximately 20 seconds)
sleep 20

# Download required Ollama models
docker-compose exec ollama ollama pull llama3:8b nomic-embed-text

# Verify services are running
docker-compose ps
```

Once services are running, you can ingest documents and query the system using the API endpoints documented below.

> **Note**: For detailed deployment instructions, including GitHub repository setup, authentication, and advanced configuration, see [GITHUB_DEPLOYMENT.md](GITHUB_DEPLOYMENT.md).

## Setup Instructions

### Prerequisites

- **Docker** and **Docker Compose** installed
- **Python 3.11+** (for local development)
- **8GB+ VRAM** (for running Llama 3 8B locally)
- **10GB+ disk space** (for Docker images and data)

### Docker Setup

1. **Clone the repository**:
   ```bash
   # Clone via HTTPS
   git clone https://github.com/zabwie/Closed-Loop-RAG-System

   # Or clone via SSH
   git clone git@github.com:zabwie/Closed-Loop-RAG-System

   # Navigate into the project directory
   cd rag-system
   ```

2. **Configure environment variables** (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your custom settings
   ```

3. **Start all services**:
   ```bash
   docker-compose up -d
   ```

4. **Verify services are running**:
   ```bash
   docker-compose ps
   ```

5. **Download Ollama models**:
   ```bash
   docker-compose exec ollama ollama pull llama3:8b nomic-embed-text
   ```

### Local Development Setup

1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Set PYTHONPATH**:
   ```bash
   export PYTHONPATH=$(pwd)/src  # Linux/Mac
   set PYTHONPATH=Z:\Gemini\RAG\src  # Windows
   ```

3. **Run tests**:
   ```bash
   pytest tests/ -v --cov=src/rag_system
   ```

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                         FastAPI                             │
│  ┌──────────────┐              ┌──────────────┐            │
│  │  /ingest     │              │   /query     │            │
│  └──────┬───────┘              └──────┬───────┘            │
└─────────┼──────────────────────────────┼───────────────────┘
          │                              │
          ▼                              ▼
┌──────────────────────┐    ┌──────────────────────────────┐
│  Ingestion Pipeline  │    │      RAG Pipeline            │
│  ┌────────────────┐  │    │  ┌────────────────────────┐  │
│  │ MarkItDown     │  │    │  │ EmbeddingService       │  │
│  │ Converter      │  │    │  │ (nomic-embed-text)     │  │
│  └────────┬───────┘  │    │  └───────────┬────────────┘  │
│           │          │    │              │               │
│           ▼          │    │              ▼               │
│  ┌────────────────┐  │    │  ┌────────────────────────┐  │
│  │ TextChunker    │  │    │  │ MilvusVectorStore      │  │
│  │ (512 tokens)   │  │    │  │ (HNSW index)           │  │
│  └────────┬───────┘  │    │  └───────────┬────────────┘  │
│           │          │    │              │               │
│           ▼          │    │              ▼               │
│  ┌────────────────┐  │    │  ┌────────────────────────┐  │
│  │ Embedding      │  │    │  │ OllamaClient           │  │
│  │ Service        │  │    │  │ (llama3:8b)            │  │
│  └────────┬───────┘  │    │  └───────────┬────────────┘  │
│           │          │    │              │               │
│           ▼          │    │              ▼               │
│  ┌────────────────┐  │    │  ┌────────────────────────┐  │
│  │ Milvus         │  │    │  │ SimulatedEvaluator     │  │
│  │ VectorStore    │  │    │  │ (RAG Triad Metrics)    │  │
│  └────────────────┘  │    │  └────────────────────────┘  │
└──────────────────────┘    └──────────────────────────────┘
```

### Data Flow

**Ingestion Pipeline**:
1. Document upload → MarkItDown conversion (extracts text and metadata)
2. Text chunking → Fixed-size chunks (512 tokens, 50 overlap)
3. Embedding generation → nomic-embed-text model
4. Milvus storage → Vector indexing with HNSW

**Query Pipeline**:
1. Query embedding → nomic-embed-text model
2. Similarity search → Milvus HNSW index (top_k results)
3. Context assembly → Join retrieved documents
4. LLM generation → llama3:8b with RAG prompt
5. Response evaluation → SimulatedEvaluator metrics

### Services

| Service | Image | Ports | Purpose |
|---------|-------|-------|---------|
| etcd | quay.io/coreos/etcd:v3.5.5 | 2379 | Metadata storage for Milvus |
| minio | minio/minio:RELEASE.2023-03-20T20-16-18Z | 9000 | Object storage for Milvus |
| milvus-standalone | milvusdb/milvus:v2.3.3 | 19530, 9091 | Vector database |
| ollama | ollama/ollama:latest | 11434 | LLM and embedding service |
| trulens | ghcr.io/truera/trulens:latest | 8501 | Evaluation dashboard |
| rag-app | python:3.11-slim | 8000 | FastAPI application |

## API Documentation

### POST /ingest

Upload and ingest a document into the RAG system.

**Request**:
```http
POST /ingest
Content-Type: multipart/form-data

file: <binary file data>
```

**Response**:
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "chunk_count": 42,
  "source": "sample.pdf",
  "error": null
}
```

**Supported Formats**: PDF, Word (.docx), Excel (.xlsx), PowerPoint (.pptx), HTML, CSV, Markdown, and more.

### POST /query

Query the RAG system with evaluation metrics.

**Request**:
```http
POST /query
Content-Type: application/json

{
  "query": "What is RAG?",
  "top_k": 5
}
```

**Response**:
```json
{
  "query": "What is RAG?",
  "answer": "RAG (Retrieval-Augmented Generation) is a technique that combines...",
  "sources": [
    {
      "text": "RAG is a technique that enhances LLM responses...",
      "score": 0.85,
      "metadata": {
        "source": "sample.pdf",
        "chunk_index": 3
      }
    }
  ],
  "retrieved_count": 5,
  "evaluation": {
    "faithfulness": 0.75,
    "relevance": 0.80,
    "context_precision": 0.70,
    "context_recall": 0.60,
    "overall_score": 0.72
  }
}
```

**Parameters**:
- `query` (string, required): The question to ask
- `top_k` (integer, optional): Number of documents to retrieve (default: 5, range: 1-20)

**Evaluation Metrics**:
- **Faithfulness**: Word overlap between answer and sources (0-1)
- **Relevance**: Word overlap between query and answer (0-1)
- **Context Precision**: Average source score (0-1)
- **Context Recall**: Retrieved count / ideal count (0-1)
- **Overall Score**: Weighted average (faithfulness 30%, relevance 30%, precision 20%, recall 20%)

### GET /health

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "services": {
    "ollama": "connected",
    "milvus": "connected"
  }
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | http://localhost:11434 | Ollama API endpoint |
| `MILVUS_HOST` | localhost | Milvus host |
| `MILVUS_PORT` | 19530 | Milvus port |
| `MODEL_NAME` | llama3:8b | LLM model name |
| `EMBEDDING_MODEL` | nomic-embed-text | Embedding model name |
| `COLLECTION_NAME` | documents | Milvus collection name |
| `CHUNK_SIZE` | 512 | Chunk size in tokens |
| `CHUNK_OVERLAP` | 50 | Chunk overlap in tokens |
| `TOP_K` | 5 | Default number of results to retrieve |

### Milvus Configuration

- **Index Type**: HNSW (Hierarchical Navigable Small World)
- **Metric Type**: IP (Inner Product)
- **Dimension**: 768 (nomic-embed-text embedding dimension)

## Troubleshooting

### Services not starting

**Problem**: Docker services fail to start or crash immediately.

**Solution**:
```bash
# Check service status
docker-compose ps

# Check service logs
docker-compose logs [service-name]

# Restart services
docker-compose restart

# Rebuild and restart
docker-compose up -d --build
```

### Ollama model not found

**Problem**: Error message "model 'llama3:8b' not found".

**Solution**:
```bash
# Pull required models
docker-compose exec ollama ollama pull llama3:8b nomic-embed-text

# Verify models are installed
docker-compose exec ollama ollama list
```

### Milvus connection errors

**Problem**: Error connecting to Milvus database.

**Solution**:
```bash
# Check Milvus logs
docker-compose logs milvus-standalone

# Verify etcd and minio are running
docker-compose ps etcd minio

# Restart Milvus
docker-compose restart milvus-standalone

# Check Milvus health
docker-compose exec milvus-standalone curl http://localhost:9091/healthz
```

### Evaluation metrics not showing

**Problem**: Evaluation scores are all zero or missing.

**Solution**:
```bash
# Check TruLens logs
docker-compose logs trulens

# Verify database file exists
ls -la trulens.db

# Restart TruLens service
docker-compose restart trulens
```

### Out of memory errors

**Problem**: Ollama crashes with OOM (Out of Memory) errors.

**Solution**:
```bash
# Check available memory
docker stats

# Reduce Ollama memory limit in docker-compose.yml
# Add: deploy: resources: limits: memory: 8g

# Use a smaller model
docker-compose exec ollama ollama pull llama3:8b-instruct-q4_0
```

### Slow query performance

**Problem**: Queries take more than 10 seconds to complete.

**Solution**:
```bash
# Check Milvus index status
docker-compose exec milvus-standalone python -c "from pymilvus import utility; print(utility.index_info('documents'))"

# Reduce top_k parameter in query
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "...", "top_k": 3}'

# Increase Milvus resources in docker-compose.yml
```

### Port conflicts

**Problem**: Error "port is already allocated".

**Solution**:
```bash
# Find process using the port
netstat -ano | findstr :8000  # Windows
lsof -i :8000  # Linux/Mac

# Kill the process or change port in docker-compose.yml
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/rag_system

# Run specific test categories
pytest tests/unit/ -v -m unit
pytest tests/integration/ -v -m integration
pytest tests/e2e/ -v -m e2e

# Run specific test file
pytest tests/unit/ingestion/test_markitdown_converter.py -v
```

### Code Quality

```bash
# Format code with black
black src/ tests/

# Lint with ruff
ruff check src/ tests/

# Fix linting issues
ruff check --fix src/ tests/
```

### Project Structure

```
RAG/
├── src/
│   └── rag_system/
│       ├── ingestion/          # Document ingestion pipeline
│       │   ├── markitdown_converter.py
│       │   ├── chunker.py
│       │   ├── embeddings.py
│       │   └── ingester.py
│       ├── vector_store/       # Milvus integration
│       │   └── milvus_client.py
│       ├── generation/         # LLM and RAG engine
│       │   ├── ollama_client.py
│       │   └── rag_engine.py
│       ├── evaluation/         # Evaluation metrics
│       │   └── trulens_evaluator.py
│       ├── api/                # FastAPI endpoints
│       │   ├── main.py
│       │   └── models.py
│       └── utils/              # Utilities
│           └── config.py
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── e2e/                    # End-to-end tests
├── docker-compose.yml          # Docker orchestration
├── Dockerfile                  # FastAPI container
├── pyproject.toml              # Project configuration
├── requirements.txt            # Production dependencies
└── requirements-dev.txt        # Development dependencies
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Write tests for your changes
4. Ensure all tests pass with >80% coverage
5. Submit a pull request

## Acknowledgments

- **LlamaIndex**: RAG framework
- **Milvus**: Vector database
- **Ollama**: LLM and embedding service
- **TruLens**: Evaluation framework
- **MarkItDown**: Document conversion library

> **Songs listened during the making**: 11:11 (Roa), PPC (ROA, Hades66), MAMI 100PRE SABE (interlude) (Alvaro Diaz, Nsqk)
