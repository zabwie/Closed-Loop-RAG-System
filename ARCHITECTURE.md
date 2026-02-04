# Architecture Documentation

## System Overview

The Closed-Loop RAG System is a production-ready Retrieval-Augmented Generation (RAG) system that enables document ingestion, semantic search, and automated response evaluation. The system follows a closed-loop architecture where query responses are automatically evaluated using heuristic-based metrics, providing continuous feedback on retrieval and generation quality.

### Key Characteristics

- **Self-Hosted**: All components run locally using Docker Compose, with no external API dependencies
- **Multi-Format Support**: Ingests PDF, Word, Excel, PowerPoint, HTML, CSV, and more via MarkItDown
- **Semantic Search**: Vector-based retrieval using Milvus with HNSW indexing
- **Automated Evaluation**: SimulatedEvaluator provides RAG triad metrics without external APIs
- **RESTful API**: Simple FastAPI endpoints for ingestion and querying
- **Production-Ready**: Full test coverage (>80%), Docker-based deployment, comprehensive error handling

### System Goals

1. **Accuracy**: Retrieve relevant documents and generate accurate responses
2. **Evaluability**: Provide metrics to assess response quality
3. **Scalability**: Handle multiple documents and concurrent queries
4. **Maintainability**: Clean architecture with clear separation of concerns
5. **Self-Contained**: No external API dependencies for core functionality

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Client Layer                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         REST Client                                   │  │
│  │  (curl, Postman, Python requests, etc.)                               │  │
│  └──────────────────────────────┬───────────────────────────────────────┘  │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API Layer                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         FastAPI Application                           │  │
│  │  ┌──────────────┐              ┌──────────────┐                      │  │
│  │  │  POST /ingest│              │  POST /query │                      │  │
│  │  └──────┬───────┘              └──────┬───────┘                      │  │
│  └─────────┼──────────────────────────────┼───────────────────────────────┘  │
└────────────┼──────────────────────────────┼──────────────────────────────────┘
             │                              │
             ▼                              ▼
┌──────────────────────────────┐    ┌──────────────────────────────────────┐
│  Ingestion Pipeline          │    │      RAG Pipeline                    │
│  ┌────────────────────────┐  │    │  ┌────────────────────────────────┐  │
│  │ MarkItDownConverter    │  │    │  │ EmbeddingService               │  │
│  │ (Document → Text)      │  │    │  │ (Query → Vector)               │  │
│  └───────────┬────────────┘  │    │  └────────────┬───────────────────┘  │
│              │               │    │               │                       │
│              ▼               │    │               ▼                       │
│  ┌────────────────────────┐  │    │  ┌────────────────────────────────┐  │
│  │ TextChunker            │  │    │  │ MilvusVectorStore              │  │
│  │ (Text → Chunks)        │  │    │  │ (Vector Search)                │  │
│  └───────────┬────────────┘  │    │  └────────────┬───────────────────┘  │
│              │               │    │               │                       │
│              ▼               │    │               ▼                       │
│  ┌────────────────────────┐  │    │  ┌────────────────────────────────┐  │
│  │ EmbeddingService       │  │    │  │ OllamaClient                   │  │
│  │ (Chunks → Vectors)     │  │    │  │ (Context + Query → Answer)     │  │
│  └───────────┬────────────┘  │    │  └────────────┬───────────────────┘  │
│              │               │    │               │                       │
│              ▼               │    │               ▼                       │
│  ┌────────────────────────┐  │    │  ┌────────────────────────────────┐  │
│  │ MilvusVectorStore      │  │    │  │ SimulatedEvaluator             │  │
│  │ (Vectors → Storage)    │  │    │  │ (RAG Triad Metrics)            │  │
│  └────────────────────────┘  │    │  └────────────────────────────────┘  │
└──────────────────────────────┘    └──────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Infrastructure Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Milvus     │  │    Ollama    │  │    TruLens   │  │   MinIO/     │  │
│  │  (Vector DB) │  │   (LLM/Emb)  │  │  (Dashboard) │  │    etcd      │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### Ingestion Pipeline

The ingestion pipeline transforms raw documents into indexed vector representations stored in Milvus.

#### MarkItDownConverter

**Purpose**: Converts various document formats into plain text with metadata extraction.

**Supported Formats**:
- PDF (.pdf)
- Word (.docx)
- Excel (.xlsx)
- PowerPoint (.pptx)
- HTML (.html)
- CSV (.csv)
- Markdown (.md)
- Plain text (.txt)

**Implementation**:
- Uses Microsoft's MarkItDown library for conversion
- Extracts text content and metadata (source, format, character count, word count)
- Handles errors gracefully with logging
- Returns structured output with conversion results

**Key Methods**:
- `convert(file_path: str) -> ConversionResult`: Converts document to text

#### TextChunker

**Purpose**: Splits long text into manageable chunks for embedding and retrieval.

**Configuration**:
- Chunk size: 512 tokens (default)
- Overlap: 50 tokens (default)
- Preserves word boundaries

**Implementation**:
- Fixed-size chunking with configurable parameters
- Creates overlapping chunks to preserve context
- Returns list of chunks with metadata (source, chunk_index, character count)
- Handles edge cases (empty text, single chunk)

**Key Methods**:
- `chunk(text: str) -> List[Chunk]`: Splits text into chunks

#### EmbeddingService

**Purpose**: Generates vector embeddings for text chunks and queries.

**Configuration**:
- Model: nomic-embed-text (Ollama)
- Embedding dimension: 768
- Batch processing support

**Implementation**:
- Uses Ollama Python library for embedding generation
- Supports batch embedding for efficiency
- Provides single text embedding convenience method
- Handles connection errors and timeouts

**Key Methods**:
- `embed(texts: List[str]) -> List[List[float]]`: Generates embeddings for multiple texts
- `embed_single(text: str) -> List[float]`: Generates embedding for single text

#### DocumentIngester

**Purpose**: Orchestrates the complete ingestion pipeline.

**Implementation**:
- Coordinates: Converter → Chunker → Embedding → Milvus upsert
- Generates unique document_id using UUID
- Tracks ingestion status (completed/failed), chunk_count, and source
- Merges metadata from converter and chunker with document_id
- Handles errors gracefully with detailed error information

**Key Methods**:
- `ingest(file_path: str) -> IngestionResult`: Ingests document into the system

### RAG Pipeline

The RAG pipeline processes user queries, retrieves relevant documents, generates responses, and evaluates quality.

#### EmbeddingService (Query)

**Purpose**: Generates vector embedding for user queries.

**Implementation**:
- Reuses the same EmbeddingService from ingestion pipeline
- Uses nomic-embed-text model for query embedding
- Ensures consistency between document and query embeddings

#### MilvusVectorStore

**Purpose**: Stores and retrieves vector embeddings with similarity search.

**Configuration**:
- Index type: HNSW (Hierarchical Navigable Small World)
- Metric type: IP (Inner Product)
- Dimension: 768 (nomic-embed-text embedding dimension)
- Collection name: documents

**Implementation**:
- Uses PyMilvus library for vector database operations
- Implements async `insert()` method for batch embedding insertion
- Implements async `search()` method for similarity search with top_k parameter
- Generates unique UUIDs for each document ID
- Returns search results with similarity scores and metadata

**Key Methods**:
- `connect() -> None`: Connects to Milvus server
- `create_collection() -> None`: Creates collection with HNSW index
- `insert(embeddings: List[List[float]], metadata: List[Dict]) -> None`: Inserts embeddings
- `search(query_embedding: List[float], top_k: int) -> List[SearchResult]`: Performs similarity search

#### OllamaClient

**Purpose**: Generates responses using Llama 3 8B model.

**Configuration**:
- Model: llama3:8b (Ollama)
- Base URL: http://localhost:11434
- Timeout: 120 seconds

**Implementation**:
- Wraps Ollama HTTP API for chat completion using httpx.AsyncClient
- Supports custom base_url and model
- Implements async `chat()` method with prompt and optional context
- Formats RAG prompt with context and question
- Handles connection failures, timeouts, and generic errors

**Key Methods**:
- `chat(prompt: str, context: Optional[str] = None) -> str`: Generates response

#### RAGQueryEngine

**Purpose**: Orchestrates the complete RAG pipeline.

**Implementation**:
- Coordinates: Embedding → Retrieval → Generation
- Uses EmbeddingService for query embedding
- Uses MilvusVectorStore for similarity search
- Uses OllamaClient for chat completion with context
- Returns structured response with answer, sources, and retrieved_count
- Handles no results gracefully with "No relevant documents found." message
- Assembles context by joining retrieved document texts

**Key Methods**:
- `query(question: str, top_k: int = 5) -> QueryResult`: Processes query and returns results

### Evaluation

#### SimulatedEvaluator

**Purpose**: Evaluates RAG response quality using heuristic-based metrics.

**Metrics**:
- **Faithfulness**: Word overlap between answer and sources (0-1)
- **Context Precision**: Average source score (0-1)
- **Context Recall**: Retrieved count / ideal count (0-1)
- **Answer Relevance**: Word overlap between query and answer (0-1)
- **Overall Score**: Weighted average (faithfulness 30%, relevance 30%, precision 20%, recall 20%)

**Implementation**:
- Uses heuristics instead of external APIs (self-hosted constraint)
- Calculates word overlap for faithfulness and relevance
- Averages source scores for context precision
- Uses ideal count of 5 for context recall
- Returns evaluation object with all metrics

**Key Methods**:
- `evaluate_query(query: str, answer: str, sources: List[Dict]) -> EvaluationResult`: Evaluates query response

### API Layer

#### FastAPI Application

**Purpose**: Provides RESTful API endpoints for ingestion and querying.

**Endpoints**:
- `POST /ingest`: Upload and ingest documents
- `POST /query`: Query the RAG system with evaluation
- `GET /health`: Health check endpoint
- `GET /`: Root endpoint

**Implementation**:
- Uses FastAPI framework for async request handling
- Integrates DocumentIngester for ingestion
- Integrates RAGQueryEngine and SimulatedEvaluator for querying
- Returns structured responses with Pydantic models
- Handles errors gracefully with HTTPException
- Provides health check for service status

**Models**:
- `IngestRequest`: File upload via multipart/form-data
- `IngestResponse`: document_id, status, chunk_count, source, error
- `QueryRequest`: query (str), top_k (int, default 5, range 1-20)
- `QueryResponse`: query, answer, sources, retrieved_count, evaluation

## Data Flow

### Ingestion Flow

```
┌─────────────┐
│   Client    │
│  Uploads    │
│  Document   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ POST /ingest (FastAPI)                                      │
│ 1. Receives file via multipart/form-data                    │
│ 2. Creates temporary file                                   │
│ 3. Generates document_id (UUID)                             │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ DocumentIngester.ingest()                                   │
└──────┬──────────────────────────────────────────────────────┘
       │
       ├──────────────────────────────────────────────────────┐
       │                                                      │
       ▼                                                      ▼
┌──────────────────────┐                          ┌──────────────────────┐
│ MarkItDownConverter  │                          │  Temporary File      │
│ .convert()           │                          │  Cleanup             │
│ 1. Extracts text     │                          │  (finally block)     │
│ 2. Extracts metadata │                          └──────────────────────┘
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ TextChunker.chunk()  │
│ 1. Splits text       │
│ 2. Creates chunks    │
│ 3. Adds metadata     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ EmbeddingService     │
│ .embed()             │
│ 1. Generates vectors │
│ 2. Returns embeddings│
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ MilvusVectorStore    │
│ .insert()            │
│ 1. Upserts vectors   │
│ 2. Stores metadata   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ IngestionResult      │
│ 1. document_id       │
│ 2. status            │
│ 3. chunk_count       │
│ 4. source            │
│ 5. error (if any)    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ POST /ingest Response│
│ JSON: {              │
│   "document_id": "...",│
│   "status": "completed",│
│   "chunk_count": 42,   │
│   "source": "sample.pdf",│
│   "error": null        │
│ }                     │
└──────────────────────┘
```

### Query Flow

```
┌─────────────┐
│   Client    │
│  Sends      │
│  Query      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ POST /query (FastAPI)                                       │
│ 1. Receives JSON: {query, top_k}                           │
│ 2. Validates top_k (1-20)                                   │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ RAGQueryEngine.query()                                      │
└──────┬──────────────────────────────────────────────────────┘
       │
       ├──────────────────────────────────────────────────────┐
       │                                                      │
       ▼                                                      ▼
┌──────────────────────┐                          ┌──────────────────────┐
│ EmbeddingService     │                          │  Error Handling      │
│ .embed_single()      │                          │  (if any step fails) │
│ 1. Generates query   │                          └──────────────────────┘
│    embedding         │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ MilvusVectorStore    │
│ .search()            │
│ 1. Performs vector   │
│    similarity search │
│ 2. Returns top_k     │
│    results           │
└──────┬───────────────┘
       │
       ├──────────────────────────────────────────────────────┐
       │                                                      │
       ▼                                                      ▼
┌──────────────────────┐                          ┌──────────────────────┐
│ Results Found?       │                          │ No Results           │
│                      │                          │ Return:              │
│ Yes                  │                          │ "No relevant         │
│                      │                          │  documents found."   │
└──────┬───────────────┘                          └──────────────────────┘
       │
       ▼
┌──────────────────────┐
│ Context Assembly     │
│ 1. Joins retrieved   │
│    document texts    │
│ 2. Formats context   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ OllamaClient.chat()  │
│ 1. Sends context +   │
│    question to LLM   │
│ 2. Receives answer   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ QueryResult          │
│ 1. answer            │
│ 2. sources           │
│ 3. retrieved_count   │
└──────┬───────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ SimulatedEvaluator.evaluate_query()                         │
│ 1. Calculates faithfulness (answer ↔ sources)               │
│ 2. Calculates relevance (query ↔ answer)                    │
│ 3. Calculates context precision (avg source score)          │
│ 4. Calculates context recall (retrieved / ideal)            │
│ 5. Calculates overall score (weighted average)              │
└──────┬──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────┐
│ EvaluationResult     │
│ 1. faithfulness      │
│ 2. relevance         │
│ 3. context_precision │
│ 4. context_recall    │
│ 5. overall_score     │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ POST /query Response │
│ JSON: {              │
│   "query": "...",    │
│   "answer": "...",   │
│   "sources": [...],  │
│   "retrieved_count": 5,│
│   "evaluation": {    │
│     "faithfulness": 0.75,│
│     "relevance": 0.80,  │
│     "context_precision": 0.70,│
│     "context_recall": 0.60,│
│     "overall_score": 0.72 │
│   }                  │
│ }                     │
└──────────────────────┘
```

## Deployment Architecture

### Docker Compose Orchestration

The system uses Docker Compose to orchestrate all services in a single deployment.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Docker Host                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                         Docker Network                                │  │
│  │  (rag-network)                                                        │  │
│  │                                                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │   etcd       │  │    MinIO     │  │   Milvus     │              │  │
│  │  │  :2379       │  │   :9000      │  │  :19530      │              │  │
│  │  │  (Metadata)  │  │  (Storage)   │  │  (Vector DB) │              │  │
│  │  └──────────────┘  └──────────────┘  └──────┬───────┘              │  │
│  │       │                  │                  │                       │  │
│  │       └──────────────────┴──────────────────┘                       │  │
│  │                              │                                       │  │
│  │                              ▼                                       │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │   Ollama     │  │   TruLens    │  │   rag-app    │              │  │
│  │  │  :11434      │  │   :8501      │  │   :8000      │              │  │
│  │  │  (LLM/Emb)   │  │  (Dashboard) │  │  (FastAPI)   │              │  │
│  │  └──────────────┘  └──────────────┘  └──────┬───────┘              │  │
│  │                                                   │                   │  │
│  └───────────────────────────────────────────────────┼───────────────────┘  │
│                                                      │                      │
└──────────────────────────────────────────────────────┼──────────────────────┘
                                                       │
                                                       ▼
                                              ┌────────────────┐
                                              │   Client       │
                                              │  (Browser,     │
                                              │   curl, etc.)  │
                                              └────────────────┘
```

### Service Dependencies

```
etcd (Metadata Storage)
    │
    └──► milvus-standalone (depends on etcd: service_started)

minio (Object Storage)
    │
    └──► milvus-standalone (depends on minio: service_healthy)

milvus-standalone (Vector Database)
    │
    └──► rag-app (depends on milvus-standalone: service_started)

ollama (LLM Service)
    │
    └──► rag-app (depends on ollama: service_started)

trulens (Evaluation Dashboard)
    │
    └──► (independent service)
```

### Volume Configuration

```
Named Volumes:
- ollama_data: Persistent storage for Ollama models
- trulens_data: Persistent storage for TruLens evaluation data

Bind Mounts:
- ./docker-data/etcd: etcd data directory
- ./docker-data/minio: MinIO data directory
- ./docker-data/milvus: Milvus data directory
- ./data:/app/data: Application data directory
- ./trulens.db:/trulens/trulens.db: TruLens database
```

### Port Mappings

| Service | Internal Port | External Port | Purpose |
|---------|---------------|---------------|---------|
| etcd | 2379 | - | Metadata storage (internal only) |
| minio | 9000 | - | Object storage (internal only) |
| milvus-standalone | 19530 | 19530 | Vector database API |
| milvus-standalone | 9091 | 9091 | Milvus metrics |
| ollama | 11434 | 11434 | LLM and embedding API |
| trulens | 8501 | 8501 | Evaluation dashboard |
| rag-app | 8000 | 8000 | FastAPI application |

## Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **RAG Framework** | LlamaIndex | Latest | RAG orchestration and utilities |
| **Vector Database** | Milvus | v2.3.3 | Vector storage and similarity search |
| **LLM** | Ollama (Llama 3 8B) | Latest | Text generation |
| **Embedding Model** | Ollama (nomic-embed-text) | Latest | Text-to-vector conversion |
| **API Framework** | FastAPI | Latest | RESTful API |
| **Document Conversion** | MarkItDown | Latest | Multi-format document parsing |
| **Evaluation** | SimulatedEvaluator | Custom | Heuristic-based RAG metrics |
| **Orchestration** | Docker Compose | v3.8+ | Service deployment |

### Python Dependencies

**Production**:
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `pydantic`: Data validation
- `pydantic-settings`: Configuration management
- `httpx`: Async HTTP client
- `pymilvus`: Milvus Python client
- `ollama`: Ollama Python client
- `markitdown`: Document conversion

**Development**:
- `pytest`: Testing framework
- `pytest-asyncio`: Async test support
- `pytest-cov`: Coverage reporting
- `pytest-mock`: Mocking utilities
- `black`: Code formatting
- `ruff`: Code linting

### Infrastructure

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Container Runtime** | Docker | Application containerization |
| **Orchestration** | Docker Compose | Multi-container management |
| **Metadata Storage** | etcd | Milvus metadata |
| **Object Storage** | MinIO | Milvus data storage |
| **Vector Index** | HNSW | Fast similarity search |

## Design Decisions

### Evaluation Approach

**Decision**: Use SimulatedEvaluator with heuristic-based metrics instead of TruLens instrumentation.

**Rationale**:
- TruLens requires OpenAI API for evaluation metrics
- User specified self-hosted Llama 3 constraint
- Heuristics provide reasonable approximation for demo purposes
- No external API dependencies for core functionality

**Trade-offs**:
- Heuristics are less accurate than LLM-based evaluation
- Word overlap has limitations (common words create false positives)
- Suitable for demo and development, may need enhancement for production

### Vector Index Selection

**Decision**: Use HNSW (Hierarchical Navigable Small World) index for Milvus.

**Rationale**:
- Fast query performance for similarity search
- Reasonable memory footprint
- Good balance between speed and accuracy
- Well-suited for demo and production use cases

**Trade-offs**:
- Higher memory usage than IVF_FLAT
- Slower index build time than IVF_FLAT
- Suitable for most use cases, may need tuning for specific scenarios

### Embedding Model Selection

**Decision**: Use nomic-embed-text model via Ollama.

**Rationale**:
- Self-hosted (no external API dependencies)
- Good performance for semantic search
- 768-dimensional embeddings (reasonable size)
- Compatible with Ollama infrastructure

**Trade-offs**:
- Smaller model than OpenAI embeddings (less accurate)
- Requires local GPU/CPU resources
- Suitable for demo, may need larger model for production

### Generation Model Selection

**Decision**: Use Llama 3 8B model via Ollama.

**Rationale**:
- Self-hosted (no external API dependencies)
- Good balance of performance and resource requirements
- 8GB+ VRAM requirement (reasonable for modern GPUs)
- Compatible with Ollama infrastructure

**Trade-offs**:
- Smaller model than GPT-4 (less accurate)
- Requires local GPU resources
- Suitable for demo, may need larger model for production

### Chunking Strategy

**Decision**: Use fixed-size chunking with 512 tokens and 50 token overlap.

**Rationale**:
- Simple and predictable behavior
- Overlap preserves context between chunks
- 512 tokens is a reasonable size for most documents
- Compatible with embedding model context window

**Trade-offs**:
- May split semantic boundaries (paragraphs, sections)
- Fixed size may not be optimal for all document types
- Suitable for demo, may need semantic chunking for production

### Retrieval Strategy

**Decision**: Use semantic-only retrieval with top_k=5.

**Rationale**:
- Simple and effective for most use cases
- Vector search provides good semantic matching
- top_k=5 balances relevance and context
- Avoids complexity of hybrid search

**Trade-offs**:
- May miss exact keyword matches
- No keyword search for specific terms
- Suitable for demo, may need hybrid search for production

### API Design

**Decision**: Use FastAPI with simple RESTful endpoints.

**Rationale**:
- Fast and modern web framework
- Automatic API documentation (Swagger UI)
- Async support for better performance
- Pydantic models for type safety

**Trade-offs**:
- Simpler than GraphQL (less flexible)
- RESTful design may need multiple calls for complex queries
- Suitable for demo, may need GraphQL for production

### Testing Strategy

**Decision**: Use TDD with pytest and >80% coverage threshold.

**Rationale**:
- Ensures testability from the start
- Living documentation through tests
- Confidence in changes
- Industry-standard testing framework

**Trade-offs**:
- Slower initial development
- Requires discipline
- Coverage threshold may slow development initially

### Deployment Strategy

**Decision**: Use Docker Compose for local deployment.

**Rationale**:
- Simple and reproducible deployment
- All services in single configuration
- Easy to start and stop
- Suitable for local development and demo

**Trade-offs**:
- Not suitable for production scaling
- No load balancing or high availability
- Suitable for demo, may need Kubernetes for production

### Configuration Management

**Decision**: Use Pydantic Settings with environment variables.

**Rationale**:
- Type-safe configuration with validation
- Environment variable support for flexibility
- Default values for easy local development
- Clear separation of configuration and code

**Trade-offs**:
- Additional dependency (pydantic-settings)
- Requires environment variable setup for customization
- Suitable for demo, may need more sophisticated config for production

### Error Handling

**Decision**: Graceful error handling with detailed error messages.

**Rationale**:
- Better user experience
- Easier debugging
- Prevents cascading failures
- Consistent error responses

**Trade-offs**:
- More code to write and maintain
- May hide underlying issues
- Suitable for demo, may need more sophisticated error handling for production

### Logging Strategy

**Decision**: Use Python logging with appropriate log levels.

**Rationale**:
- Standard Python logging
- Configurable log levels
- Structured logging for better analysis
- Easy to integrate with log aggregation tools

**Trade-offs**:
- Requires log configuration
- May generate large log files
- Suitable for demo, may need log aggregation for production

### Data Persistence

**Decision**: Use Docker volumes for persistent data storage.

**Rationale**:
- Data survives container restarts
- Easy backup and restore
- Separation of data and containers
- Standard Docker practice

**Trade-offs**:
- Requires volume management
- May need cleanup for old data
- Suitable for demo, may need more sophisticated storage for production

### Security Considerations

**Decision**: Basic security with no authentication (demo mode).

**Rationale**:
- Simple setup for demo
- No external dependencies
- Easy to test and debug
- Suitable for local development

**Trade-offs**:
- Not suitable for production
- No access control
- Suitable for demo, must add authentication for production

### Performance Optimization

**Decision**: Batch embedding and async operations.

**Rationale**:
- Batch embedding reduces API calls
- Async operations improve concurrency
- Better resource utilization
- Faster overall performance

**Trade-offs**:
- More complex code
- Requires async/await understanding
- Suitable for demo, may need more optimization for production

### Scalability Considerations

**Decision**: Single-instance deployment (demo mode).

**Rationale**:
- Simple setup for demo
- No load balancing needed
- Easy to understand and debug
- Suitable for local development

**Trade-offs**:
- Not suitable for production scaling
- No high availability
- Suitable for demo, must add scaling for production

### Monitoring and Observability

**Decision**: Basic health check endpoint and TruLens dashboard.

**Rationale**:
- Simple health monitoring
- Evaluation metrics visualization
- Easy to understand system status
- Suitable for demo

**Trade-offs**:
- Limited monitoring capabilities
- No alerting or notifications
- Suitable for demo, may need comprehensive monitoring for production

### Documentation Strategy

**Decision**: Comprehensive README and ARCHITECTURE documentation.

**Rationale**:
- Clear project overview
- Easy onboarding for new developers
- Architecture documentation for understanding
- API documentation for integration

**Trade-offs**:
- Requires maintenance
- May become outdated
- Suitable for demo, must keep updated for production
