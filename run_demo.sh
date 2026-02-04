#!/bin/bash

# RAG System Demo Script
# This script demonstrates the RAG system by:
# 1. Checking/starting Docker services
# 2. Downloading Ollama models
# 3. Ingesting sample documents
# 4. Running queries with evaluation
# 5. Displaying TruLens dashboard URL

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_URL="http://localhost:8000"
OLLAMA_URL="http://localhost:11434"
TRULENS_URL="http://localhost:8501"
DOCUMENTS_DIR="data/documents"

# Sample documents
DOCUMENTS=(
    "sample_tech_doc.md"
    "sample_api_docs.md"
    "sample_faq.csv"
)

# Pre-defined queries
QUERIES=(
    "What is RAG?"
    "How does the ingestion pipeline work?"
    "What models are supported?"
    "How do I query the system?"
    "What is the chunk size?"
    "How are documents indexed?"
    "What evaluation metrics are used?"
    "How do I start the services?"
    "What is the API endpoint for ingestion?"
    "How does semantic search work?"
)

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker services are running
check_docker_services() {
    print_info "Checking Docker services status..."
    
    if ! docker-compose ps > /dev/null 2>&1; then
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    
    # Check if any services are running
    RUNNING_SERVICES=$(docker-compose ps --services --filter "status=running" | wc -l)
    
    if [ "$RUNNING_SERVICES" -eq 0 ]; then
        print_warning "No Docker services are running."
        return 1
    else
        print_success "Docker services are running ($RUNNING_SERVICES services)."
        return 0
    fi
}

# Function to start Docker services
start_docker_services() {
    print_info "Starting Docker services..."
    
    if docker-compose up -d; then
        print_success "Docker services started successfully."
    else
        print_error "Failed to start Docker services."
        exit 1
    fi
}

# Function to wait for services to be ready
wait_for_services() {
    print_info "Waiting for services to be ready..."
    
    # Wait for rag-app (FastAPI)
    print_info "Waiting for FastAPI service ($API_URL)..."
    MAX_RETRIES=30
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s -f "$API_URL/health" > /dev/null 2>&1; then
            print_success "FastAPI service is ready."
            break
        fi
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo -n "."
        sleep 2
    done
    
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        print_error "FastAPI service did not become ready in time."
        exit 1
    fi
    
    # Wait for Ollama
    print_info "Waiting for Ollama service ($OLLAMA_URL)..."
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -s -f "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
            print_success "Ollama service is ready."
            break
        fi
        RETRY_COUNT=$((RETRY_COUNT + 1))
        echo -n "."
        sleep 2
    done
    
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        print_error "Ollama service did not become ready in time."
        exit 1
    fi
    
    echo ""
}

# Function to download Ollama models
download_ollama_models() {
    print_info "Checking Ollama models..."
    
    # Check if models are already downloaded
    MODELS=$(docker-compose exec -T ollama ollama list 2>/dev/null || echo "")
    
    if echo "$MODELS" | grep -q "llama3:8b"; then
        print_success "Model llama3:8b is already downloaded."
    else
        print_info "Downloading llama3:8b model..."
        if docker-compose exec -T ollama ollama pull llama3:8b; then
            print_success "Model llama3:8b downloaded successfully."
        else
            print_error "Failed to download llama3:8b model."
            exit 1
        fi
    fi
    
    if echo "$MODELS" | grep -q "nomic-embed-text"; then
        print_success "Model nomic-embed-text is already downloaded."
    else
        print_info "Downloading nomic-embed-text model..."
        if docker-compose exec -T ollama ollama pull nomic-embed-text; then
            print_success "Model nomic-embed-text downloaded successfully."
        else
            print_error "Failed to download nomic-embed-text model."
            exit 1
        fi
    fi
}

# Function to ingest a document
ingest_document() {
    local doc_file="$1"
    local doc_path="$DOCUMENTS_DIR/$doc_file"
    
    if [ ! -f "$doc_path" ]; then
        print_error "Document not found: $doc_path"
        return 1
    fi
    
    print_info "Ingesting document: $doc_file"
    
    RESPONSE=$(curl -s -X POST "$API_URL/ingest" \
        -F "file=@$doc_path" \
        -w "\n%{http_code}")
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')
    
    if [ "$HTTP_CODE" -eq 200 ]; then
        print_success "Document ingested successfully."
        echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
        return 0
    else
        print_error "Failed to ingest document (HTTP $HTTP_CODE)."
        echo "$BODY"
        return 1
    fi
}

# Function to run a query
run_query() {
    local query="$1"
    local query_num="$2"
    
    print_info "Query $query_num: $query"
    
    RESPONSE=$(curl -s -X POST "$API_URL/query" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$query\", \"top_k\": 5}" \
        -w "\n%{http_code}")
    
    HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
    BODY=$(echo "$RESPONSE" | sed '$d')
    
    if [ "$HTTP_CODE" -eq 200 ]; then
        print_success "Query executed successfully."
        
        # Extract and display evaluation scores
        if command -v python3 &> /dev/null; then
            echo "$BODY" | python3 -c "
import json
import sys
try:
    data = json.load(sys.stdin)
    print(f\"  Answer: {data.get('answer', 'N/A')[:100]}...\")
    print(f\"  Retrieved: {data.get('retrieved_count', 0)} documents\")
    eval_data = data.get('evaluation', {})
    print(f\"  Evaluation Scores:\")
    print(f\"    - Faithfulness: {eval_data.get('faithfulness', 0):.2f}\")
    print(f\"    - Relevance: {eval_data.get('relevance', 0):.2f}\")
    print(f\"    - Context Precision: {eval_data.get('context_precision', 0):.2f}\")
    print(f\"    - Context Recall: {eval_data.get('context_recall', 0):.2f}\")
    print(f\"    - Overall Score: {eval_data.get('overall_score', 0):.2f}\")
except:
    print('  Failed to parse response')
"
        else
            echo "$BODY" | python -m json.tool 2>/dev/null || echo "$BODY"
        fi
        return 0
    else
        print_error "Failed to execute query (HTTP $HTTP_CODE)."
        echo "$BODY"
        return 1
    fi
}

# Main execution
main() {
    echo "=========================================="
    echo "  RAG System Demo"
    echo "=========================================="
    echo ""
    
    # Step 1: Check/Start Docker services
    if ! check_docker_services; then
        start_docker_services
    fi
    
    # Step 2: Wait for services to be ready
    wait_for_services
    
    # Step 3: Download Ollama models
    download_ollama_models
    
    echo ""
    echo "=========================================="
    echo "  Document Ingestion"
    echo "=========================================="
    echo ""
    
    # Step 4: Ingest sample documents
    INGESTED_COUNT=0
    for doc in "${DOCUMENTS[@]}"; do
        if ingest_document "$doc"; then
            INGESTED_COUNT=$((INGESTED_COUNT + 1))
        fi
        echo ""
    done
    
    if [ $INGESTED_COUNT -eq 0 ]; then
        print_error "No documents were ingested. Exiting."
        exit 1
    fi
    
    print_success "Ingested $INGESTED_COUNT documents."
    
    echo ""
    echo "=========================================="
    echo "  Query Execution"
    echo "=========================================="
    echo ""
    
    # Step 5: Run queries
    QUERY_COUNT=0
    SUCCESS_COUNT=0
    
    for query in "${QUERIES[@]}"; do
        QUERY_COUNT=$((QUERY_COUNT + 1))
        if run_query "$query" "$QUERY_COUNT"; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        fi
        echo ""
    done
    
    print_success "Executed $SUCCESS_COUNT/$QUERY_COUNT queries successfully."
    
    # Step 6: Display TruLens dashboard URL
    echo ""
    echo "=========================================="
    echo "  TruLens Dashboard"
    echo "=========================================="
    echo ""
    print_info "TruLens Dashboard URL: $TRULENS_URL"
    print_info "Open this URL in your browser to view evaluation metrics."
    echo ""
    
    echo "=========================================="
    echo "  Demo Complete!"
    echo "=========================================="
}

# Run main function
main
