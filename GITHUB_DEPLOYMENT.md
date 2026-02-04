# GitHub Deployment Guide

This guide provides step-by-step instructions for deploying the Closed-Loop RAG System to GitHub and setting up the demo from a cloned repository.

## Table of Contents

- [Creating a GitHub Repository](#creating-a-github-repository)
- [Pushing Code to GitHub](#pushing-code-to-github)
- [Setting up GitHub Actions (Optional)](#setting-up-github-actions-optional)
- [Demo Setup from GitHub Clone](#demo-setup-from-github-clone)
- [Troubleshooting](#troubleshooting)

---

## Creating a GitHub Repository

### Option 1: Create via GitHub Web Interface

1. **Sign in to GitHub** at [github.com](https://github.com)

2. **Create a new repository**:
   - Click the **+** icon in the top-right corner
   - Select **New repository**

3. **Configure repository settings**:
   - **Repository name**: `rag-system` (or your preferred name)
   - **Description**: `Production-ready RAG system with automated evaluation using TruLens`
   - **Visibility**: Choose **Public** or **Private**
   - **Initialize with**: Leave all checkboxes unchecked (we'll push existing code)

4. **Click** **Create repository**

5. **Copy the repository URL** (you'll need it for pushing code):
   - HTTPS: `https://github.com/YOUR_USERNAME/rag-system.git`
   - SSH: `git@github.com:YOUR_USERNAME/rag-system.git`

### Option 2: Create via GitHub CLI

If you have the [GitHub CLI](https://cli.github.com/) installed:

```bash
# Create a new repository
gh repo create rag-system \
  --description "Production-ready RAG system with automated evaluation using TruLens" \
  --public \
  --source=. \
  --remote=origin \
  --push
```

---

## Pushing Code to GitHub

### Prerequisites

Ensure you have Git installed and configured:

```bash
# Check Git installation
git --version

# Configure Git (if not already done)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### Option 1: Push via HTTPS

1. **Initialize Git repository** (if not already initialized):

```bash
cd Z:\Gemini\RAG
git init
```

2. **Add all files to staging**:

```bash
git add .
```

3. **Create initial commit**:

```bash
git commit -m "Initial commit: Closed-Loop RAG System"
```

4. **Add remote repository**:

```bash
git remote add origin https://github.com/YOUR_USERNAME/rag-system.git
```

5. **Push to GitHub**:

```bash
# First push (sets upstream branch)
git push -u origin main

# If your local branch is named 'master', rename it first:
git branch -M main
git push -u origin main
```

6. **Authenticate** (if prompted):
   - Enter your GitHub username
   - Enter a **Personal Access Token** (password authentication is deprecated)
   - Create a token at: https://github.com/settings/tokens

### Option 2: Push via SSH

1. **Set up SSH keys** (if not already done):

```bash
# Generate SSH key pair
ssh-keygen -t ed25519 -C "your.email@example.com"

# Start SSH agent
eval "$(ssh-agent -s)"

# Add private key to agent
ssh-add ~/.ssh/id_ed25519
```

2. **Add public key to GitHub**:
   - Copy your public key: `cat ~/.ssh/id_ed25519.pub`
   - Go to: https://github.com/settings/keys
   - Click **New SSH key**
   - Paste your public key and save

3. **Initialize Git repository** (if not already initialized):

```bash
cd Z:\Gemini\RAG
git init
```

4. **Add all files to staging**:

```bash
git add .
```

5. **Create initial commit**:

```bash
git commit -m "Initial commit: Closed-Loop RAG System"
```

6. **Add remote repository**:

```bash
git remote add origin git@github.com:YOUR_USERNAME/rag-system.git
```

7. **Push to GitHub**:

```bash
# First push (sets upstream branch)
git push -u origin main

# If your local branch is named 'master', rename it first:
git branch -M main
git push -u origin main
```

### Verify Deployment

After pushing, verify your repository on GitHub:

1. Visit your repository URL: `https://github.com/YOUR_USERNAME/rag-system`
2. Check that all files are present
3. Verify the README.md displays correctly
4. Confirm LICENSE and CONTRIBUTING.md are visible

---

## Setting up GitHub Actions (Optional)

GitHub Actions can automate testing and deployment workflows. This section provides a basic setup for running tests on push.

### Create GitHub Actions Workflow

1. **Create the workflow directory**:

```bash
mkdir -p .github/workflows
```

2. **Create a test workflow file** (`.github/workflows/test.yml`):

```yaml
name: Run Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-dev.txt

    - name: Run tests with coverage
      run: |
        pytest tests/ -v --cov=src/rag_system --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

3. **Commit and push the workflow**:

```bash
git add .github/workflows/test.yml
git commit -m "Add GitHub Actions test workflow"
git push
```

4. **Monitor workflow runs**:
   - Go to your repository on GitHub
   - Click the **Actions** tab
   - View workflow runs and logs

### Optional: Docker Image Build Workflow

For automated Docker image building and pushing to Docker Hub:

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [ main ]
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3

    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}

    - name: Build and push
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          YOUR_USERNAME/rag-system:latest
          YOUR_USERNAME/rag-system:${{ github.ref_name }}
```

---

## Demo Setup from GitHub Clone

This section provides instructions for setting up and running the demo from a freshly cloned repository.

### Prerequisites

- **Docker** and **Docker Compose** installed
- **Git** installed
- **8GB+ VRAM** (for running Llama 3 8B locally)
- **10GB+ disk space** (for Docker images and data)

### Step 1: Clone the Repository

```bash
# Clone via HTTPS
git clone https://github.com/YOUR_USERNAME/rag-system.git

# Or clone via SSH
git clone git@github.com:YOUR_USERNAME/rag-system.git

# Navigate into the project directory
cd rag-system
```

### Step 2: Configure Environment Variables (Optional)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your custom settings (optional)
# Default values work for most use cases
```

### Step 3: Start Docker Services

```bash
# Start all services in detached mode
docker-compose up -d

# Verify services are running
docker-compose ps
```

Expected output:
```
NAME                  STATUS              PORTS
etcd                  Up                  2379-2380/tcp
minio                 Up                  9000-9001/tcp
milvus-standalone     Up                  19530/tcp, 9091/tcp
ollama                Up                  11434/tcp
rag-app               Up                  0.0.0.0:8000->8000/tcp
trulens               Up                  0.0.0.0:8501->8501/tcp
```

### Step 4: Wait for Services to Initialize

```bash
# Wait approximately 20 seconds for services to be ready
sleep 20

# Check service health
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "ollama": "connected",
    "milvus": "connected"
  }
}
```

### Step 5: Download Ollama Models

```bash
# Pull required models (Llama 3 8B and nomic-embed-text)
docker-compose exec ollama ollama pull llama3:8b nomic-embed-text

# Verify models are installed
docker-compose exec ollama ollama list
```

Expected output:
```
NAME                    ID              SIZE    MODIFIED
llama3:8b               a6990ed6be09    4.7 GB  2 hours ago
nomic-embed-text        0a109f422b47    274 MB  2 hours ago
```

### Step 6: Run the Demo

#### Option 1: Using the Demo Script

```bash
# Make the demo script executable (Linux/Mac)
chmod +x run_demo.sh

# Run the demo
./run_demo.sh
```

#### Option 2: Manual Demo Steps

**Ingest a document**:

```bash
# Create a sample document (or use your own)
echo "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval systems with generative models to enhance response accuracy and relevance." > sample.txt

# Ingest the document
curl -X POST http://localhost:8000/ingest \
  -F "file=@sample.txt"
```

Expected response:
```json
{
  "document_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "chunk_count": 1,
  "source": "sample.txt",
  "error": null
}
```

**Query the system**:

```bash
# Query the ingested document
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'
```

Expected response:
```json
{
  "query": "What is RAG?",
  "answer": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval systems with generative models to enhance response accuracy and relevance.",
  "sources": [
    {
      "text": "RAG (Retrieval-Augmented Generation) is a technique that combines retrieval systems with generative models to enhance response accuracy and relevance.",
      "score": 0.95,
      "metadata": {
        "source": "sample.txt",
        "chunk_index": 0
      }
    }
  ],
  "retrieved_count": 1,
  "evaluation": {
    "faithfulness": 0.90,
    "relevance": 0.95,
    "context_precision": 0.95,
    "context_recall": 1.00,
    "overall_score": 0.93
  }
}
```

### Step 7: View the TruLens Dashboard

Open your browser and navigate to:

```
http://localhost:8501
```

The TruLens dashboard displays evaluation metrics and query history.

### Step 8: Stop Services

When you're done, stop all services:

```bash
# Stop all services
docker-compose down

# Stop services and remove volumes (clears all data)
docker-compose down -v
```

---

## Troubleshooting

### Git Push Issues

**Problem**: Authentication failed when pushing to GitHub.

**Solution**:
```bash
# For HTTPS: Use a Personal Access Token instead of password
# Create token at: https://github.com/settings/tokens

# For SSH: Verify SSH key is added to GitHub
ssh -T git@github.com

# If SSH fails, check your SSH key:
ls -la ~/.ssh/id_ed25519*
```

**Problem**: "remote origin already exists" error.

**Solution**:
```bash
# Remove existing remote
git remote remove origin

# Add new remote
git remote add origin https://github.com/YOUR_USERNAME/rag-system.git
```

### Docker Issues

**Problem**: Docker services fail to start.

**Solution**:
```bash
# Check Docker status
docker ps

# Check service logs
docker-compose logs [service-name]

# Restart services
docker-compose restart

# Rebuild and restart
docker-compose up -d --build
```

**Problem**: Port already in use (e.g., port 8000).

**Solution**:
```bash
# Find process using the port (Windows)
netstat -ano | findstr :8000

# Find process using the port (Linux/Mac)
lsof -i :8000

# Kill the process or change port in docker-compose.yml
```

### Ollama Model Issues

**Problem**: "model 'llama3:8b' not found" error.

**Solution**:
```bash
# Pull required models
docker-compose exec ollama ollama pull llama3:8b nomic-embed-text

# Verify models are installed
docker-compose exec ollama ollama list

# If pull fails, check Ollama logs
docker-compose logs ollama
```

### Milvus Connection Issues

**Problem**: Cannot connect to Milvus database.

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

### Out of Memory Errors

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

### Slow Query Performance

**Problem**: Queries take more than 10 seconds to complete.

**Solution**:
```bash
# Check Milvus index status
docker-compose exec milvus-standalone python -c "from pymilvus import utility; print(utility.index_info('documents'))"

# Reduce top_k parameter in query
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query": "...", "top_k": 3}'

# Increase Milvus resources in docker-compose.yml
```

### GitHub Actions Issues

**Problem**: Workflow fails with "Python not found" error.

**Solution**:
```yaml
# Ensure Python version is correctly specified in workflow
- name: Set up Python
  uses: actions/setup-python@v4
  with:
    python-version: '3.11'
```

**Problem**: Tests fail in CI but pass locally.

**Solution**:
```bash
# Check workflow logs for specific error messages
# Ensure all dependencies are in requirements.txt
# Verify PYTHONPATH is set correctly in workflow
```

### Additional Resources

- **GitHub Documentation**: https://docs.github.com
- **Docker Documentation**: https://docs.docker.com
- **Ollama Documentation**: https://ollama.ai/docs
- **Milvus Documentation**: https://milvus.io/docs
- **Project README**: See [README.md](README.md) for detailed project documentation

---

## Next Steps

After successfully deploying to GitHub:

1. **Set up GitHub Pages** (optional) for documentation hosting
2. **Configure branch protection rules** for main branch
3. **Set up issue templates** for better issue tracking
4. **Add a CODEOWNERS file** for code review assignments
5. **Configure Dependabot** for automated dependency updates

For more information on contributing to the project, see [CONTRIBUTING.md](CONTRIBUTING.md).
