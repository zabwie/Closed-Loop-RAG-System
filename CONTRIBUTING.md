# Contributing Guidelines

Thank you for your interest in contributing to the Closed-Loop RAG System! This document provides guidelines for contributing to the project.

## Prerequisites

Before contributing, ensure you have the following installed:

- **Docker** and **Docker Compose** - For running the full stack locally
- **Python 3.11+** - For local development and testing
- **8GB+ VRAM** - For running Llama 3 8B model locally
- **10GB+ disk space** - For Docker images and data

## Development Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd RAG
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 3. Set PYTHONPATH

Set the `PYTHONPATH` environment variable to include the `src` directory:

**Linux/Mac:**
```bash
export PYTHONPATH=$(pwd)/src
```

**Windows:**
```bash
set PYTHONPATH=Z:\Gemini\RAG\src
```

For convenience, add this to your shell profile (`.bashrc`, `.zshrc`, etc.) or Windows environment variables.

### 4. Start Docker Services (Optional)

For integration testing or full-stack development:

```bash
docker-compose up -d
```

Wait approximately 20 seconds for services to initialize, then pull required models:

```bash
docker-compose exec ollama ollama pull llama3:8b nomic-embed-text
```

## Code Style

This project uses **Black** for code formatting and **Ruff** for linting.

### Formatting Code

Format your code before committing:

```bash
black src/ tests/
```

### Linting

Check for linting issues:

```bash
ruff check src/ tests/
```

Auto-fix linting issues where possible:

```bash
ruff check --fix src/ tests/
```

### Pre-commit Hook (Recommended)

Consider installing a pre-commit hook to automatically format and lint code:

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml with:
# - repo: https://github.com/psf/black
#   rev: 23.12.1
#   hooks:
#     - id: black
# - repo: https://github.com/astral-sh/ruff-pre-commit
#   rev: v0.1.9
#   hooks:
#     - id: ruff
#       args: [--fix]

# Install the hook
pre-commit install
```

## Testing Requirements

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src/rag_system

# Run specific test categories
pytest tests/unit/ -v -m unit
pytest tests/integration/ -v -m integration
pytest tests/e2e/ -v -m e2e

# Run specific test file
pytest tests/unit/ingestion/test_markitdown_converter.py -v
```

### Coverage Requirements

All contributions must maintain **>80% test coverage**. Check coverage with:

```bash
pytest tests/ --cov=src/rag_system --cov-report=term-missing
```

### Test-Driven Development (TDD)

We follow the TDD workflow:

1. **RED**: Write a failing test for the new feature
2. **GREEN**: Write minimal code to make the test pass
3. **REFACTOR**: Improve the code while keeping tests green

### Test Structure

- **Unit tests** (`tests/unit/`): Test individual functions and classes in isolation
- **Integration tests** (`tests/integration/`): Test interactions between components
- **End-to-end tests** (`tests/e2e/`): Test the full system through API endpoints

## Pull Request Process

### 1. Fork the Repository

Fork the repository on GitHub and clone your fork:

```bash
git clone https://github.com/<your-username>/RAG.git
cd RAG
```

### 2. Create a Feature Branch

Create a descriptive branch for your changes:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### 3. Make Your Changes

- Write clear, concise code following the project's style
- Add tests for new functionality
- Update documentation if needed
- Ensure all tests pass

### 4. Run Tests Locally

Before submitting, ensure all tests pass:

```bash
pytest tests/ -v --cov=src/rag_system
```

Verify coverage is above 80%.

### 5. Format and Lint

```bash
black src/ tests/
ruff check --fix src/ tests/
```

### 6. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: brief description of changes"
```

### 7. Push to Your Fork

```bash
git push origin feature/your-feature-name
```

### 8. Submit a Pull Request

1. Go to the original repository on GitHub
2. Click "New Pull Request"
3. Select your feature branch
4. Provide a clear description of your changes
5. Link any related issues

### Pull Request Checklist

Before submitting, ensure:

- [ ] All tests pass locally
- [ ] Test coverage is >80%
- [ ] Code is formatted with Black
- [ ] No linting errors from Ruff
- [ ] Documentation is updated (if applicable)
- [ ] Commit messages are clear and descriptive
- [ ] PR description explains the what and why

## Getting Help

If you need help:

- Check existing issues for similar problems
- Read the [README.md](README.md) for project documentation
- Ask questions in GitHub Discussions (if available)

## Code of Conduct

Be respectful and constructive in all interactions. We welcome contributors from all backgrounds and experience levels.
