"""Tests for configuration module."""

import pytest
from pydantic import ValidationError


def test_loads_defaults():
    """Test that config loads with default values."""
    from rag_system.config import Settings

    settings = Settings()

    assert settings.ollama_url == "http://localhost:11434"
    assert settings.milvus_host == "localhost"
    assert settings.milvus_port == 19530
    assert settings.model_name == "llama2"
    assert settings.embedding_model == "nomic-embed-text"


def test_loads_from_env():
    """Test that config loads from environment variables."""
    import os
    from rag_system.config import Settings

    os.environ["OLLAMA_URL"] = "http://custom-ollama:11434"
    os.environ["MILVUS_HOST"] = "custom-milvus"
    os.environ["MILVUS_PORT"] = "19531"
    os.environ["MODEL_NAME"] = "mistral"
    os.environ["EMBEDDING_MODEL"] = "custom-embed"

    settings = Settings()

    assert settings.ollama_url == "http://custom-ollama:11434"
    assert settings.milvus_host == "custom-milvus"
    assert settings.milvus_port == 19531
    assert settings.model_name == "mistral"
    assert settings.embedding_model == "custom-embed"

    # Clean up
    del os.environ["OLLAMA_URL"]
    del os.environ["MILVUS_HOST"]
    del os.environ["MILVUS_PORT"]
    del os.environ["MODEL_NAME"]
    del os.environ["EMBEDDING_MODEL"]


def test_validates_url_format():
    """Test that config validates URL format."""
    from rag_system.config import Settings

    with pytest.raises(ValidationError):
        Settings(ollama_url="not-a-valid-url")


def test_validates_port_range():
    """Test that config validates port range."""
    from rag_system.config import Settings

    with pytest.raises(ValidationError):
        Settings(milvus_port=99999)

    with pytest.raises(ValidationError):
        Settings(milvus_port=-1)
