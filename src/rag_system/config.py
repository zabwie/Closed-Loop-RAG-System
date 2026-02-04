"""Configuration module for RAG system.

Uses Pydantic BaseSettings for environment-based configuration.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Ollama configuration
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="URL for Ollama API",
    )

    # Milvus configuration
    milvus_host: str = Field(
        default="localhost",
        description="Milvus server host",
    )
    milvus_port: int = Field(
        default=19530,
        description="Milvus server port",
        ge=1,
        le=65535,
    )

    # Model configuration
    model_name: str = Field(
        default="llama2",
        description="Name of the LLM model to use",
    )
    embedding_model: str = Field(
        default="nomic-embed-text",
        description="Name of the embedding model to use",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("ollama_url")
    @classmethod
    def validate_ollama_url(cls, v: str) -> str:
        """Validate that ollama_url is a valid URL."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("ollama_url must start with http:// or https://")
        return v
