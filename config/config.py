"""Configuration management for ChatPDF QA Agent.

This module handles all configuration settings using Pydantic for validation
and python-dotenv for environment variable management.
"""

import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = Field(default="text-embedding-3-small")
    chat_model: str = Field(default="gpt-4-turbo-preview")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, gt=0)

    @validator("api_key")
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("OPENAI_API_KEY must be set in environment variables")
        return v


class PineconeConfig(BaseModel):
    """Pinecone vector database configuration."""
    api_key: str = Field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    environment: str = Field(default_factory=lambda: os.getenv("PINECONE_ENV", "gcp-starter"))
    index_name: str = Field(default="chatpdf-qa-agent")
    dimension: int = Field(default=1536)
    metric: str = Field(default="cosine")

    @validator("api_key")
    def validate_api_key(cls, v):
        if not v:
            raise ValueError("PINECONE_API_KEY must be set in environment variables")
        return v


class ChunkingConfig(BaseModel):
    """Document chunking configuration."""
    chunk_size: int = Field(default=1000, gt=0, le=4000)
    chunk_overlap: int = Field(default=200, ge=0)
    separators: List[str] = Field(default=["\n\n", "\n", ". ", " ", ""])

    @validator("chunk_overlap")
    def validate_overlap(cls, v, values):
        if "chunk_size" in values and v >= values["chunk_size"]:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class RetrievalConfig(BaseModel):
    """Retrieval configuration for hybrid search."""
    top_k: int = Field(default=5, gt=0, le=20)
    vector_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    min_confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    enable_reranking: bool = Field(default=True)

    @validator("bm25_weight")
    def validate_weights(cls, v, values):
        if "vector_weight" in values and abs(values["vector_weight"] + v - 1.0) > 0.01:
            raise ValueError("vector_weight + bm25_weight must equal 1.0")
        return v


class AppConfig(BaseModel):
    """Application-level configuration."""
    app_name: str = Field(default="ChatPDF QA Agent")
    upload_dir: Path = Field(default=Path("data/uploads"))
    cache_dir: Path = Field(default=Path("data/cache"))
    max_file_size_mb: int = Field(default=50, gt=0)
    supported_formats: List[str] = Field(default=["pdf", "docx", "txt"])
    enable_conversation_memory: bool = Field(default=True)
    max_conversation_history: int = Field(default=10, gt=0)
    enable_source_citations: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    def __init__(self, **data):
        super().__init__(**data)
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


class Config(BaseModel):
    """Master configuration class."""
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    pinecone: PineconeConfig = Field(default_factory=PineconeConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    app: AppConfig = Field(default_factory=AppConfig)

    class Config:
        arbitrary_types_allowed = True


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the singleton configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Reload configuration from environment variables."""
    global _config
    load_dotenv(override=True)
    _config = Config()
    return _config


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print(f"Configuration loaded successfully!")
    print(f"App Name: {config.app.app_name}")
    print(f"Chunk Size: {config.chunking.chunk_size}")
    print(f"Top K: {config.retrieval.top_k}")
