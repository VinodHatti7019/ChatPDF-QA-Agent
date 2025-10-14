"""RAG (Retrieval-Augmented Generation) Package

This package provides modular components for building a RAG pipeline:
- Document loading and parsing
- Text chunking strategies
- Embedding generation
- Vector storage and retrieval
- Answer generation with citations
"""

__version__ = "0.1.0"
__author__ = "Vinod Hatti"

from .loaders import PDFLoader
from .chunker import TextChunker
from .embedder import Embedder
from .vector import VectorStore
from .retriever import Retriever
from .generator import Generator
from .pipeline import RAGPipeline
from .citation import CitationManager

__all__ = [
    "PDFLoader",
    "TextChunker",
    "Embedder",
    "VectorStore",
    "Retriever",
    "Generator",
    "RAGPipeline",
    "CitationManager",
]
