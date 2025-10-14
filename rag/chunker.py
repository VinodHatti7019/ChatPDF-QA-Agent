"""Text Chunking Module

Provides strategies for splitting documents into smaller chunks
for embedding and retrieval. Supports multiple chunking strategies.
"""

import re
import logging
from typing import List, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    metadata: Dict
    chunk_id: Optional[int] = None
    
    def __repr__(self):
        return f"Chunk(id={self.chunk_id}, length={len(self.text)})"


class TextChunker:
    """Splits text into overlapping chunks.
    
    Args:
        chunk_size: Maximum size of each chunk (in characters)
        chunk_overlap: Number of characters to overlap between chunks
        separator: Text separator to split on (default: newline)
        keep_separator: Whether to keep the separator in chunks
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n",
        keep_separator: bool = True
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.keep_separator = keep_separator
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """Split text into chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Split by separator first
        splits = self._split_text(text)
        
        # Merge splits into chunks of appropriate size
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for split in splits:
            split_length = len(split)
            
            # If single split exceeds chunk_size, split it further
            if split_length > self.chunk_size:
                # Save current chunk if it exists
                if current_chunk:
                    chunk_text = self._join_splits(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, metadata, chunk_id))
                    chunk_id += 1
                    current_chunk = []
                    current_length = 0
                
                # Split large text by character
                for i in range(0, len(split), self.chunk_size - self.chunk_overlap):
                    sub_split = split[i:i + self.chunk_size]
                    chunks.append(self._create_chunk(sub_split, metadata, chunk_id))
                    chunk_id += 1
            
            # If adding split would exceed chunk_size, save current chunk
            elif current_length + split_length > self.chunk_size:
                if current_chunk:
                    chunk_text = self._join_splits(current_chunk)
                    chunks.append(self._create_chunk(chunk_text, metadata, chunk_id))
                    chunk_id += 1
                
                # Start new chunk with overlap from previous chunk
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = [overlap_text] if overlap_text else []
                current_length = len(overlap_text) if overlap_text else 0
                
                current_chunk.append(split)
                current_length += split_length
            
            # Otherwise, add split to current chunk
            else:
                current_chunk.append(split)
                current_length += split_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = self._join_splits(current_chunk)
            chunks.append(self._create_chunk(chunk_text, metadata, chunk_id))
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def _split_text(self, text: str) -> List[str]:
        """Split text by separator."""
        if self.keep_separator:
            # Split but keep separator
            splits = text.split(self.separator)
            # Add separator back except for last split
            return [s + self.separator for s in splits[:-1]] + [splits[-1]]
        else:
            return text.split(self.separator)
    
    def _join_splits(self, splits: List[str]) -> str:
        """Join splits back into text."""
        if self.keep_separator:
            return "".join(splits)
        else:
            return self.separator.join(splits)
    
    def _get_overlap_text(self, splits: List[str]) -> str:
        """Get overlap text from end of previous chunk."""
        if not splits:
            return ""
        
        full_text = self._join_splits(splits)
        if len(full_text) <= self.chunk_overlap:
            return full_text
        
        return full_text[-self.chunk_overlap:]
    
    def _create_chunk(self, text: str, metadata: Dict, chunk_id: int) -> Chunk:
        """Create a Chunk object with metadata."""
        chunk_metadata = metadata.copy()
        chunk_metadata['chunk_id'] = chunk_id
        chunk_metadata['chunk_size'] = len(text)
        
        return Chunk(text=text, metadata=chunk_metadata, chunk_id=chunk_id)


class SemanticChunker:
    """Chunks text based on semantic boundaries (sentences, paragraphs).
    
    Args:
        chunk_size: Target size for each chunk
        chunk_overlap: Overlap between chunks
    """
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Chunk]:
        """Split text at sentence boundaries."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        metadata = metadata or {}
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = chunk_id
                chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata, chunk_id=chunk_id))
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in overlap_sentences)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_id'] = chunk_id
            chunks.append(Chunk(text=chunk_text, metadata=chunk_metadata, chunk_id=chunk_id))
        
        logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap from end of chunk."""
        overlap_text = []
        overlap_length = 0
        
        for sentence in reversed(sentences):
            if overlap_length + len(sentence) <= self.chunk_overlap:
                overlap_text.insert(0, sentence)
                overlap_length += len(sentence)
            else:
                break
        
        return overlap_text
