"""Text chunking module with advanced strategies.

Provides multiple chunking strategies with configurable overlap and separators
for optimal RAG performance.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Chunk of text with metadata."""
    text: str
    chunk_id: str
    doc_id: str
    chunk_index: int
    metadata: Dict[str, Any]
    start_char: int = 0
    end_char: int = 0
    
    def __len__(self):
        return len(self.text)


class TextChunker:
    """Advanced text chunker with multiple strategies."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        min_chunk_size: int = 100,
    ):
        """Initialize text chunker.
        
        Args:
            chunk_size: Target size for chunks in characters
            chunk_overlap: Number of overlapping characters between chunks
            separators: List of separators to split on (in priority order)
            min_chunk_size: Minimum chunk size to avoid tiny chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        self.min_chunk_size = min_chunk_size
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        logger.info(
            f"TextChunker initialized: chunk_size={chunk_size}, "
            f"overlap={chunk_overlap}, separators={len(self.separators)}"
        )
    
    def chunk_text(self, text: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk text using recursive character splitting.
        
        Args:
            text: Text to chunk
            doc_id: Document ID for tracking
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = self._recursive_split(text, self.separators)
        
        # Create Chunk objects with proper IDs and metadata
        chunk_objects = []
        char_position = 0
        
        for idx, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < self.min_chunk_size:
                # Skip very small chunks or merge with previous
                if chunk_objects:
                    # Merge with previous chunk
                    prev_chunk = chunk_objects[-1]
                    prev_chunk.text += " " + chunk_text
                    prev_chunk.end_char = char_position + len(chunk_text)
                    char_position += len(chunk_text)
                    continue
            
            chunk_obj = Chunk(
                text=chunk_text.strip(),
                chunk_id=f"{doc_id}_chunk_{idx}",
                doc_id=doc_id,
                chunk_index=idx,
                metadata={
                    **metadata,
                    'chunk_method': 'recursive_split',
                    'original_length': len(text),
                },
                start_char=char_position,
                end_char=char_position + len(chunk_text),
            )
            chunk_objects.append(chunk_obj)
            char_position += len(chunk_text)
        
        logger.info(f"Created {len(chunk_objects)} chunks for document {doc_id}")
        return chunk_objects
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using hierarchical separators.
        
        This ensures splits happen at natural boundaries when possible.
        """
        if not separators:
            # Base case: no more separators, split by character count
            return self._split_by_length(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator:
            splits = text.split(separator)
        else:
            # Empty separator means split by character
            return self._split_by_length(text)
        
        # Rebuild chunks respecting size limits
        final_chunks = []
        current_chunk = []
        current_length = 0
        
        for split in splits:
            split_length = len(split)
            
            # If single split is too large, recursively split it
            if split_length > self.chunk_size:
                # First, add current accumulated chunk if any
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Recursively split the large piece
                sub_chunks = self._recursive_split(split, remaining_separators)
                final_chunks.extend(sub_chunks)
                continue
            
            # Check if adding this split would exceed chunk size
            if current_length + split_length + len(separator) > self.chunk_size and current_chunk:
                # Save current chunk and start new one
                chunk_text = separator.join(current_chunk)
                final_chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if self.chunk_overlap > 0:
                    # Keep last part of previous chunk for overlap
                    overlap_text = chunk_text[-self.chunk_overlap:]
                    current_chunk = [overlap_text, split]
                    current_length = len(overlap_text) + split_length + len(separator)
                else:
                    current_chunk = [split]
                    current_length = split_length
            else:
                # Add to current chunk
                current_chunk.append(split)
                current_length += split_length + len(separator)
        
        # Add remaining chunk
        if current_chunk:
            final_chunks.append(separator.join(current_chunk))
        
        return final_chunks
    
    def _split_by_length(self, text: str) -> List[str]:
        """Split text by fixed length with overlap."""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_by_sentences(self, text: str, doc_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Chunk text by sentences while respecting size limits.
        
        Args:
            text: Text to chunk
            doc_id: Document ID
            metadata: Optional metadata
            
        Returns:
            List of Chunk objects
        """
        # Use regex to split by sentences
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap (last sentence)
                if self.chunk_overlap > 0 and current_chunk:
                    current_chunk = [current_chunk[-1], sentence]
                    current_length = len(current_chunk[-2]) + sentence_length
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Convert to Chunk objects
        metadata = metadata or {}
        chunk_objects = [
            Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_sent_chunk_{idx}",
                doc_id=doc_id,
                chunk_index=idx,
                metadata={**metadata, 'chunk_method': 'sentence_based'},
            )
            for idx, chunk_text in enumerate(chunks)
        ]
        
        logger.info(f"Created {len(chunk_objects)} sentence-based chunks")
        return chunk_objects
    
    def chunk_by_paragraphs(
        self,
        text: str,
        doc_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        paragraph_separator: str = "\n\n"
    ) -> List[Chunk]:
        """Chunk text by paragraphs.
        
        Args:
            text: Text to chunk
            doc_id: Document ID
            metadata: Optional metadata
            paragraph_separator: Separator to identify paragraphs
            
        Returns:
            List of Chunk objects
        """
        paragraphs = text.split(paragraph_separator)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_length = len(para)
            
            # If paragraph is too large, split it further
            if para_length > self.chunk_size:
                if current_chunk:
                    chunks.append(paragraph_separator.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Use recursive split for large paragraph
                sub_chunks = self._recursive_split(para, self.separators[1:])
                chunks.extend(sub_chunks)
                continue
            
            if current_length + para_length > self.chunk_size and current_chunk:
                chunks.append(paragraph_separator.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        if current_chunk:
            chunks.append(paragraph_separator.join(current_chunk))
        
        # Convert to Chunk objects
        metadata = metadata or {}
        chunk_objects = [
            Chunk(
                text=chunk_text,
                chunk_id=f"{doc_id}_para_chunk_{idx}",
                doc_id=doc_id,
                chunk_index=idx,
                metadata={**metadata, 'chunk_method': 'paragraph_based'},
            )
            for idx, chunk_text in enumerate(chunks)
        ]
        
        logger.info(f"Created {len(chunk_objects)} paragraph-based chunks")
        return chunk_objects
    
    def get_chunk_stats(self, chunks: List[Chunk]) -> Dict[str, Any]:
        """Get statistics about chunks.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {}
        
        lengths = [len(chunk.text) for chunk in chunks]
        
        return {
            'total_chunks': len(chunks),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_chars': sum(lengths),
        }


if __name__ == "__main__":
    # Test chunker
    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    
    test_text = "This is a test document. " * 100
    chunks = chunker.chunk_text(test_text, "test_doc")
    
    print(f"Created {len(chunks)} chunks")
    print(f"Stats: {chunker.get_chunk_stats(chunks)}")
