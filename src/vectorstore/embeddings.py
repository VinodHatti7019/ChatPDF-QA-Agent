"""Embedding generation module using OpenAI.

Provides functionality for generating embeddings from text chunks
with batching and caching capabilities.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using OpenAI API with caching."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        cache_dir: Optional[Path] = None,
        batch_size: int = 100,
    ):
        """Initialize embedding generator.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model name for embeddings
            cache_dir: Directory to cache embeddings
            batch_size: Number of texts to embed in each batch
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Embedding cache enabled at: {self.cache_dir}")
        
        logger.info(f"EmbeddingGenerator initialized with model: {model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Check cache first
            if self.cache_dir:
                cached_embeddings = self._get_cached_embeddings(batch)
                if cached_embeddings:
                    all_embeddings.extend(cached_embeddings)
                    logger.info(f"Loaded {len(cached_embeddings)} embeddings from cache")
                    continue
            
            try:
                # Generate embeddings via API
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Cache embeddings
                if self.cache_dir:
                    self._cache_embeddings(batch, batch_embeddings)
                
                logger.info(f"Generated {len(batch_embeddings)} embeddings (batch {i//self.batch_size + 1})")
                
                # Rate limiting
                if i + self.batch_size < len(texts):
                    time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error generating embeddings: {str(e)}")
                raise
        
        return all_embeddings
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"{self.model}_{text_hash}"
    
    def _get_cached_embeddings(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Try to get embeddings from cache."""
        if not self.cache_dir:
            return None
        
        embeddings = []
        for text in texts:
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if cache_file.exists():
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                        embeddings.append(data['embedding'])
                except Exception as e:
                    logger.warning(f"Failed to load cache: {str(e)}")
                    return None
            else:
                return None
        
        return embeddings if len(embeddings) == len(texts) else None
    
    def _cache_embeddings(self, texts: List[str], embeddings: List[List[float]]):
        """Cache embeddings to disk."""
        if not self.cache_dir:
            return
        
        for text, embedding in zip(texts, embeddings):
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            try:
                with open(cache_file, 'w') as f:
                    json.dump({
                        'text': text[:100],  # Store preview
                        'embedding': embedding,
                        'model': self.model,
                    }, f)
            except Exception as e:
                logger.warning(f"Failed to cache embedding: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for this model.
        
        Returns:
            Embedding dimension
        """
        # Model dimensions
        dimensions = {
            'text-embedding-3-small': 1536,
            'text-embedding-3-large': 3072,
            'text-embedding-ada-002': 1536,
        }
        
        return dimensions.get(self.model, 1536)
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        vec1_arr = np.array(vec1)
        vec2_arr = np.array(vec2)
        
        dot_product = np.dot(vec1_arr, vec2_arr)
        norm1 = np.linalg.norm(vec1_arr)
        norm2 = np.linalg.norm(vec2_arr)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if not self.cache_dir:
            logger.warning("No cache directory configured")
            return
        
        cache_files = list(self.cache_dir.glob("*.json"))
        for cache_file in cache_files:
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Failed to delete cache file {cache_file}: {str(e)}")
        
        logger.info(f"Cleared {len(cache_files)} cached embeddings")


class BatchEmbeddingProcessor:
    """Process embeddings for multiple chunks efficiently."""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        """Initialize batch processor.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
        """
        self.generator = embedding_generator
        logger.info("BatchEmbeddingProcessor initialized")
    
    def process_chunks(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        """Process chunks and generate embeddings.
        
        Args:
            chunks: List of Chunk objects
            
        Returns:
            List of dictionaries with chunk data and embeddings
        """
        if not chunks:
            return []
        
        # Extract texts from chunks
        texts = [chunk.text for chunk in chunks]
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.generator.generate_embeddings(texts)
        
        # Combine chunks with embeddings
        results = []
        for chunk, embedding in zip(chunks, embeddings):
            results.append({
                'chunk_id': chunk.chunk_id,
                'doc_id': chunk.doc_id,
                'text': chunk.text,
                'embedding': embedding,
                'metadata': chunk.metadata,
                'chunk_index': chunk.chunk_index,
            })
        
        logger.info(f"Processed {len(results)} chunks with embeddings")
        return results


if __name__ == "__main__":
    # Test embedding generator
    generator = EmbeddingGenerator()
    
    test_texts = [
        "This is a test sentence.",
        "Another test sentence for embeddings.",
    ]
    
    embeddings = generator.generate_embeddings(test_texts)
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Similarity: {generator.cosine_similarity(embeddings[0], embeddings[1])}")
