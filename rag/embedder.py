"""Embeddings Module

Provides embedding generation for text chunks using various models.
Supports OpenAI, HuggingFace, and Sentence Transformers.
"""

import logging
from typing import List, Optional, Union
import numpy as np

try:
    import openai
except ImportError:
    openai = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings for text using various models.
    
    Args:
        model_name: Name of the embedding model to use
        provider: Embedding provider ('openai', 'huggingface', 'sentence-transformers')
        api_key: API key for provider (if required)
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        provider: str = "openai",
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.provider = provider.lower()
        self.api_key = api_key
        self.model = None
        self.embedding_dim = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model based on provider."""
        if self.provider == "openai":
            if openai is None:
                raise ImportError("OpenAI not installed. Install with: pip install openai")
            
            if self.api_key:
                openai.api_key = self.api_key
            
            # Set embedding dimension for known models
            if "ada-002" in self.model_name:
                self.embedding_dim = 1536
            elif "ada-001" in self.model_name:
                self.embedding_dim = 1024
            else:
                self.embedding_dim = 1536  # Default
            
            logger.info(f"Initialized OpenAI embedder with model {self.model_name}")
        
        elif self.provider in ["sentence-transformers", "huggingface"]:
            if SentenceTransformer is None:
                raise ImportError(
                    "Sentence Transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(
                f"Initialized Sentence Transformer with model {self.model_name}, "
                f"dimension {self.embedding_dim}"
            )
        
        else:
            raise ValueError(
                f"Unknown provider: {self.provider}. "
                f"Supported: 'openai', 'huggingface', 'sentence-transformers'"
            )
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts
        non_empty_texts = [t for t in texts if t and t.strip()]
        if not non_empty_texts:
            logger.warning("All texts are empty, returning empty list")
            return []
        
        if self.provider == "openai":
            return self._embed_with_openai(non_empty_texts)
        elif self.provider in ["sentence-transformers", "huggingface"]:
            return self._embed_with_sentence_transformers(non_empty_texts)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _embed_with_openai(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using OpenAI API."""
        try:
            response = openai.Embedding.create(
                input=texts,
                model=self.model_name
            )
            
            embeddings = [np.array(item['embedding']) for item in response['data']]
            logger.info(f"Generated {len(embeddings)} embeddings with OpenAI")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {e}")
            raise
    
    def _embed_with_sentence_transformers(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using Sentence Transformers."""
        try:
            embeddings = self.model.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            # Ensure embeddings is a list of arrays
            if isinstance(embeddings, np.ndarray):
                embeddings = [embeddings[i] for i in range(len(embeddings))]
            
            logger.info(f"Generated {len(embeddings)} embeddings with Sentence Transformers")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating Sentence Transformer embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this embedder."""
        return self.embedding_dim
    
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
