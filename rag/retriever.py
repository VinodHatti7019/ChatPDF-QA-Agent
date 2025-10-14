"""Retriever Module - Handles document retrieval from vector store"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant documents from vector store.
    
    Args:
        vector_store: VectorStore instance
        embedder: Embedder instance
        top_k: Number of documents to retrieve
        score_threshold: Minimum similarity score threshold
    """
    
    def __init__(self, vector_store, embedder, top_k: int = 5, score_threshold: float = 0.0):
        self.vector_store = vector_store
        self.embedder = embedder
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def retrieve(self, query: str) -> List[Dict]:
        """Retrieve relevant documents for query.
        
        Args:
            query: User query string
            
        Returns:
            List of document dictionaries with text, score, and metadata
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)
        
        # Search vector store
        results = self.vector_store.search(query_embedding, k=self.top_k)
        
        # Filter by threshold and format results
        filtered_results = []
        for doc_text, score, metadata in results:
            if score >= self.score_threshold:
                filtered_results.append({
                    'text': doc_text,
                    'score': score,
                    'metadata': metadata
                })
        
        logger.info(f"Retrieved {len(filtered_results)} documents for query")
        return filtered_results
    
    def retrieve_with_reranking(self, query: str) -> List[Dict]:
        """Retrieve documents with optional reranking."""
        # Basic retrieval
        results = self.retrieve(query)
        
        # TODO: Add reranking logic here if needed
        # For now, return as-is
        return results
