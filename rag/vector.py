"""Vector Storage Module

Provides vector database functionality for storing and retrieving embeddings.
Supports FAISS and Chroma backends.
"""

import logging
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    import chromadb
except ImportError:
    chromadb = None

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector database for storing and searching embeddings.
    
    Args:
        dimension: Dimension of embedding vectors
        backend: Vector database backend ('faiss' or 'chroma')
        index_type: FAISS index type (if using FAISS)
        collection_name: Collection name (if using Chroma)
    """
    
    def __init__(
        self,
        dimension: int,
        backend: str = "faiss",
        index_type: str = "flat",
        collection_name: str = "documents"
    ):
        self.dimension = dimension
        self.backend = backend.lower()
        self.index_type = index_type
        self.collection_name = collection_name
        
        self.index = None
        self.documents = []
        self.metadatas = []
        
        self._initialize_backend()
    
    def _initialize_backend(self):
        """Initialize the vector database backend."""
        if self.backend == "faiss":
            self._initialize_faiss()
        elif self.backend == "chroma":
            self._initialize_chroma()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _initialize_faiss(self):
        """Initialize FAISS index."""
        if faiss is None:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
        
        if self.index_type == "flat":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            raise ValueError(f"Unknown FAISS index type: {self.index_type}")
        
        logger.info(f"Initialized FAISS {self.index_type} index with dimension {self.dimension}")
    
    def _initialize_chroma(self):
        """Initialize Chroma collection."""
        if chromadb is None:
            raise ImportError("ChromaDB not installed. Install with: pip install chromadb")
        
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.info(f"Initialized Chroma collection: {self.collection_name}")
    
    def add_documents(
        self,
        embeddings: List[np.ndarray],
        documents: List[str],
        metadatas: Optional[List[Dict]] = None
    ):
        """Add documents with their embeddings to the vector store.
        
        Args:
            embeddings: List of embedding vectors
            documents: List of document texts
            metadatas: List of metadata dictionaries
        """
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        metadatas = metadatas or [{} for _ in documents]
        
        if self.backend == "faiss":
            self._add_to_faiss(embeddings, documents, metadatas)
        elif self.backend == "chroma":
            self._add_to_chroma(embeddings, documents, metadatas)
    
    def _add_to_faiss(self, embeddings: List[np.ndarray], documents: List[str], metadatas: List[Dict]):
        """Add documents to FAISS index."""
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Train index if needed (for IVF)
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            self.index.train(embeddings_array)
        
        # Add to index
        self.index.add(embeddings_array)
        
        # Store documents and metadata
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        
        logger.info(f"Added {len(documents)} documents to FAISS index (total: {len(self.documents)})")
    
    def _add_to_chroma(self, embeddings: List[np.ndarray], documents: List[str], metadatas: List[Dict]):
        """Add documents to Chroma collection."""
        # Generate IDs
        start_id = len(self.documents)
        ids = [f"doc_{start_id + i}" for i in range(len(documents))]
        
        # Convert embeddings to list of lists
        embeddings_list = [emb.tolist() for emb in embeddings]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings_list,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        self.documents.extend(documents)
        logger.info(f"Added {len(documents)} documents to Chroma (total: {len(self.documents)})")
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """Search for most similar documents.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of (document, score, metadata) tuples
        """
        if self.backend == "faiss":
            return self._search_faiss(query_embedding, k)
        elif self.backend == "chroma":
            return self._search_chroma(query_embedding, k)
    
    def _search_faiss(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float, Dict]]:
        """Search FAISS index."""
        if len(self.documents) == 0:
            logger.warning("No documents in index")
            return []
        
        # Ensure query is 2D array
        query_array = np.array([query_embedding]).astype('float32')
        
        # Search
        k = min(k, len(self.documents))
        distances, indices = self.index.search(query_array, k)
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append((
                    self.documents[idx],
                    float(dist),
                    self.metadatas[idx]
                ))
        
        return results
    
    def _search_chroma(self, query_embedding: np.ndarray, k: int) -> List[Tuple[str, float, Dict]]:
        """Search Chroma collection."""
        query_list = [query_embedding.tolist()]
        
        results = self.collection.query(
            query_embeddings=query_list,
            n_results=k
        )
        
        # Build results
        output = []
        for i in range(len(results['documents'][0])):
            output.append((
                results['documents'][0][i],
                results['distances'][0][i],
                results['metadatas'][0][i] if results['metadatas'] else {}
            ))
        
        return output
    
    def save(self, path: str):
        """Save vector store to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.backend == "faiss":
            # Save FAISS index
            faiss.write_index(self.index, str(path / "index.faiss"))
            
            # Save documents and metadata
            with open(path / "documents.pkl", 'wb') as f:
                pickle.dump({'documents': self.documents, 'metadatas': self.metadatas}, f)
            
            logger.info(f"Saved FAISS index to {path}")
        
        elif self.backend == "chroma":
            logger.info("Chroma persists automatically")
    
    def load(self, path: str):
        """Load vector store from disk."""
        path = Path(path)
        
        if self.backend == "faiss":
            # Load FAISS index
            self.index = faiss.read_index(str(path / "index.faiss"))
            
            # Load documents and metadata
            with open(path / "documents.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadatas = data['metadatas']
            
            logger.info(f"Loaded FAISS index from {path} with {len(self.documents)} documents")
        
        elif self.backend == "chroma":
            logger.info("Chroma loads automatically from persistent storage")
    
    def get_document_count(self) -> int:
        """Get number of documents in vector store."""
        return len(self.documents)
