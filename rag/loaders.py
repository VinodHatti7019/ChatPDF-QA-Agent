"""Document Loaders Module

Provides classes for loading and parsing PDF documents.
Supports text extraction, metadata extraction, and preprocessing.
"""

import io
import logging
from typing import List, Dict, Optional
from pathlib import Path

try:
    import PyPDF2
    import pdfplumber
except ImportError:
    PyPDF2 = None
    pdfplumber = None

logger = logging.getLogger(__name__)


class Document:
    """Represents a loaded document with text and metadata."""
    
    def __init__(self, text: str, metadata: Optional[Dict] = None):
        self.text = text
        self.metadata = metadata or {}
    
    def __repr__(self):
        return f"Document(text_length={len(self.text)}, metadata={self.metadata})"


class PDFLoader:
    """Loads and parses PDF documents.
    
    Supports multiple PDF parsing backends:
    - PyPDF2: Fast, basic text extraction
    - pdfplumber: More accurate, handles complex layouts
    
    Args:
        use_pdfplumber: If True, use pdfplumber backend (default: True)
        extract_images: If True, extract image descriptions (default: False)
    """
    
    def __init__(self, use_pdfplumber: bool = True, extract_images: bool = False):
        self.use_pdfplumber = use_pdfplumber and pdfplumber is not None
        self.extract_images = extract_images
        
        if self.use_pdfplumber and pdfplumber is None:
            logger.warning("pdfplumber not installed, falling back to PyPDF2")
            self.use_pdfplumber = False
    
    def load(self, file_path: str) -> List[Document]:
        """Load PDF from file path.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of Document objects, one per page
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        with open(file_path, 'rb') as file:
            return self.load_from_stream(file, filename=path.name)
    
    def load_from_stream(self, stream, filename: str = "unknown.pdf") -> List[Document]:
        """Load PDF from file stream or bytes.
        
        Args:
            stream: File-like object or bytes
            filename: Original filename for metadata
            
        Returns:
            List of Document objects
        """
        if self.use_pdfplumber:
            return self._load_with_pdfplumber(stream, filename)
        else:
            return self._load_with_pypdf2(stream, filename)
    
    def _load_with_pdfplumber(self, stream, filename: str) -> List[Document]:
        """Load PDF using pdfplumber backend."""
        documents = []
        
        try:
            with pdfplumber.open(stream) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    
                    metadata = {
                        "source": filename,
                        "page": page_num,
                        "total_pages": len(pdf.pages),
                    }
                    
                    documents.append(Document(text=text, metadata=metadata))
                    
            logger.info(f"Loaded {len(documents)} pages from {filename}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF with pdfplumber: {e}")
            raise
    
    def _load_with_pypdf2(self, stream, filename: str) -> List[Document]:
        """Load PDF using PyPDF2 backend."""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")
        
        documents = []
        
        try:
            pdf_reader = PyPDF2.PdfReader(stream)
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text() or ""
                
                metadata = {
                    "source": filename,
                    "page": page_num,
                    "total_pages": len(pdf_reader.pages),
                }
                
                documents.append(Document(text=text, metadata=metadata))
            
            logger.info(f"Loaded {len(documents)} pages from {filename}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF with PyPDF2: {e}")
            raise
    
    @staticmethod
    def merge_documents(documents: List[Document]) -> Document:
        """Merge multiple documents into one.
        
        Args:
            documents: List of documents to merge
            
        Returns:
            Single merged document
        """
        merged_text = "\n\n".join(doc.text for doc in documents)
        merged_metadata = {
            "sources": [doc.metadata.get("source") for doc in documents],
            "total_pages": sum(doc.metadata.get("total_pages", 1) for doc in documents),
        }
        return Document(text=merged_text, metadata=merged_metadata)
