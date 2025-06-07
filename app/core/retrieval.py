from __future__ import annotations

import logging
from typing import List, Optional
from threading import Lock

from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from .config import settings


logger = logging.getLogger(__name__)


class VectorClient:
    """Singleton vector database client for Qdrant operations."""
    
    _instance: Optional[VectorClient] = None
    _lock: Lock = Lock()
    
    def __new__(cls) -> VectorClient:
        """Ensure only one instance of VectorClient exists."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the VectorClient if not already initialized."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._setup_client()
    
    def _setup_client(self) -> None:
        """Setup Qdrant client and ensure collection exists."""
        try:
            print(settings.qdrant_url)
            print(settings.qdrant_key)
            # Initialize Qdrant client
            self._qdrant_client = QdrantClient(url=settings.qdrant_url,
                                                   api_key=settings.qdrant_key)
            
            # Initialize OpenAI embeddings
            self._embeddings = OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key
            )
            
            # Collection name
            self._collection_name = "credit_docs"
            
            # Ensure collection exists
            self._ensure_collection_exists()
            
            # Initialize Langchain Qdrant wrapper
            self._vectorstore = Qdrant(
                client=self._qdrant_client,
                collection_name=self._collection_name,
                embeddings=self._embeddings
            )
            
            logger.info(f"VectorClient initialized successfully with collection '{self._collection_name}'")
            
        except Exception as e:
            logger.error(f"Failed to initialize VectorClient: {e}")
            raise
    
    def _ensure_collection_exists(self) -> None:
        """Create collection if it doesn't exist."""
        try:
            # Check if collection exists
            collections = self._qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self._collection_name not in collection_names:
                logger.info(f"Creating collection '{self._collection_name}'")
                
                # Get embedding dimension from OpenAI embeddings
                # Using a sample text to determine the embedding dimension
                sample_embedding = self._embeddings.embed_query("sample")
                embedding_dim = len(sample_embedding)
                
                # Create collection with appropriate vector configuration
                self._qdrant_client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_dim,
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self._collection_name}' created successfully")
            else:
                logger.info(f"Collection '{self._collection_name}' already exists")
                
        except UnexpectedResponse as e:
            logger.error(f"Qdrant API error while ensuring collection exists: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while ensuring collection exists: {e}")
            raise
    
    def query(self, text: str, k: int = 5) -> List[Document]:
        """
        Query the vector database for similar documents.
        
        Args:
            text: Query text to search for similar documents
            k: Number of similar documents to return (default: 5)
            
        Returns:
            List of Document objects containing similar content
        """
        try:
            if not hasattr(self, '_vectorstore'):
                raise RuntimeError("VectorClient not properly initialized")
            
            # Perform similarity search
            results = self._vectorstore.similarity_search(
                query=text,
                k=k
            )
            
            logger.debug(f"Query returned {len(results)} documents for text: '{text[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            raise
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of Document objects to add to the collection
        """
        try:
            if not hasattr(self, '_vectorstore'):
                raise RuntimeError("VectorClient not properly initialized")
            
            self._vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to collection '{self._collection_name}'")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {e}")
            raise
    
    def get_collection_info(self) -> dict:
        """
        Get information about the collection.
        
        Returns:
            Dictionary containing collection information
        """
        try:
            collection_info = self._qdrant_client.get_collection(self._collection_name)
            return {
                "name": self._collection_name,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise


# Convenience function to get the singleton instance
def get_vector_client() -> VectorClient:
    """Get the singleton VectorClient instance."""
    return VectorClient() 