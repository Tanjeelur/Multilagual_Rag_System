"""
Long-term memory implementation using vector database storage.

This module manages persistent document storage and retrieval using
vector embeddings for semantic similarity search.
"""

from typing import List, Dict, Any, Optional
import logging
from ..vectorization.vector_store import VectorStore
from ..vectorization.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)

class LongTermMemory:
    """
    Manages long-term memory storage using vector database for document persistence.
    
    This class handles the storage and retrieval of documents with semantic search
    capabilities, maintaining persistent knowledge across sessions.
    """
    
    def __init__(
        self, 
        db_config: Dict[str, Any], 
        embedding_model: str,
        vector_dimension: int = 384
    ):
        """
        Initialize long-term memory with database and embedding configurations.
        
        Args:
            db_config (Dict[str, Any]): Database configuration
            embedding_model (str): Name of the embedding model to use
            vector_dimension (int): Dimension of embedding vectors
        """
        self.db_config = db_config
        self.embedding_model_name = embedding_model
        self.vector_dimension = vector_dimension
        
        # Initialize components
        logger.info("Initializing long-term memory components")
        
        try:
            # Initialize vector store
            self.vector_store = VectorStore(db_config, vector_dimension)
            
            # Initialize embedding generator
            self.embedding_generator = EmbeddingGenerator(embedding_model)
            
            # Verify embedding dimensions match
            actual_dim = self.embedding_generator.get_embedding_dimension()
            if actual_dim != vector_dimension:
                logger.warning(
                    f"Embedding dimension mismatch: expected {vector_dimension}, "
                    f"got {actual_dim}. Updating vector store dimension."
                )
                self.vector_dimension = actual_dim
            
            logger.info("Long-term memory initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing long-term memory: {str(e)}")
            raise
    
    def store_documents(
        self, 
        documents: List[str], 
        metadata: Optional[List[Dict]] = None,
        source: str = "default",
        language: str = "mixed",
        batch_size: int = 50
    ) -> List[int]:
        """
        Store documents in long-term memory with semantic indexing.
        
        Args:
            documents (List[str]): List of document texts to store
            metadata (Optional[List[Dict]]): Optional metadata for each document
            source (str): Source identifier for the documents
            language (str): Language code for the documents
            batch_size (int): Batch size for processing
            
        Returns:
            List[int]: List of stored document IDs
        """
        if not documents:
            logger.warning("No documents provided for storage")
            return []
        
        try:
            logger.info(f"Storing {len(documents)} documents in long-term memory")
            
            # Generate embeddings for all documents
            logger.info("Generating embeddings for documents")
            embeddings = self.embedding_generator.encode_text(
                documents, 
                batch_size=batch_size,
                show_progress=True
            )
            
            # Prepare metadata
            if metadata is None:
                metadata = [{"stored_at": "auto-generated"} for _ in documents]
            
            # Add additional metadata
            enhanced_metadata = []
            for i, meta in enumerate(metadata):
                enhanced_meta = meta.copy() if meta else {}
                enhanced_meta.update({
                    "chunk_id": i,
                    "total_chunks": len(documents),
                    "char_count": len(documents[i]),
                    "word_count": len(documents[i].split())
                })
                enhanced_metadata.append(enhanced_meta)
            
            # Store in vector database
            document_ids = self.vector_store.add_documents(
                texts=documents,
                embeddings=embeddings,
                metadata=enhanced_metadata,
                source=source,
                language=language,
                batch_size=batch_size
            )
            
            logger.info(f"Successfully stored {len(document_ids)} documents")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            raise
    
    def retrieve_relevant_documents(
        self, 
        query: str, 
        k: int = 5,
        source_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
        similarity_threshold: float = 0.3,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a given query using semantic search.
        
        Args:
            query (str): Search query
            k (int): Number of documents to retrieve
            source_filter (Optional[str]): Filter by document source
            language_filter (Optional[str]): Filter by language
            similarity_threshold (float): Minimum similarity threshold
            include_metadata (bool): Whether to include document metadata
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents with similarity scores
        """
        if not query.strip():
            logger.warning("Empty query provided for document retrieval")
            return []
        
        try:
            logger.info(f"Retrieving relevant documents for query: '{query[:50]}...'")
            
            # Generate query embedding
            query_embedding = self.embedding_generator.encode_single_text(query)
            
            # Search for similar documents
            similar_docs = self.vector_store.similarity_search(
                query_embedding=query_embedding,
                k=k,
                source_filter=source_filter,
                language_filter=language_filter,
                similarity_threshold=similarity_threshold
            )
            
            # Process results
            results = []
            for doc in similar_docs:
                result = {
                    "content": doc["content"],
                    "similarity_score": doc["similarity_score"],
                    "source": doc["source"],
                    "language": doc["language"]
                }
                
                if include_metadata:
                    result["metadata"] = doc["metadata"]
                    result["chunk_index"] = doc["chunk_index"]
                    result["document_id"] = doc["id"]
                
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored documents.
        
        Returns:
            Dict[str, Any]: Memory statistics
        """
        try:
            total_docs = self.vector_store.get_document_count()
            sources = self.vector_store.get_sources()
            
            stats = {
                "total_documents": total_docs,
                "total_sources": len(sources),
                "sources": sources,
                "embedding_model": self.embedding_model_name,
                "vector_dimension": self.vector_dimension
            }
            
            # Get document count per source
            source_counts = {}
            for source in sources:
                count = self.vector_store.get_document_count(source_filter=source)
                source_counts[source] = count
            
            stats["documents_per_source"] = source_counts
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory statistics: {str(e)}")
            return {"error": str(e)}
    
    def clear_memory(self) -> bool:
        """
        Clear all stored documents from memory.
        
        Returns:
            bool: True if successful
        """
        try:
            success = self.vector_store.clear_all_documents()
            if success:
                logger.info("Cleared all documents from long-term memory")
            return success
            
        except Exception as e:
            logger.error(f"Error clearing memory: {str(e)}")
            return False
    
    def close(self) -> None:
        """Close long-term memory connections."""
        try:
            if hasattr(self, 'vector_store'):
                self.vector_store.close()
            logger.info("Long-term memory connections closed")
        except Exception as e:
            logger.error(f"Error closing long-term memory: {str(e)}")
