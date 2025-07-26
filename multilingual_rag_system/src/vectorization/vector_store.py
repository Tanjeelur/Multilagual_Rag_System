"""
Vector database implementation using PostgreSQL with pgvector extension.

This module provides a comprehensive vector storage solution optimized for
multilingual document retrieval with advanced similarity search capabilities.
"""

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class VectorStore:
    """
    PostgreSQL-based vector store with pgvector extension for similarity search.
    
    This class provides comprehensive vector storage and retrieval capabilities
    optimized for multilingual document processing and semantic search.
    """
    
    def __init__(self, db_config: Dict[str, Any], vector_dimension: int = 384):
        """
        Initialize the vector store with database configuration.
        
        Args:
            db_config (Dict[str, Any]): Database connection configuration
            vector_dimension (int): Dimension of embedding vectors
        """
        self.db_config = db_config
        self.vector_dimension = vector_dimension
        self.connection = None
        
        # Connect to database and initialize tables
        self._connect()
        self._initialize_database()
        
    def _connect(self) -> None:
        """Establish connection to PostgreSQL database."""
        try:
            logger.info("Connecting to PostgreSQL database")
            self.connection = psycopg2.connect(**self.db_config)
            self.connection.autocommit = False
            logger.info("Successfully connected to PostgreSQL")
            
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def _initialize_database(self) -> None:
        """Initialize database tables and extensions."""
        try:
            with self.connection.cursor() as cursor:
                # Enable pgvector extension
                logger.info("Enabling pgvector extension")
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create documents table with vector column
                logger.info("Creating documents table")
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector({self.vector_dimension}),
                        metadata JSONB DEFAULT '{{}}',
                        source VARCHAR(255),
                        chunk_index INTEGER DEFAULT 0,
                        language VARCHAR(10) DEFAULT 'mixed',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Create indexes for better performance
                logger.info("Creating database indexes")
                
                # Vector similarity search index
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS documents_embedding_cosine_idx 
                    ON documents USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                
                # Text search indexes
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS documents_source_idx 
                    ON documents (source);
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS documents_language_idx 
                    ON documents (language);
                """)
                
                self.connection.commit()
                logger.info("Successfully initialized database tables and indexes")
                
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def add_documents(
        self, 
        texts: List[str], 
        embeddings: np.ndarray, 
        metadata: Optional[List[Dict]] = None,
        source: str = "unknown",
        language: str = "mixed",
        batch_size: int = 100
    ) -> List[int]:
        """
        Add multiple documents to the vector store with batch processing.
        
        Args:
            texts (List[str]): List of document texts
            embeddings (np.ndarray): Corresponding embeddings
            metadata (Optional[List[Dict]]): Optional metadata for each document
            source (str): Source identifier for the documents
            language (str): Language code for the documents
            batch_size (int): Batch size for database insertion
            
        Returns:
            List[int]: List of inserted document IDs
        """
        if len(texts) != len(embeddings):
            raise ValueError("Number of texts must match number of embeddings")
        
        if metadata and len(metadata) != len(texts):
            raise ValueError("Number of metadata entries must match number of texts")
        
        if not metadata:
            metadata = [{}] * len(texts)
        
        try:
            logger.info(f"Adding {len(texts)} documents to vector store")
            
            inserted_ids = []
            
            # Process documents in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size]
                
                batch_ids = self._insert_document_batch(
                    batch_texts, 
                    batch_embeddings, 
                    batch_metadata,
                    source,
                    language,
                    i  # chunk_index_offset
                )
                
                inserted_ids.extend(batch_ids)
                
                logger.debug(f"Inserted batch {i//batch_size + 1}: {len(batch_ids)} documents")
            
            self.connection.commit()
            logger.info(f"Successfully added {len(inserted_ids)} documents")
            
            return inserted_ids
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error adding documents: {str(e)}")
            raise
    
    def _insert_document_batch(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: List[Dict],
        source: str,
        language: str,
        chunk_index_offset: int
    ) -> List[int]:
        """Insert a batch of documents into the database."""
        try:
            with self.connection.cursor() as cursor:
                # Prepare data for batch insertion
                values = []
                for idx, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
                    values.append((
                        text,
                        embedding.tolist(),  # Convert numpy array to list
                        json.dumps(meta),
                        source,
                        chunk_index_offset + idx,  # chunk_index
                        language,
                        datetime.now(),  # created_at
                        datetime.now()   # updated_at
                    ))
                
                # Execute batch insertion and get IDs
                insert_query = """
                    INSERT INTO documents 
                    (content, embedding, metadata, source, chunk_index, language, created_at, updated_at)
                    VALUES %s
                    RETURNING id;
                """
                
                execute_values(cursor, insert_query, values, fetch=True)
                inserted_ids = [row[0] for row in cursor.fetchall()]
                
                return inserted_ids
                
        except Exception as e:
            logger.error(f"Error in batch insertion: {str(e)}")
            raise
    
    def similarity_search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        source_filter: Optional[str] = None,
        language_filter: Optional[str] = None,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search with optional filtering.
        
        Args:
            query_embedding (np.ndarray): Query embedding vector
            k (int): Number of results to return
            source_filter (Optional[str]): Filter by source
            language_filter (Optional[str]): Filter by language
            similarity_threshold (float): Minimum similarity threshold
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with metadata
        """
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Build the query with optional filters
                query = """
                    SELECT 
                        id,
                        content,
                        metadata,
                        source,
                        chunk_index,
                        language,
                        created_at,
                        1 - (embedding <=> %s) as similarity_score
                    FROM documents
                    WHERE 1 = 1
                """
                
                params = [query_embedding.tolist()]
                
                # Add filters
                if source_filter:
                    query += " AND source = %s"
                    params.append(source_filter)
                
                if language_filter:
                    query += " AND language = %s"
                    params.append(language_filter)
                
                # Add similarity threshold
                if similarity_threshold > 0:
                    query += f" AND (1 - (embedding <=> %s)) >= %s"
                    params.extend([query_embedding.tolist(), similarity_threshold])
                
                # Order by similarity and limit
                query += " ORDER BY embedding <=> %s LIMIT %s"
                params.extend([query_embedding.tolist(), k])
                
                cursor.execute(query, params)
                results = cursor.fetchall()
                
                # Convert to list of dictionaries
                similar_docs = []
                for row in results:
                    doc = dict(row)
                    similar_docs.append(doc)
                
                logger.debug(f"Found {len(similar_docs)} similar documents")
                return similar_docs
                
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def get_document_count(self, source_filter: Optional[str] = None) -> int:
        """
        Get the total number of documents in the store.
        
        Args:
            source_filter (Optional[str]): Optional source filter
            
        Returns:
            int: Number of documents
        """
        try:
            with self.connection.cursor() as cursor:
                if source_filter:
                    cursor.execute("SELECT COUNT(*) FROM documents WHERE source = %s", (source_filter,))
                else:
                    cursor.execute("SELECT COUNT(*) FROM documents")
                
                count = cursor.fetchone()[0]
                return count
                
        except Exception as e:
            logger.error(f"Error getting document count: {str(e)}")
            return 0
    
    def get_sources(self) -> List[str]:
        """
        Get list of all sources in the database.
        
        Returns:
            List[str]: List of unique sources
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("SELECT DISTINCT source FROM documents ORDER BY source")
                sources = [row[0] for row in cursor.fetchall()]
                return sources
                
        except Exception as e:
            logger.error(f"Error getting sources: {str(e)}")
            return []
    
    def clear_all_documents(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            bool: True if successful
        """
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DELETE FROM documents")
                deleted_count = cursor.rowcount
                
                # Reset sequence
                cursor.execute("ALTER SEQUENCE documents_id_seq RESTART WITH 1")
                
                self.connection.commit()
                logger.info(f"Cleared all {deleted_count} documents from vector store")
                
                return True
                
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error clearing documents: {str(e)}")
            return False
    
    def close(self) -> None:
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
