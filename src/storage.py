import psycopg2
from pgvector.psycopg2 import register_vector
import redis
import uuid
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, filename='logs/rag_pipeline.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Storage:
    def __init__(self, db_config: dict, redis_config: dict):
        """
        Initialize PostgreSQL and Redis connections.
        
        Args:
            db_config (dict): PostgreSQL connection parameters.
            redis_config (dict): Redis connection parameters.
        """
        try:
            self.db_config = db_config  # Store config for reconnection if needed
            self.conn = psycopg2.connect(**db_config)
            self.conn.set_session(autocommit=False)  # Explicitly set transaction mode
            register_vector(self.conn)
            # Don't create a persistent cursor as it can cause transaction issues
            self.redis = redis.Redis(**redis_config)
            self._create_table()
            logger.info("Initialized storage with PostgreSQL and Redis")
        except Exception as e:
            logger.error(f"Storage initialization failed: {str(e)}")
            raise

    def _create_table(self):
        """Create table for storing chunks and embeddings."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id UUID PRIMARY KEY,
                        text TEXT,
                        embedding VECTOR(768),
                        page INTEGER
                    );
                """)
                self.conn.commit()
                logger.info("Created chunks table in PostgreSQL")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Table creation failed: {str(e)}")
            raise

    def store_chunks(self, chunks: List[str], embeddings: List[list], pages: List[int]):
        """
        Store chunks and their embeddings in PostgreSQL.
        
        Args:
            chunks (List[str]): List of text chunks.
            embeddings (List[list]): List of corresponding embeddings.
            pages (List[int]): List of page numbers.
        """
        try:
            # Use a transaction to ensure atomicity
            with self.conn.cursor() as cursor:
                cursor.execute("DELETE FROM chunks")  # Clear existing data
                for chunk, embedding, page in zip(chunks, embeddings, pages):
                    chunk_id = str(uuid.uuid4())
                    cursor.execute(
                        "INSERT INTO chunks (id, text, embedding, page) VALUES (%s, %s, %s, %s)",
                        (chunk_id, chunk, embedding, page)
                    )
                self.conn.commit()
                logger.info(f"Stored {len(chunks)} chunks in PostgreSQL")
        except Exception as e:
            # Rollback the transaction in case of error
            self.conn.rollback()
            logger.error(f"Chunk storage failed: {str(e)}")
            raise

    def store_query(self, query: str):
        """
        Store query in Redis for short-term memory.
        
        Args:
            query (str): Query to store.
        """
        try:
            self.redis.rpush("queries", query)
            self.redis.ltrim("queries", -5, -1)  # Keep last 5 queries
            logger.info("Stored query in Redis")
        except Exception as e:
            logger.error(f"Query storage failed: {str(e)}")
            raise

    def get_recent_queries(self) -> List[str]:
        """
        Retrieve recent queries from Redis.
        
        Returns:
            List[str]: List of recent queries.
        """
        try:
            return [q.decode() for q in self.redis.lrange("queries", -5, -1)]
        except Exception as e:
            logger.error(f"Query retrieval failed: {str(e)}")
            return []
            
    def reconnect_db(self):
        """
        Reconnect to the PostgreSQL database if the connection is closed or in error state.
        
        Returns:
            bool: True if reconnection was successful, False otherwise.
        """
        try:
            if self.conn.closed:
                logger.info("Reconnecting to PostgreSQL database...")
                self.conn = psycopg2.connect(**self.db_config)
                self.conn.set_session(autocommit=False)
                register_vector(self.conn)
                logger.info("Successfully reconnected to PostgreSQL")
                return True
            return True
        except Exception as e:
            logger.error(f"Database reconnection failed: {str(e)}")
            return False