import psycopg2
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename='logs/rag_pipeline.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self, conn):
        """
        Initialize retriever with PostgreSQL connection.
        
        Args:
            conn: PostgreSQL connection object.
        """
        self.conn = conn
        # Don't create a persistent cursor as it can cause transaction issues
        # Instead, create a new cursor for each operation

    def retrieve(self, query_embedding: list, top_k: int = 5) -> List[Tuple]:
        """
        Retrieve top-k chunks using cosine similarity.
        
        Args:
            query_embedding (list): Embedding of the query.
            top_k (int): Number of chunks to retrieve.
        
        Returns:
            List[Tuple]: List of (id, text, page) tuples.
        """
        try:
            # Create a new cursor for this operation to avoid transaction issues
            with self.conn.cursor() as cursor:
                cursor.execute(
                    "SELECT id, text, page FROM chunks ORDER BY embedding <-> %s::vector LIMIT %s",
                    (query_embedding, top_k)
                )
                results = cursor.fetchall()
                logger.info(f"Retrieved {len(results)} chunks for query")
                return results
        except Exception as e:
            # Rollback the transaction in case of error
            self.conn.rollback()
            logger.error(f"Retrieval failed: {str(e)}")
            raise