import pytest
import psycopg2
from src.storage import Storage
from src.retrieval import Retriever
from src.config import DB_CONFIG, REDIS_CONFIG

def test_retrieval():
    """Test chunk retrieval."""
    try:
        storage = Storage(DB_CONFIG, REDIS_CONFIG)
        retriever = Retriever(storage.conn)
        query_embedding = [0.0] * 768  # Dummy embedding
        chunks = retriever.retrieve(query_embedding, top_k=3)
        assert isinstance(chunks, list)
    except psycopg2.Error:
        pytest.skip("Database connection failed")