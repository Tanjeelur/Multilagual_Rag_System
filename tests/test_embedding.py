import pytest
from src.embedding import Embedder

def test_embedder():
    """Test embedding generation."""
    embedder = Embedder()
    texts = ["এটি একটি পরীক্ষা", "This is a test"]
    embeddings = embedder.embed(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 768  # LaBSE embedding dimension