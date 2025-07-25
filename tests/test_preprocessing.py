import pytest
from src.preprocessing import clean_text, chunk_text

def test_clean_text():
    """Test text cleaning."""
    text = "ক্ষ ক্ষ ক্ষ   Hello   World"
    cleaned = clean_text(text)
    assert cleaned == "ক্ষ Hello World"

def test_chunk_text():
    """Test text chunking."""
    text = "এটি একটি পরীক্ষা। এটি আরেকটি বাক্য। " * 50
    chunks = chunk_text(text, max_words=10, overlap_words=2)
    assert len(chunks) > 0
    assert all(len(chunk.split()) <= 10 for chunk in chunks)