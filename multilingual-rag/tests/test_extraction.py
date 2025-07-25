import pytest
from src.text_extraction import extract_text_from_pdf

def test_extract_text_from_pdf():
    """Test PDF text extraction."""
    try:
        pages = extract_text_from_pdf("data/HSC26-Bangla1st-Paper.pdf")
        assert isinstance(pages, list)
        assert len(pages) > 0
    except FileNotFoundError:
        pytest.skip("PDF file not found")