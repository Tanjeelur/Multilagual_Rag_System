"""Basic tests for the RAG system."""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import detect_language, is_valid_query
from src.data_processing.text_cleaner import TextCleaner

class TestHelpers:
    """Test utility functions."""
    
    def test_detect_language_bengali(self):
        """Test Bengali language detection."""
        bengali_text = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
        assert detect_language(bengali_text) == "bn"
    
    def test_detect_language_english(self):
        """Test English language detection."""
        english_text = "What is the main theme of the story?"
        assert detect_language(english_text) == "en"
    
    def test_is_valid_query(self):
        """Test query validation."""
        assert is_valid_query("Valid query") == True
        assert is_valid_query("") == False
        assert is_valid_query("Hi") == False
        assert is_valid_query("   ") == False

class TestTextCleaner:
    """Test text cleaning functionality."""
    
    def setUp(self):
        self.cleaner = TextCleaner()
    
    def test_clean_text(self):
        """Test text cleaning."""
        cleaner = TextCleaner()
        dirty_text = "  এটি   একটি   পরীক্ষা  "
        clean_text = cleaner.clean_text(dirty_text)
        assert clean_text == "এটি একটি পরীক্ষা"
    
    def test_split_sentences(self):
        """Test sentence splitting."""
        cleaner = TextCleaner()
        text = "এটি প্রথম বাক্য। এটি দ্বিতীয় বাক্য।"
        sentences = cleaner.split_into_sentences(text)
        assert len(sentences) == 2

if __name__ == "__main__":
    pytest.main([__file__])
