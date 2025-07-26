"""
Text cleaning and preprocessing utilities for Bengali and English text.

This module provides comprehensive text cleaning specifically designed
for Bengali literature with OCR error correction capabilities.
"""

import re
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class TextCleaner:
    """
    Comprehensive text cleaning and preprocessing for Bengali and English text.
    
    This class handles OCR error correction, text normalization, and
    intelligent text segmentation for multilingual content.
    """
    
    def __init__(self):
        """Initialize the text cleaner with language-specific rules."""
        # Bengali to English digit mapping
        self.bn_to_en_digits = str.maketrans('০১২৩৪৫৬৭৮৯', '0123456789')
        
        # Common OCR errors in Bengali text and their corrections
        self.ocr_corrections = {
            # Common conjunct consonant corrections
            'র্': 'র্',
            'ু': 'ু',
            'ূ': 'ূ',
            '়': '়',
            'ৃ': 'ৃ',
            'ে': 'ে',
            'ো': 'ো',
            'ৌ': 'ৌ',
            # Add more corrections based on your specific OCR errors
        }
        
        # Bengali sentence enders
        self.bengali_sentence_enders = ['।', '৷', '؟']
        
        # English sentence enders
        self.english_sentence_enders = ['.', '!', '?', ';']
        
        # Combined sentence enders for splitting
        self.all_sentence_enders = self.bengali_sentence_enders + self.english_sentence_enders
        
    def clean_text(self, text: str, aggressive_cleaning: bool = False) -> str:
        """
        Clean and normalize text with optional aggressive cleaning.
        
        Args:
            text (str): Raw text to clean
            aggressive_cleaning (bool): Whether to apply aggressive cleaning rules
            
        Returns:
            str: Cleaned and normalized text
        """
        if not text or not text.strip():
            return ""
        
        try:
            # Step 1: Basic whitespace normalization
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Step 2: Convert Bengali digits to English for consistency
            text = text.translate(self.bn_to_en_digits)
            
            # Step 3: Fix common OCR errors
            text = self._fix_ocr_errors(text)
            
            # Step 4: Normalize punctuation
            text = self._normalize_punctuation(text)
            
            # Step 5: Remove unwanted characters (aggressive cleaning)
            if aggressive_cleaning:
                text = self._aggressive_character_filtering(text)
            else:
                text = self._basic_character_filtering(text)
            
            # Step 6: Final whitespace cleanup
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
            
        except Exception as e:
            logger.warning(f"Error cleaning text: {str(e)}")
            return text.strip()
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors in Bengali text."""
        for incorrect, correct in self.ocr_corrections.items():
            text = text.replace(incorrect, correct)
        return text
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        # Fix multiple consecutive punctuation marks
        text = re.sub(r'[।]{2,}', '।', text)  # Bengali danda
        text = re.sub(r'[\.]{2,}', '.', text)  # English period
        text = re.sub(r'[!]{2,}', '!', text)   # Exclamation
        text = re.sub(r'[?]{2,}', '?', text)   # Question mark
        
        return text
    
    def _basic_character_filtering(self, text: str) -> str:
        """Basic character filtering to remove unwanted symbols."""
        # Keep Bengali, English, numbers, and basic punctuation
        # Bengali range: \u0980-\u09FF
        # English range: \u0020-\u007F
        pattern = r'[^\u0980-\u09FF\u0020-\u007F\s]'
        text = re.sub(pattern, '', text)
        return text
    
    def _aggressive_character_filtering(self, text: str) -> str:
        """Aggressive character filtering for cleaner output."""
        # More restrictive filtering
        # Keep only letters, digits, and essential punctuation
        pattern = r'[^\u0980-\u09FF\u0041-\u005A\u0061-\u007A\u0030-\u0039\s।\.!?,:;()"\'-]'
        text = re.sub(pattern, '', text)
        return text
    
    def split_into_sentences(self, text: str, min_sentence_length: int = 10) -> List[str]:
        """
        Split text into sentences considering both Bengali and English patterns.
        
        Args:
            text (str): Input text to split
            min_sentence_length (int): Minimum length for a valid sentence
            
        Returns:
            List[str]: List of sentences
        """
        if not text.strip():
            return []
        
        try:
            # Create pattern for sentence splitting
            enders_pattern = '[' + ''.join(re.escape(e) for e in self.all_sentence_enders) + ']'
            
            # Split by sentence enders
            sentences = re.split(enders_pattern + r'+', text)
            
            # Clean and filter sentences
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                
                # Filter by length and content
                if (len(sentence) >= min_sentence_length and 
                    self._is_valid_sentence(sentence)):
                    cleaned_sentences.append(sentence)
            
            return cleaned_sentences
            
        except Exception as e:
            logger.warning(f"Error splitting sentences: {str(e)}")
            # Fallback: return text as single sentence
            return [text] if len(text) >= min_sentence_length else []
    
    def _is_valid_sentence(self, sentence: str) -> bool:
        """Check if a sentence is valid (not just punctuation or numbers)."""
        # Must contain at least some letters
        has_letters = bool(re.search(r'[\u0980-\u09FF\u0041-\u005A\u0061-\u007A]', sentence))
        
        # Must not be just numbers or punctuation
        not_just_numbers = not re.match(r'^\d+\s*$', sentence)
        
        return has_letters and not_just_numbers
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 500, 
        overlap: int = 50,
        preserve_sentences: bool = True
    ) -> List[str]:
        """
        Split text into overlapping chunks with intelligent boundary detection.
        
        Args:
            text (str): Input text to chunk
            chunk_size (int): Maximum characters per chunk
            overlap (int): Number of overlapping characters between chunks
            preserve_sentences (bool): Whether to try preserving sentence boundaries
            
        Returns:
            List[str]: List of text chunks
        """
        if not text.strip():
            return []
        
        if len(text) <= chunk_size:
            return [text.strip()]
        
        try:
            chunks = []
            start = 0
            
            while start < len(text):
                end = start + chunk_size
                
                # If this would be the last chunk, include all remaining text
                if end >= len(text):
                    chunk = text[start:].strip()
                    if chunk:
                        chunks.append(chunk)
                    break
                
                # Try to find a good break point
                if preserve_sentences:
                    end = self._find_sentence_boundary(text, start, end)
                else:
                    end = self._find_word_boundary(text, start, end)
                
                # Extract chunk
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)
                
                # Move start position with overlap
                start = end - overlap if overlap < (end - start) else end
            
            logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            # Fallback: simple chunking
            return self._simple_chunk(text, chunk_size, overlap)
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within the chunk range."""
        # Look for sentence enders in the last 100 characters of the chunk
        search_start = max(start, end - 100)
        
        # Find all sentence enders in the search range
        enders_in_range = []
        for ender in self.all_sentence_enders:
            pos = text.rfind(ender, search_start, end)
            if pos > start:
                enders_in_range.append(pos + 1)  # +1 to include the ender
        
        if enders_in_range:
            return max(enders_in_range)  # Return the latest sentence boundary
        
        # Fallback to word boundary
        return self._find_word_boundary(text, start, end)
    
    def _find_word_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best word boundary within the chunk range."""
        # Look for spaces in the last 50 characters
        search_start = max(start, end - 50)
        
        # Find the last space in the search range
        last_space = text.rfind(' ', search_start, end)
        
        if last_space > start:
            return last_space
        
        # If no good boundary found, return the original end
        return end
    
    def _simple_chunk(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Simple fallback chunking method."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < len(text) else len(text)
        
        return chunks
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """
        Extract key phrases from text (simple implementation).
        
        Args:
            text (str): Input text
            max_phrases (int): Maximum number of phrases to extract
            
        Returns:
            List[str]: List of key phrases
        """
        try:
            # Simple extraction based on noun phrases and important words
            sentences = self.split_into_sentences(text)
            
            key_phrases = []
            for sentence in sentences[:max_phrases]:
                # Extract first few words of each sentence as potential key phrases
                words = sentence.split()[:5]  # First 5 words
                if len(words) >= 2:
                    phrase = ' '.join(words)
                    key_phrases.append(phrase)
            
            return key_phrases[:max_phrases]
            
        except Exception as e:
            logger.warning(f"Error extracting key phrases: {str(e)}")
            return []
