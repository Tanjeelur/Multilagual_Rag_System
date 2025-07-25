import re
import unicodedata
import nltk
import os
from typing import List
import logging
from .bengali_tokenizer import bengali_sent_tokenize, bengali_word_tokenize

# Configure logging
try:
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename='logs/rag_pipeline.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
except Exception as e:
    # Fallback to console logging if file logging fails
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.warning(f"Could not set up file logging: {e}")

logger = logging.getLogger(__name__)

# Download NLTK data for English
try:
    nltk.download('punkt', quiet=True)
    logger.info("NLTK punkt tokenizer downloaded for English")
except Exception as e:
    logger.error(f"NLTK initialization error: {e}")

def clean_text(text: str) -> str:
    """
    Clean extracted text by normalizing and removing noise.
    
    Args:
        text (str): Raw text input.
    
    Returns:
        str: Cleaned text.
    """
    try:
        text = unicodedata.normalize('NFKC', text)
        text = re.sub(r'(.)\1+', r'\1', text)  # Remove repetitive characters
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
        logger.debug(f"Cleaned text: {text[:50]}...")
        return text
    except Exception as e:
        logger.error(f"Text cleaning failed: {str(e)}")
        raise

def chunk_text(text: str, max_words: int = 300, overlap_words: int = 50) -> List[str]:
    """
    Chunk text into paragraphs with overlap for semantic retrieval.
    
    Args:
        text (str): Input text to chunk.
        max_words (int): Maximum words per chunk.
        overlap_words (int): Number of words to overlap between chunks.
    
    Returns:
        List[str]: List of text chunks.
    """
    try:
        # Detect language and use appropriate tokenizer
        is_bengali = any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in text[:100])
        
        if is_bengali:
            logger.info("Detected Bengali text, using Bengali tokenizer")
            sentences = bengali_sent_tokenize(text)
        else:
            logger.info("Using English tokenizer")
            sentences = nltk.sent_tokenize(text)
            
        chunks = []
        current_chunk = []
        word_count = 0

        for sentence in sentences:
            # Use appropriate word tokenizer
            if is_bengali:
                words = bengali_word_tokenize(sentence)
            else:
                words = sentence.split()
                
            word_count += len(words)
            current_chunk.append(sentence)

            if word_count >= max_words:
                chunks.append(' '.join(current_chunk))
                current_chunk = current_chunk[-overlap_words:] if len(current_chunk) > overlap_words else current_chunk
                word_count = len(' '.join(current_chunk).split())

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    except Exception as e:
        logger.error(f"Text chunking failed: {str(e)}")
        raise