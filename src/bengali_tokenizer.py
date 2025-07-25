import re
import logging
import os
from typing import List

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

# Bengali sentence ending punctuation marks
BENGALI_SENTENCE_ENDINGS = ['।', '?', '!', '॥']

# Compile regex pattern for Bengali sentence boundaries
BENGALI_SENTENCE_PATTERN_STR = r'([^' + ''.join(BENGALI_SENTENCE_ENDINGS) + r']*[' + ''.join(BENGALI_SENTENCE_ENDINGS) + r'])'
BENGALI_SENTENCE_PATTERN = re.compile(BENGALI_SENTENCE_PATTERN_STR)

def bengali_sent_tokenize(text: str) -> List[str]:
    """
    Custom sentence tokenizer for Bengali text.
    
    Args:
        text (str): Bengali text to tokenize into sentences
        
    Returns:
        List[str]: List of Bengali sentences
    """
    try:
        if not text or not isinstance(text, str):
            return []
            
        # Find all sentences using the pattern
        sentences = BENGALI_SENTENCE_PATTERN.findall(text)
        
        # If no sentences found with punctuation, return the whole text as one sentence
        if not sentences and text.strip():
            return [text.strip()]
            
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        logger.info(f"Tokenized Bengali text into {len(sentences)} sentences")
        return sentences
    except Exception as e:
        logger.error(f"Bengali sentence tokenization failed: {str(e)}")
        # Fallback to simple splitting by punctuation
        fallback = []
        current = ""
        for char in text:
            current += char
            if char in BENGALI_SENTENCE_ENDINGS:
                fallback.append(current.strip())
                current = ""
        if current.strip():
            fallback.append(current.strip())
        return fallback

def bengali_word_tokenize(text: str) -> List[str]:
    """
    Custom word tokenizer for Bengali text.
    
    Args:
        text (str): Bengali text to tokenize into words
        
    Returns:
        List[str]: List of Bengali words
    """
    try:
        # Simple whitespace-based tokenization for Bengali
        # This is a basic implementation and can be improved
        words = [w for w in re.split(r'\s+', text) if w]
        
        logger.info(f"Tokenized Bengali text into {len(words)} words")
        return words
    except Exception as e:
        logger.error(f"Bengali word tokenization failed: {str(e)}")
        return text.split()