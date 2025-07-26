"""
Utility functions and helpers for the multilingual RAG system.

This module provides various helper functions for language detection,
text validation, logging setup, and other common operations.
"""

import re
import uuid
import logging
from typing import List, Optional
from pathlib import Path

def detect_language(text: str) -> str:
    """
    Detect the primary language of the text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Language code ('en', 'bn', or 'mixed')
    """
    if not text.strip():
        return 'en'  # Default to English for empty text
    
    # Count Bengali characters (Unicode range for Bengali)
    bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    
    # Count English characters
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    total_chars = bengali_chars + english_chars
    
    if total_chars == 0:
        return 'en'  # Default to English if no letters found
    
    bengali_ratio = bengali_chars / total_chars
    
    # Determine language based on character ratio
    if bengali_ratio > 0.7:
        return 'bn'  # Predominantly Bengali
    elif bengali_ratio < 0.3:
        return 'en'  # Predominantly English
    else:
        return 'mixed'  # Mixed language content

def is_valid_query(query: str, min_length: int = 3, max_length: int = 1000) -> bool:
    """
    Check if a query is valid for processing.
    
    Args:
        query (str): User query to validate
        min_length (int): Minimum acceptable length
        max_length (int): Maximum acceptable length
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not query or not isinstance(query, str):
        return False
    
    # Check if query is just whitespace
    if not query.strip():
        return False
    
    # Check length constraints
    query_length = len(query.strip())
    if query_length < min_length or query_length > max_length:
        return False
    
    # Check if query contains at least some meaningful characters
    # Must contain at least one letter (Bengali or English)
    if not re.search(r'[\u0980-\u09FF\u0041-\u005A\u0061-\u007A]', query):
        return False
    
    return True

def generate_session_id() -> str:
    """
    Generate a unique session ID for conversation tracking.
    
    Returns:
        str: Unique session identifier
    """
    return str(uuid.uuid4())

def generate_document_id(source: str, chunk_index: int) -> str:
    """
    Generate a document ID based on source and chunk index.
    
    Args:
        source (str): Source identifier
        chunk_index (int): Index of the chunk
        
    Returns:
        str: Document ID
    """
    # Clean source name for use in ID
    clean_source = re.sub(r'[^\w\-_]', '_', source.lower())
    return f"{clean_source}_{chunk_index:04d}"

def clean_filename(filename: str) -> str:
    """
    Clean a filename to be safe for file system usage.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Cleaned filename
    """
    # Remove or replace unsafe characters
    cleaned = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple consecutive underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores and spaces
    cleaned = cleaned.strip('_ ')
    
    # Ensure filename is not empty
    if not cleaned:
        cleaned = "unnamed_file"
    
    return cleaned

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with optional suffix.
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length including suffix
        suffix (str): Suffix to add when truncating
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    # Account for suffix length
    truncate_length = max_length - len(suffix)
    
    if truncate_length <= 0:
        return suffix[:max_length]
    
    return text[:truncate_length] + suffix

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text (simple implementation).
    
    Args:
        text (str): Input text
        max_keywords (int): Maximum number of keywords to extract
        
    Returns:
        List[str]: List of extracted keywords
    """
    if not text.strip():
        return []
    
    # Bengali and English stop words (simplified list)
    stop_words = {
        # English stop words
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'were', 'will', 'with', 'you', 'your', 'this', 'they',
        # Bengali stop words  
        'এই', 'এটি', 'এটা', 'সেই', 'সেটি', 'সেটা', 'যেই', 'যেটি', 'যেটা',
        'আমি', 'আমার', 'আমাদের', 'তুমি', 'তোমার', 'তোমাদের', 'সে', 'তার',
        'তাদের', 'এবং', 'বা', 'কিন্তু', 'যদি', 'তাহলে', 'কেন', 'কিভাবে',
        'কখন', 'কোথায়', 'কি', 'কী', 'কে', 'কাকে', 'কার', 'কাদের', 'হয়',
        'হল', 'হলো', 'করে', 'করেছে', 'করেছেন', 'করবে', 'করবেন'
    }
    
    # Extract words and filter
    # Remove punctuation and convert to lowercase
    cleaned_text = re.sub(r'[।\.!?,:;()"\'\-]', ' ', text.lower())
    words = cleaned_text.split()
    
    # Filter keywords
    keywords = []
    for word in words:
        word = word.strip()
        if (len(word) > 2 and 
            word not in stop_words and 
            word not in keywords and
            re.search(r'[\u0980-\u09FF\u0041-\u005A\u0061-\u007A]', word)):
            keywords.append(word)
            
            if len(keywords) >= max_keywords:
                break
    
    return keywords

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.
    
    Args:
        size_bytes (int): Size in bytes
        
    Returns:
        str: Formatted file size
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def setup_logging(
    level: str = "INFO", 
    log_file: Optional[str] = None,
    include_console: bool = True
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (Optional[str]): Path to log file (optional)  
        include_console (bool): Whether to include console output
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Add console handler if requested
    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if log file specified
    if log_file:
        # Create log directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

def validate_config(config: dict, required_keys: List[str]) -> bool:
    """
    Validate that a configuration dictionary contains required keys.
    
    Args:
        config (dict): Configuration dictionary to validate
        required_keys (List[str]): List of required keys
        
    Returns:
        bool: True if all required keys are present
    """
    missing_keys = []
    
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        logging.error(f"Missing required configuration keys: {missing_keys}")
        return False
    
    return True

def safe_json_loads(json_string: str, default=None):
    """
    Safely load JSON string with fallback to default value.
    
    Args:
        json_string (str): JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON object or default value
    """
    try:
        import json
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default

def get_file_info(file_path: str) -> dict:
    """
    Get information about a file.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        dict: File information including size, modification time, etc.
    """
    try:
        path = Path(file_path)
        
        if not path.exists():
            return {"error": "File not found"}
        
        stat = path.stat()
        
        return {
            "name": path.name,
            "size": stat.st_size,
            "size_formatted": format_file_size(stat.st_size),
            "modified": stat.st_mtime,
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
            "extension": path.suffix.lower()
        }
        
    except Exception as e:
        return {"error": str(e)}

# Constants for common regex patterns
PATTERNS = {
    "bengali_text": r'[\u0980-\u09FF]+',
    "english_text": r'[a-zA-Z]+',
    "numbers": r'\d+',
    "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    "url": r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
}

def extract_pattern(text: str, pattern_name: str) -> List[str]:
    """
    Extract text matching a predefined pattern.
    
    Args:
        text (str): Input text
        pattern_name (str): Name of the pattern to use
        
    Returns:
        List[str]: List of matches
    """
    if pattern_name not in PATTERNS:
        return []
    
    pattern = PATTERNS[pattern_name]
    matches = re.findall(pattern, text)
    
    return matches
