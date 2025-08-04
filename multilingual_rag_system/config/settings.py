"""
Configuration settings for the Multilingual RAG system.

This module contains all configuration parameters including database credentials,
model settings, and processing parameters.
"""

import os
from pathlib import Path
from typing import Dict, Any, List

# Base directory for the project
BASE_DIR = Path(__file__).parent.parent

# PDF file path - place your Bengali book here
PDF_PATH = BASE_DIR / "data" / "HSC26-Bangla1st-Paper.pdf"

# Database configuration for PostgreSQL with pgvector (long-term memory)
DB_CONFIG: Dict[str, Any] = {
    "dbname": "postgres",
    "user": "postgres.sqnqmlvoyphdfcqplmyf", 
    "password": "POSTGRESS_PASSWORD",
    "host": "aws-0-ap-southeast-1.pooler.supabase.com",
    "port": "5432"
}

# Redis configuration for short-term memory (conversation history)
REDIS_CONFIG: Dict[str, Any] = {
    "host": "fitting-puma-60812.upstash.io",
    "password": "REDIS-PASSWORD",
    "port": 6379,
    "ssl": True
}

# OpenAI API configuration - Add your API key here
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")

# Model configurations
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"  # Multilingual embeddings
LLM_MODEL: str = "gpt-3.5-turbo"  # Generation model

# Text processing parameters
CHUNK_SIZE: int = 500  # Maximum characters per text chunk
CHUNK_OVERLAP: int = 50  # Overlapping characters between chunks
MAX_CHUNKS_TO_RETRIEVE: int = 5  # Number of relevant chunks to retrieve
MAX_CONVERSATION_HISTORY: int = 10  # Maximum conversation turns to store

# Language settings
SUPPORTED_LANGUAGES: List[str] = ["en", "bn"]  # English and Bengali

# Logging configuration
LOG_LEVEL: str = "INFO"
LOG_FILE: str = str(BASE_DIR / "logs" / "rag_system.log")

# OCR settings for Bengali text processing
OCR_LANGUAGES: List[str] = ['en', 'bn']
OCR_GPU: bool = False  # Set to True if you have GPU support

# API settings
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000
API_RELOAD: bool = True

# Vector database settings
VECTOR_DIMENSION: int = 384  # Dimension for all-MiniLM-L6-v2 model
