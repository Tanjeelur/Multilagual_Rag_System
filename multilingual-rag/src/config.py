"""Configuration settings for the RAG pipeline.
"""
import os

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Use absolute path for PDF file
PDF_PATH = os.path.join(BASE_DIR, "data", "HSC26-Bangla1st-Paper.pdf")
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres.sqnqmlvoyphdfcqplmyf",
    "password": "IZu8Dd7pItX6jzOt",
    "host": "aws-0-ap-southeast-1.pooler.supabase.com",
    "port": "5432"
}
REDIS_CONFIG = {
    "host": "fitting-puma-60812.upstash.io",
    "password": "Ae2MAAIjcDEzNzYzNDU1ODBmNGU0NWY2ODQ4MGQxYWY1NTExOWVhN3AxMA",
    "port": 6379,
    "ssl": True
}