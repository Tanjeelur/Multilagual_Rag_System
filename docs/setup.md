Setup Guide
Prerequisites

Python 3.10+
Tesseract OCR with Bengali support (tesseract-ocr-ben)
PostgreSQL with a database rag_db (user: user, password: password)
Redis server

Installation

Clone the Repository:git clone https://github.com/your-username/multilingual-rag.git
cd multilingual-rag


Install Tesseract OCR:
Linux (Ubuntu): sudo apt-get install tesseract-ocr tesseract-ocr-ben
Mac: brew install tesseract tesseract-lang
Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki, add to PATH, install tesseract-ocr-ben.


Install PostgreSQL and Redis:
PostgreSQL: Install from https://www.postgresql.org/download/, create database rag_db with user user and password password.
Redis: Install from https://redis.io/docs/install/install-redis/, start with redis-server.


Install Python Dependencies:pip install -r requirements.txt


Add PDF:Place HSC26-Bangla1st-Paper.pdf in the data/ directory.
Initialize the Pipeline:python -m uvicorn src.api:app --host 0.0.0.0 --port 8000



Configuration

Update src/config.py with your PostgreSQL and Redis settings if different.
Ensure the PDF path in config.py points to the correct file.

Running the API
Start the FastAPI server:
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

Access the API at http://localhost:8000/docs.
Testing
Run unit tests:
pytest tests/

Troubleshooting

OCR Issues: Ensure tesseract-ocr-ben is installed. Use Google Cloud Vision API for better accuracy if needed.
Database Errors: Verify PostgreSQL (rag_db, user user, password password) and Redis are running.
PDF Missing: Ensure HSC26-Bangla1st-Paper.pdf is in data/.
