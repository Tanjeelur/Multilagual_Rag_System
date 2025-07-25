Multilingual RAG System
A Retrieval-Augmented Generation (RAG) system for processing queries in English and Bengali, using the "HSC26-Bangla1st-Paper.pdf" corpus.
Features

Text Extraction: Uses PyMuPDF and Tesseract for accurate text extraction, including Bengali OCR.
Preprocessing: Cleans and chunks text into paragraphs with overlap for semantic retrieval.
Embedding: LaBSE for multilingual embeddings.
Storage: PostgreSQL with pgvector for vector storage, Redis for short-term query memory.
Retrieval: Cosine similarity-based retrieval of top-k chunks.
Generation: mT5 for generating answers in English or Bengali.
API: FastAPI-based REST API with OpenAPI documentation.
Evaluation: Metrics for groundedness, relevance, and accuracy.

Setup

Clone the Repository:git clone https://github.com/your-username/multilingual-rag.git
cd multilingual-rag


Install Prerequisites:
Install Python 3.10+.
Install Tesseract OCR with Bengali support:
Linux (Ubuntu): sudo apt-get install tesseract-ocr tesseract-ocr-ben
Mac: brew install tesseract tesseract-lang
Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki and install tesseract-ocr-ben.


Install PostgreSQL and create a database rag_db with user user and password password.
Install and start Redis (redis-server).


Install Python Dependencies:pip install -r requirements.txt


Add PDF:Place HSC26-Bangla1st-Paper.pdf in the data/ directory.
Run the API:python -m uvicorn src.api:app --host 0.0.0.0 --port 8000



API Usage

Endpoint: POST /query
Request Example:{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
  "language": "bn"
}


Response Example:{
  "answer": "শুম্ভুনাথ",
  "confidence": 0.85
}


Access API docs at http://localhost:8000/docs.

Sample Queries



Query
Expected Answer



অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
শুম্ভুনাথ


কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
মামাকে


বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
১৫ বছর


Evaluation
Run evaluation with:
python -m src.evaluation

Results are logged to logs/rag_pipeline.log and summarized in docs/evaluation.md.
Documentation

docs/setup.md: Detailed setup instructions.
docs/api.md: API documentation.
docs/evaluation.md: Evaluation results and metrics.

Testing
Run unit tests:
pytest tests/

Requirements
See requirements.txt for dependencies.
Notes

Ensure the HSC26-Bangla1st-Paper.pdf contains the relevant Bengali passages for accurate results.
If OCR quality is poor, consider using Google Cloud Vision API for better accuracy.
Ensure PostgreSQL and Redis are running before starting the API.

