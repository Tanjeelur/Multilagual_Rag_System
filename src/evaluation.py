from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.text_extraction import extract_text_from_pdf
from src.preprocessing import clean_text, chunk_text
from src.embedding import Embedder
from src.storage import Storage
from src.retrieval import Retriever
from src.generation import Generator
from src.config import DB_CONFIG, REDIS_CONFIG, PDF_PATH
import logging
import psycopg2

# Configure logging
logging.basicConfig(level=logging.INFO, filename='logs/rag_pipeline.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Multilingual RAG API", description="A RAG system for English and Bengali queries.")

class QueryRequest(BaseModel):
    query: str
    language: str = "bn"

class QueryResponse(BaseModel):
    answer: str
    confidence: float

def initialize_pipeline():
    """
    Initialize the RAG pipeline by processing the PDF and storing chunks.
    
    Returns:
        Tuple[Embedder, Storage]: Initialized embedder and storage objects.
    """
    try:
        # Extract and process PDF
        pages = extract_text_from_pdf(PDF_PATH)
        cleaned_pages = [clean_text(page) for page in pages]
        all_chunks = []
        page_numbers = []
        for page_num, page in enumerate(cleaned_pages, 1):
            chunks = chunk_text(page)
            all_chunks.extend(chunks)
            page_numbers.extend([page_num] * len(chunks))

        # Embed and store
        embedder = Embedder()
        embeddings = embedder.embed(all_chunks)
        storage = Storage(DB_CONFIG, REDIS_CONFIG)
        storage.store_chunks(all_chunks, embeddings, page_numbers)
        
        return embedder, storage
    except Exception as e:
        logger.error(f"Pipeline initialization failed: {str(e)}")
        raise

# Initialize pipeline
embedder, storage = initialize_pipeline()
retriever = Retriever(storage.conn)
generator = Generator()

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    Handle query and return answer with confidence score.
    
    Args:
        request (QueryRequest): Query and language.
    
    Returns:
        QueryResponse: Generated answer and confidence.
    """
    try:
        # Get context from recent queries
        recent_queries = storage.get_recent_queries()
        full_query = " ".join(recent_queries + [request.query])

        # Retrieve and generate
        query_embedding = embedder.embed([full_query])[0]
        chunks = retriever.retrieve(query_embedding)
        answer, confidence = generator.generate(full_query, chunks, request.language, embedder)

        # Store query
        storage.store_query(request.query)

        return {"answer": answer, "confidence": confidence}
    except Exception as e:
        logger.error(f"API query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Log startup event."""
    logger.info("FastAPI server started")