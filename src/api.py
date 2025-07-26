from fastapi import FastAPI, HTTPException, BackgroundTasks
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
import asyncio
import time
from contextlib import contextmanager

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
    def is_story_line(line):
        # Remove empty lines and lines with only whitespace
        if not line.strip():
            return False
        # Remove known repeated headers/artifacts
        if "অনলাহন ব্যাচ বাংলা ইংরেজি:" in line:
            return False
        # Remove lines with only numbers or symbols
        if line.strip().isdigit():
            return False
        # Keep all lines with Bengali characters
        bengali_chars = sum(0x0980 <= ord(c) <= 0x09FF for c in line)
        if bengali_chars > 0:
            return True
        return False

    try:
        # Extract and process PDF
        pages = extract_text_from_pdf(PDF_PATH)
        # Apply story line cleaning to each page
        cleaned_pages = []
        for page in pages:
            lines = page.splitlines()
            story_lines = [line for line in lines if is_story_line(line)]
            cleaned_pages.append("\n".join(story_lines))
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
        
        # Ensure database connection is valid before proceeding
        if storage.conn.closed:
            if not storage.reconnect_db():
                raise HTTPException(status_code=503, detail="Database connection unavailable. Please try again later.")
            retriever.conn = storage.conn
            
        # Use retriever with the current connection
        chunks = retriever.retrieve(query_embedding)
        logger.info(f"Retrieved chunks for query '{request.query}': {[chunk[1][:100] for chunk in chunks]}")
        answer, confidence = generator.generate(full_query, chunks, request.language, embedder)

        # Store query
        storage.store_query(request.query)

        return {"answer": answer, "confidence": confidence}
    except psycopg2.OperationalError as e:
        logger.error(f"Database connection error: {str(e)}")
        # Try to reconnect
        if storage.reconnect_db():
            retriever.conn = storage.conn
            # Return a more user-friendly error message
            raise HTTPException(status_code=500, detail="Database connection was reset. Please try your query again.")
        else:
            raise HTTPException(status_code=503, detail="Database service unavailable. Please try again later.")
    except Exception as e:
        logger.error(f"API query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Context manager for database operations
@contextmanager
def get_db_cursor():
    """Context manager for database operations to ensure proper transaction handling."""
    if storage.conn.closed:
        storage.reconnect_db()
        
    # Create a new cursor
    cursor = storage.conn.cursor()
    try:
        yield cursor
        storage.conn.commit()
    except Exception as e:
        storage.conn.rollback()
        logger.error(f"Database operation failed: {str(e)}")
        raise
    finally:
        cursor.close()

# Background task to check database connection periodically
async def check_db_connection():
    """Periodically check database connection and reconnect if needed."""
    while True:
        try:
            if storage.conn.closed:
                logger.warning("Database connection is closed, attempting to reconnect...")
                if storage.reconnect_db():
                    logger.info("Successfully reconnected to database")
                    # Update retriever's connection reference
                    retriever.conn = storage.conn
                else:
                    logger.error("Failed to reconnect to database")
            else:
                # Test the connection with a simple query
                with get_db_cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                logger.debug("Database connection is healthy")
        except Exception as e:
            logger.error(f"Error checking database connection: {str(e)}")
            try:
                storage.reconnect_db()
                retriever.conn = storage.conn
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect to database: {str(reconnect_error)}")
        
        # Wait for 60 seconds before checking again
        await asyncio.sleep(60)

@app.on_event("startup")
async def startup_event():
    """Log startup event and start background tasks."""
    logger.info("FastAPI server started")
    
    # Start the background task for checking database connection
    asyncio.create_task(check_db_connection())
    
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when shutting down."""
    logger.info("FastAPI server shutting down, closing connections...")
    
    # Close database connection
    if hasattr(storage, 'conn') and storage.conn and not storage.conn.closed:
        try:
            storage.conn.close()
            logger.info("Database connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")
    
    # Close Redis connection
    if hasattr(storage, 'redis'):
        try:
            storage.redis.close()
            logger.info("Redis connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")
    
@app.get("/health")
async def health_check():
    """Check the health of the API and its dependencies."""
    health_status = {"status": "healthy", "database": "connected", "redis": "connected"}
    
    try:
        # Check database connection
        if storage.conn.closed:
            storage.reconnect_db()
            
        with storage.conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
    except Exception as e:
        health_status["database"] = f"disconnected: {str(e)}"
        health_status["status"] = "unhealthy"
    
    try:
        # Check Redis connection
        storage.redis.ping()
    except Exception as e:
        health_status["redis"] = f"disconnected: {str(e)}"
        health_status["status"] = "unhealthy"
    
    return health_status