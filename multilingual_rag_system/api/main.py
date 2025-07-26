"""
FastAPI server for the Multilingual RAG System.
This file provides RESTful API endpoints for web access.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
import uuid
from contextlib import asynccontextmanager
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import MultilingualRAGSystem  # Import from root main.py
from config.settings import *

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global RAG system instance
rag_system: Optional[MultilingualRAGSystem] = None
system_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global rag_system, system_ready
    
    # Startup
    logger.info("🚀 Starting FastAPI RAG server...")
    try:
        # Initialize RAG system
        rag_system = MultilingualRAGSystem()
        
        # Initialize knowledge base in background
        logger.info("📚 Initializing knowledge base...")
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None, 
            rag_system.initialize_knowledge_base
        )
        
        if success:
            system_ready = True
            logger.info("✅ RAG system ready!")
        else:
            logger.error("❌ Failed to initialize knowledge base")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG system: {str(e)}")
        yield
    finally:
        # Shutdown
        if rag_system:
            rag_system.shutdown()
            logger.info("✅ RAG system shutdown completed")

# Create FastAPI app
app = FastAPI(
    title="Multilingual RAG System API",
    description="A FastAPI server for Bengali and English document Q&A system using RAG architecture",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for query processing."""
    query: str = Field(
        ..., 
        min_length=1, 
        max_length=1000, 
        description="User query in Bengali or English",
        example="অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    )
    session_id: Optional[str] = Field(
        None, 
        description="Optional session ID for conversation continuity"
    )
    include_sources: bool = Field(
        True, 
        description="Whether to include source information in response"
    )

class QueryResponse(BaseModel):
    """Response model for query processing."""
    response: str = Field(..., description="Generated response")
    session_id: str = Field(..., description="Session ID for the conversation")
    language: str = Field(..., description="Detected language of the query")
    retrieved_documents: int = Field(..., description="Number of documents retrieved")
    context_length: int = Field(..., description="Total context length used")
    success: bool = Field(..., description="Whether the query was processed successfully")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source information")
    error: Optional[str] = Field(None, description="Error message if any")

class ConversationTurn(BaseModel):
    """Model for conversation history entry."""
    timestamp: str
    user_query: str
    assistant_response: str
    metadata: Optional[Dict[str, Any]] = None

class HistoryResponse(BaseModel):
    """Response model for conversation history."""
    session_id: str
    history: List[ConversationTurn]
    total_turns: int

class StatusResponse(BaseModel):
    """Response model for system status."""
    status: str
    message: str
    system_ready: bool
    components: Dict[str, bool]

class StatsResponse(BaseModel):
    """Response model for system statistics."""
    long_term_memory: Dict[str, Any]
    short_term_memory: Dict[str, Any]
    system_health: Dict[str, Any]

# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multilingual RAG System API",
        "description": "Bengali and English document Q&A system",
        "version": "1.0.0",
        "docs": "/docs",
        "status_endpoint": "/status",
        "query_endpoint": "/query"
    }

@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_system_status():
    """Get comprehensive system status."""
    global system_ready
    
    if not rag_system:
        return StatusResponse(
            status="error",
            message="RAG system not initialized",
            system_ready=False,
            components={"rag_system": False, "redis": False, "vector_db": False}
        )
    
    # Check component health
    components = {
        "rag_system": True,
        "redis": rag_system.short_term_memory.health_check() if hasattr(rag_system, 'short_term_memory') else False,
        "vector_db": True  # Assume healthy if system initialized
    }
    
    status = "healthy" if system_ready and all(components.values()) else "degraded"
    
    return StatusResponse(
        status=status,
        message="RAG system is running" if system_ready else "RAG system initializing",
        system_ready=system_ready,
        components=components
    )

@app.post("/query", response_model=QueryResponse, tags=["Query Processing"])
async def process_query(request: QueryRequest):
    """
    Process a user query and return AI-generated response.
    
    This endpoint accepts queries in both Bengali and English,
    retrieves relevant context from the knowledge base,
    and generates contextual responses.
    """
    if not rag_system or not system_ready:
        raise HTTPException(
            status_code=503,
            detail="RAG system not ready. Please wait for initialization to complete."
        )
    
    try:
        # Process query in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            rag_system.process_query,
            request.query,
            request.session_id,
            request.include_sources
        )
        
        # Convert to response model
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"❌ Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/history/{session_id}", response_model=HistoryResponse, tags=["Conversation"])
async def get_conversation_history(session_id: str):
    """Get conversation history for a specific session."""
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        history = rag_system.get_conversation_history(session_id)
        
        # Convert to response format
        conversation_turns = [
            ConversationTurn(
                timestamp=turn["timestamp"],
                user_query=turn["user_query"],
                assistant_response=turn["assistant_response"],
                metadata=turn.get("metadata")
            )
            for turn in history
        ]
        
        return HistoryResponse(
            session_id=session_id,
            history=conversation_turns,
            total_turns=len(conversation_turns)
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.delete("/history/{session_id}", tags=["Conversation"])
async def clear_conversation_history(session_id: str):
    """Clear conversation history for a specific session."""
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        success = rag_system.clear_session(session_id)
        
        if success:
            return {"message": f"Conversation history cleared for session {session_id}"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error clearing conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/initialize", tags=["System"])
async def reinitialize_knowledge_base(background_tasks: BackgroundTasks):
    """Reinitialize the knowledge base (admin endpoint)."""
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        # Reinitialize in background to avoid timeout
        background_tasks.add_task(rag_system.initialize_knowledge_base)
        
        return {
            "message": "Knowledge base reinitialization started",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"❌ Error reinitializing knowledge base: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/statistics", response_model=StatsResponse, tags=["System"])
async def get_system_statistics():
    """Get detailed system statistics and performance metrics."""
    if not rag_system:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        stats = rag_system.get_system_statistics()
        
        if "error" in stats:
            raise HTTPException(
                status_code=500,
                detail=f"Error getting statistics: {stats['error']}"
            )
        
        return StatsResponse(**stats)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting system statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/test-queries", tags=["Testing"])
async def get_test_queries():
    """Get sample test queries for testing the system."""
    return {
        "bengali_queries": [
            "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
            "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
            "অনুপমের চরিত্রের বৈশিষ্ট্য কী?",
            "অপরিচিতা গল্পের মূল বিষয়বস্তু কী?"
        ],
        "english_queries": [
            "Who is described as a handsome man in Anupam's language?",
            "What is the main theme of the Aparichita story?", 
            "Tell me about Anupam's character.",
            "What is the role of Mama in the story?",
            "Describe the social criticism in the story."
        ],
        "mixed_queries": [
            "What is অনুপম's relationship with his মামা?",
            "Describe কল্যাণী's character in English.",
            "What social issues does অপরিচিতা address?"
        ]
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=API_RELOAD,
        log_level="info"
    )
