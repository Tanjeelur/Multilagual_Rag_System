"""
Script to run the FastAPI server.
This script starts the web server for the RAG system API.
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import configuration
from config.settings import API_HOST, API_PORT, API_RELOAD, LOG_LEVEL

def main():
    """Start the FastAPI server."""
    print("🚀 Starting Multilingual RAG System API Server...")
    print(f"📍 Server will run at: http://{API_HOST}:{API_PORT}")
    print(f"📚 API Documentation: http://{API_HOST}:{API_PORT}/docs")
    print(f"🔍 Interactive API: http://{API_HOST}:{API_PORT}/redoc")
    print("-" * 60)
    
    try:
        uvicorn.run(
            "api.main:app",
            host=API_HOST,
            port=API_PORT,
            reload=API_RELOAD,
            log_level=LOG_LEVEL.lower(),
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n⏹️ Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {str(e)}")

if __name__ == "__main__":
    main()
