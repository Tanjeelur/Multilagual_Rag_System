"""
Main RAG system implementation for standalone usage.
This file contains the core RAG system that can be used directly
without the web API.
"""

import logging
from typing import Dict, Any, Optional
import uuid

from config.settings import *
from src.data_processing.pdf_processor import PDFProcessor
from src.data_processing.text_cleaner import TextCleaner
from src.memory.long_term_memory import LongTermMemory
from src.memory.short_term_memory import ShortTermMemory
from src.retrieval.retriever import Retriever
from src.generation.generator import ResponseGenerator
from src.utils.helpers import detect_language, is_valid_query, setup_logging

# Setup logging
setup_logging(LOG_LEVEL, LOG_FILE)
logger = logging.getLogger(__name__)

class MultilingualRAGSystem:
    """
    Main RAG system for multilingual query processing.
    
    This is the core implementation that can be used standalone
    or imported by the FastAPI server.
    """
    
    def __init__(self):
        """Initialize the RAG system with all components."""
        logger.info("Initializing Multilingual RAG System...")
        
        try:
            # Initialize all components
            self.pdf_processor = PDFProcessor(OCR_LANGUAGES, OCR_GPU)
            self.text_cleaner = TextCleaner()
            
            # Initialize memory systems
            self.long_term_memory = LongTermMemory(
                DB_CONFIG, 
                EMBEDDING_MODEL, 
                VECTOR_DIMENSION
            )
            self.short_term_memory = ShortTermMemory(
                REDIS_CONFIG, 
                MAX_CONVERSATION_HISTORY,
                session_expiry_hours=24
            )
            
            # Initialize retrieval and generation
            self.retriever = Retriever(
                self.long_term_memory, 
                self.short_term_memory,
                max_context_length=2000
            )
            self.generator = ResponseGenerator(
                LLM_MODEL, 
                OPENAI_API_KEY,
                max_tokens=500,
                temperature=0.3
            )
            
            logger.info("✅ RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Error initializing RAG system: {str(e)}")
            raise
    
    def initialize_knowledge_base(self, pdf_path: str = PDF_PATH) -> bool:
        """
        Initialize the knowledge base from PDF document.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            bool: True if successful
        """
        try:
            logger.info(f"🔄 Starting knowledge base initialization from: {pdf_path}")
            
            # Extract text from PDF using OCR
            logger.info("📄 Extracting text from PDF...")
            extracted_texts = self.pdf_processor.extract_text_from_pdf(str(pdf_path))
            
            if not extracted_texts:
                logger.error("❌ No text extracted from PDF")
                return False
            
            logger.info(f"✅ Extracted text from {len(extracted_texts)} pages")
            
            # Process and clean text
            logger.info("🧹 Cleaning and processing text...")
            all_chunks = []
            
            for page_num, page_text in enumerate(extracted_texts, 1):
                if not page_text.strip():
                    continue
                    
                # Clean the text
                cleaned_text = self.text_cleaner.clean_text(page_text)
                
                if cleaned_text and len(cleaned_text) > 50:  # Skip very short texts
                    # Create chunks from cleaned text
                    chunks = self.text_cleaner.chunk_text(
                        cleaned_text, 
                        CHUNK_SIZE, 
                        CHUNK_OVERLAP,
                        preserve_sentences=True
                    )
                    
                    # Add page metadata to chunks
                    for chunk_idx, chunk in enumerate(chunks):
                        all_chunks.append({
                            'text': chunk,
                            'metadata': {
                                'page_number': page_num,
                                'chunk_index': chunk_idx,
                                'total_chunks_in_page': len(chunks),
                                'source_file': 'HSC26-Bangla1st-Paper.pdf',
                                'char_count': len(chunk),
                                'word_count': len(chunk.split())
                            }
                        })
            
            if not all_chunks:
                logger.error("❌ No valid text chunks created")
                return False
            
            logger.info(f"✅ Created {len(all_chunks)} text chunks")
            
            # Clear existing documents and store new ones
            logger.info("🗄️ Storing documents in long-term memory...")
            
            # Clear previous documents
            self.long_term_memory.clear_memory()
            
            # Prepare texts and metadata
            texts = [chunk['text'] for chunk in all_chunks]
            metadata = [chunk['metadata'] for chunk in all_chunks]
            
            # Store in long-term memory
            document_ids = self.long_term_memory.store_documents(
                documents=texts,
                metadata=metadata,
                source="HSC26-Bangla1st-Paper.pdf",
                language="mixed",  # Bengali and English mixed
                batch_size=50
            )
            
            logger.info(f"✅ Knowledge base initialization completed!")
            logger.info(f"📊 Stored {len(document_ids)} documents")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error initializing knowledge base: {str(e)}")
            return False
    
    def process_query(
        self, 
        query: str, 
        session_id: Optional[str] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Process a user query and generate response.
        
        Args:
            query (str): User query
            session_id (Optional[str]): Session identifier
            include_sources (bool): Whether to include source information
            
        Returns:
            Dict[str, Any]: Response with metadata
        """
        try:
            # Validate query
            if not is_valid_query(query):
                return {
                    "response": "দয়া করে একটি বৈধ প্রশ্ন করুন। Please provide a valid question.",
                    "error": "Invalid query",
                    "success": False
                }
            
            # Create session if not provided
            if not session_id:
                session_id = self.short_term_memory.create_session()
            
            # Detect query language
            language = detect_language(query)
            
            logger.info(f"🔍 Processing query [{language}]: {query[:50]}...")
            
            # Retrieve relevant context
            context = self.retriever.retrieve_context(
                query=query,
                session_id=session_id,
                max_documents=MAX_CHUNKS_TO_RETRIEVE,
                max_conversation_turns=3,
                include_conversation_history=True,
                similarity_threshold=0.3
            )
            
            # Format context for generation
            formatted_context = self.retriever.format_context_for_generation(context)
            
            # Generate response
            response = self.generator.generate_response(
                query=query,
                context=formatted_context,
                language=language,
                response_style="helpful",
                include_sources=include_sources
            )
            
            # Store conversation in short-term memory
            self.short_term_memory.add_conversation_turn(
                session_id=session_id,
                user_query=query,
                assistant_response=response,
                metadata={
                    "language": language,
                    "retrieved_documents": len(context.get("retrieved_documents", [])),
                    "context_length": context.get("total_context_length", 0)
                }
            )
            
            # Prepare response
            result = {
                "response": response,
                "session_id": session_id,
                "language": language,
                "retrieved_documents": len(context.get("retrieved_documents", [])),
                "context_length": context.get("total_context_length", 0),
                "success": True
            }
            
            # Add source information if requested
            if include_sources and context.get("retrieved_documents"):
                result["sources"] = [
                    {
                        "content_preview": doc["content"][:100] + "...",
                        "similarity_score": doc["similarity_score"],
                        "source": doc["source"],
                        "page_number": doc.get("metadata", {}).get("page_number", "unknown")
                    }
                    for doc in context["retrieved_documents"]
                ]
            
            logger.info("✅ Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {str(e)}")
            return {
                "response": "দুঃখিত, আপনার প্রশ্ন প্রক্রিয়া করতে একটি ত্রুটি হয়েছে। Sorry, I encountered an error processing your question.",
                "error": str(e),
                "success": False
            }
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        return self.short_term_memory.get_conversation_history(session_id)
    
    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session."""
        return self.short_term_memory.clear_session(session_id)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        try:
            stats = {
                "long_term_memory": self.long_term_memory.get_memory_statistics(),
                "short_term_memory": self.short_term_memory.get_memory_stats(),
                "retrieval_stats": self.retriever.get_retrieval_statistics(),
                "system_health": {
                    "redis_healthy": self.short_term_memory.health_check(),
                    "embedding_model": EMBEDDING_MODEL,
                    "llm_model": LLM_MODEL
                }
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting system statistics: {str(e)}")
            return {"error": str(e)}
    
    def shutdown(self) -> None:
        """Shutdown the RAG system gracefully."""
        try:
            logger.info("🔄 Shutting down RAG system...")
            
            if hasattr(self, 'long_term_memory'):
                self.long_term_memory.close()
            
            if hasattr(self, 'short_term_memory'):
                self.short_term_memory.close()
            
            logger.info("✅ RAG system shutdown completed")
            
        except Exception as e:
            logger.error(f"❌ Error during shutdown: {str(e)}")

# Main execution for standalone usage
def main():
    """
    Main function for standalone usage of the RAG system.
    This demonstrates how to use the system without the web API.
    """
    try:
        print("🚀 Starting Multilingual RAG System...")
        
        # Initialize RAG system
        rag_system = MultilingualRAGSystem()
        
        # Initialize knowledge base
        print("📚 Initializing knowledge base...")
        success = rag_system.initialize_knowledge_base()
        
        if not success:
            print("❌ Failed to initialize knowledge base")
            return
        
        print("✅ Knowledge base initialized successfully!")
        
        # Test with sample queries
        test_queries = [
            "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
            "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
            "What is the main theme of the story?",
            "Tell me about Anupam's character."
        ]
        
        print("\n🔍 Testing with sample queries...\n")
        
        session_id = None
        
        for i, query in enumerate(test_queries, 1):
            print(f"📝 Query {i}: {query}")
            
            result = rag_system.process_query(query, session_id)
            
            if result["success"]:
                print(f"💬 Response: {result['response']}")
                print(f"🌐 Language: {result['language']}")
                print(f"📄 Retrieved documents: {result['retrieved_documents']}")
                
                if not session_id:
                    session_id = result['session_id']
                    
                if result.get('sources'):
                    print("📚 Sources:")
                    for j, source in enumerate(result['sources'][:2], 1):
                        print(f"   {j}. Page {source['page_number']} (similarity: {source['similarity_score']:.3f})")
                        print(f"      Preview: {source['content_preview']}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
            
            print("-" * 80)
        
        # Show conversation history
        if session_id:
            print(f"\n💭 Conversation History (Session: {session_id[:8]}...):")
            history = rag_system.get_conversation_history(session_id)
            for turn in history[-3:]:  # Show last 3 turns
                print(f"   User: {turn['user_query'][:50]}...")
                print(f"   Assistant: {turn['assistant_response'][:50]}...")
                print()
        
        # Show system statistics
        print("📊 System Statistics:")
        stats = rag_system.get_system_statistics()
        if "error" not in stats:
            print(f"   📚 Total documents: {stats['long_term_memory']['total_documents']}")
            print(f"   💬 Active sessions: {stats['short_term_memory']['active_sessions']}")
            print(f"   ✅ Redis healthy: {stats['system_health']['redis_healthy']}")
        
        # Shutdown
        rag_system.shutdown()
        print("\n✅ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in main: {str(e)}")
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main()
