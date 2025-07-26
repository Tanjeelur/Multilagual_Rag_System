"""
Document retrieval module combining long-term and short-term memory.

This module orchestrates the retrieval of relevant context from both
persistent document storage and recent conversation history.
"""

from typing import List, Dict, Any, Optional
import logging
from ..memory.long_term_memory import LongTermMemory
from ..memory.short_term_memory import ShortTermMemory
from ..utils.helpers import detect_language

logger = logging.getLogger(__name__)

class Retriever:
    """
    Orchestrates document retrieval combining multiple memory sources.
    
    This class intelligently combines information from long-term document storage
    and short-term conversation history to provide comprehensive context for
    question answering.
    """
    
    def __init__(
        self, 
        long_term_memory: LongTermMemory, 
        short_term_memory: ShortTermMemory,
        max_context_length: int = 2000
    ):
        """
        Initialize the retriever with memory components.
        
        Args:
            long_term_memory (LongTermMemory): Long-term document storage
            short_term_memory (ShortTermMemory): Short-term conversation memory
            max_context_length (int): Maximum characters in combined context
        """
        self.long_term_memory = long_term_memory
        self.short_term_memory = short_term_memory
        self.max_context_length = max_context_length
        
        logger.info("Document retriever initialized successfully")
    
    def retrieve_context(
        self, 
        query: str, 
        session_id: str,
        max_documents: int = 5,
        max_conversation_turns: int = 3,
        include_conversation_history: bool = True,
        similarity_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Retrieve comprehensive context for a query from all memory sources.
        
        Args:
            query (str): User query
            session_id (str): Session identifier for conversation context
            max_documents (int): Maximum documents to retrieve from long-term memory
            max_conversation_turns (int): Maximum conversation turns to include
            include_conversation_history (bool): Whether to include conversation history
            similarity_threshold (float): Minimum similarity threshold for documents
            
        Returns:
            Dict[str, Any]: Combined context from all sources
        """
        try:
            logger.info(f"Retrieving context for query: '{query[:50]}...' in session {session_id}")
            
            context = {
                "query": query,
                "session_id": session_id,
                "query_language": detect_language(query),
                "retrieved_documents": [],
                "conversation_history": [],
                "total_context_length": 0,
                "metadata": {}
            }
            
            # Retrieve from long-term memory (document knowledge base)
            try:
                relevant_documents = self.long_term_memory.retrieve_relevant_documents(
                    query=query,
                    k=max_documents,
                    similarity_threshold=similarity_threshold,
                    include_metadata=True
                )
                
                context["retrieved_documents"] = relevant_documents
                context["metadata"]["documents_found"] = len(relevant_documents)
                
                if relevant_documents:
                    avg_similarity = sum(doc["similarity_score"] for doc in relevant_documents) / len(relevant_documents)
                    context["metadata"]["average_similarity"] = round(avg_similarity, 3)
                
                logger.debug(f"Retrieved {len(relevant_documents)} relevant documents")
                
            except Exception as e:
                logger.error(f"Error retrieving from long-term memory: {str(e)}")
                context["metadata"]["long_term_error"] = str(e)
            
            # Retrieve from short-term memory (conversation history)
            if include_conversation_history:
                try:
                    conversation_history = self.short_term_memory.get_conversation_history(
                        session_id=session_id,
                        limit=max_conversation_turns
                    )
                    
                    context["conversation_history"] = conversation_history
                    context["metadata"]["conversation_turns"] = len(conversation_history)
                    
                    logger.debug(f"Retrieved {len(conversation_history)} conversation turns")
                    
                except Exception as e:
                    logger.error(f"Error retrieving from short-term memory: {str(e)}")
                    context["metadata"]["short_term_error"] = str(e)
            
            # Calculate total context length
            context["total_context_length"] = self._calculate_context_length(context)
            
            # Trim context if it exceeds maximum length
            if context["total_context_length"] > self.max_context_length:
                context = self._trim_context(context)
            
            logger.info(f"Context retrieval completed: {context['total_context_length']} characters")
            return context
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return {
                "query": query,
                "session_id": session_id,
                "error": str(e),
                "retrieved_documents": [],
                "conversation_history": []
            }
    
    def _calculate_context_length(self, context: Dict[str, Any]) -> int:
        """Calculate total character length of context."""
        try:
            total_length = 0
            
            # Add document content length
            for doc in context.get("retrieved_documents", []):
                total_length += len(doc.get("content", ""))
            
            # Add conversation history length
            for turn in context.get("conversation_history", []):
                total_length += len(turn.get("user_query", ""))
                total_length += len(turn.get("assistant_response", ""))
            
            return total_length
            
        except Exception as e:
            logger.warning(f"Error calculating context length: {str(e)}")
            return 0
    
    def _trim_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Trim context to fit within maximum length."""
        try:
            logger.info("Trimming context to fit maximum length")
            
            # Prioritize recent documents and conversation
            # Trim older conversation history first
            while (context["total_context_length"] > self.max_context_length and 
                   len(context["conversation_history"]) > 1):
                context["conversation_history"].pop(0)  # Remove oldest turn
                context["total_context_length"] = self._calculate_context_length(context)
            
            # If still too long, trim documents from the end (lowest similarity)
            while (context["total_context_length"] > self.max_context_length and 
                   len(context["retrieved_documents"]) > 1):
                context["retrieved_documents"].pop()  # Remove least similar document
                context["total_context_length"] = self._calculate_context_length(context)
            
            return context
            
        except Exception as e:
            logger.error(f"Error trimming context: {str(e)}")
            return context
    
    def format_context_for_generation(self, context: Dict[str, Any]) -> str:
        """
        Format retrieved context for text generation.
        
        Args:
            context (Dict[str, Any]): Retrieved context dictionary
            
        Returns:
            str: Formatted context string ready for LLM input
        """
        try:
            formatted_parts = []
            
            # Add conversation history if available
            if context.get("conversation_history"):
                formatted_parts.append("## Previous Conversation:")
                for turn in context["conversation_history"]:
                    formatted_parts.append(f"User: {turn['user_query']}")
                    formatted_parts.append(f"Assistant: {turn['assistant_response']}")
                formatted_parts.append("")  # Empty line for separation
            
            # Add retrieved documents if available
            if context.get("retrieved_documents"):
                formatted_parts.append("## Relevant Information:")
                for i, doc in enumerate(context["retrieved_documents"], 1):
                    similarity = doc.get("similarity_score", 0)
                    source = doc.get("source", "unknown")
                    formatted_parts.append(f"{i}. [Similarity: {similarity:.3f}, Source: {source}]")
                    formatted_parts.append(f"   {doc['content']}")
                    formatted_parts.append("")  # Empty line for separation
            
            formatted_context = "\n".join(formatted_parts)
            
            logger.debug(f"Formatted context: {len(formatted_context)} characters")
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error formatting context: {str(e)}")
            return "Error formatting context for generation."
    
    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about retrieval performance.
        
        Returns:
            Dict[str, Any]: Retrieval statistics
        """
        try:
            stats = {
                "long_term_memory": self.long_term_memory.get_memory_statistics(),
                "short_term_memory": self.short_term_memory.get_memory_stats(),
                "max_context_length": self.max_context_length
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting retrieval statistics: {str(e)}")
            return {"error": str(e)}
