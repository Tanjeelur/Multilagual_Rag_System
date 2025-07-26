"""
Short-term memory implementation using Redis for conversation history.

This module manages temporary conversation state and recent interactions
with efficient caching and automatic expiration.
"""

import redis
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class ShortTermMemory:
    """
    Manages short-term conversation memory using Redis for fast access.
    
    This class handles conversation history, user sessions, and temporary
    context with automatic expiration and cleanup mechanisms.
    """
    
    def __init__(
        self, 
        redis_config: Dict[str, Any], 
        max_history: int = 10,
        session_expiry_hours: int = 24
    ):
        """
        Initialize short-term memory with Redis configuration.
        
        Args:
            redis_config (Dict[str, Any]): Redis connection configuration
            max_history (int): Maximum conversation turns to maintain per session
            session_expiry_hours (int): Hours after which sessions expire
        """
        self.redis_config = redis_config
        self.max_history = max_history
        self.session_expiry_seconds = session_expiry_hours * 3600
        self.redis_client = None
        
        # Connect to Redis
        self._connect()
        
    def _connect(self) -> None:
        """Establish connection to Redis server."""
        try:
            logger.info("Connecting to Redis server")
            
            self.redis_client = redis.Redis(
                host=self.redis_config['host'],
                port=self.redis_config['port'],
                password=self.redis_config.get('password'),
                ssl=self.redis_config.get('ssl', False),
                decode_responses=True,
                socket_timeout=30,
                socket_connect_timeout=30,
                retry_on_timeout=True
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis")
            
        except Exception as e:
            logger.error(f"Error connecting to Redis: {str(e)}")
            raise
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            user_id (Optional[str]): Optional user identifier
            
        Returns:
            str: New session ID
        """
        try:
            session_id = str(uuid.uuid4())
            
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "conversation_count": 0
            }
            
            # Store session metadata
            session_key = f"session:{session_id}"
            self.redis_client.hset(session_key, mapping=session_data)
            self.redis_client.expire(session_key, self.session_expiry_seconds)
            
            logger.info(f"Created new session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            raise
    
    def add_conversation_turn(
        self, 
        session_id: str, 
        user_query: str, 
        assistant_response: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a conversation turn to session memory.
        
        Args:
            session_id (str): Session identifier
            user_query (str): User's query
            assistant_response (str): Assistant's response
            metadata (Optional[Dict[str, Any]]): Additional metadata for the turn
            
        Returns:
            bool: True if successful
        """
        if not session_id or not user_query.strip():
            logger.warning("Invalid session_id or empty user_query")
            return False
        
        try:
            # Prepare conversation entry
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query.strip(),
                "assistant_response": assistant_response.strip(),
                "metadata": metadata or {}
            }
            
            # Keys for conversation and session
            conversation_key = f"conversation:{session_id}"
            session_key = f"session:{session_id}"
            
            # Add conversation turn to list
            self.redis_client.lpush(conversation_key, json.dumps(conversation_entry))
            
            # Trim conversation history to max_history
            self.redis_client.ltrim(conversation_key, 0, self.max_history - 1)
            
            # Set expiration for conversation
            self.redis_client.expire(conversation_key, self.session_expiry_seconds)
            
            # Update session metadata
            self.redis_client.hset(session_key, mapping={
                "last_activity": datetime.now().isoformat(),
                "conversation_count": self.redis_client.llen(conversation_key)
            })
            self.redis_client.expire(session_key, self.session_expiry_seconds)
            
            logger.debug(f"Added conversation turn to session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding conversation turn: {str(e)}")
            return False
    
    def get_conversation_history(
        self, 
        session_id: str, 
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve conversation history for a session.
        
        Args:
            session_id (str): Session identifier
            limit (Optional[int]): Maximum number of turns to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of conversation turns in chronological order
        """
        if not session_id:
            logger.warning("No session_id provided")
            return []
        
        try:
            conversation_key = f"conversation:{session_id}"
            
            # Get conversation history
            if limit:
                history_entries = self.redis_client.lrange(conversation_key, 0, limit - 1)
            else:
                history_entries = self.redis_client.lrange(conversation_key, 0, -1)
            
            # Parse and reverse to get chronological order (oldest first)
            parsed_history = []
            for entry in reversed(history_entries):
                try:
                    parsed_entry = json.loads(entry)
                    parsed_history.append(parsed_entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing conversation entry: {str(e)}")
                    continue
            
            logger.debug(f"Retrieved {len(parsed_history)} conversation turns for session {session_id}")
            return parsed_history
            
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear conversation history and session data for a specific session.
        
        Args:
            session_id (str): Session identifier
            
        Returns:
            bool: True if successful
        """
        try:
            conversation_key = f"conversation:{session_id}"
            session_key = f"session:{session_id}"
            
            # Delete conversation history and session data
            deleted_conv = self.redis_client.delete(conversation_key)
            deleted_session = self.redis_client.delete(session_key)
            
            if deleted_conv or deleted_session:
                logger.info(f"Cleared session {session_id}")
                return True
            else:
                logger.warning(f"Session {session_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error clearing session: {str(e)}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about short-term memory usage.
        
        Returns:
            Dict[str, Any]: Memory statistics
        """
        try:
            session_keys = self.redis_client.keys("session:*")
            conversation_keys = self.redis_client.keys("conversation:*")
            
            total_conversations = 0
            for conv_key in conversation_keys:
                conv_length = self.redis_client.llen(conv_key)
                total_conversations += conv_length
            
            stats = {
                "active_sessions": len(session_keys),
                "total_conversations": total_conversations,
                "max_history_per_session": self.max_history,
                "session_expiry_hours": self.session_expiry_seconds // 3600
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting memory stats: {str(e)}")
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """
        Check if Redis connection is healthy.
        
        Returns:
            bool: True if connection is healthy
        """
        try:
            self.redis_client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {str(e)}")
            return False
    
    def close(self) -> None:
        """Close Redis connection."""
        try:
            if self.redis_client:
                self.redis_client.close()
                logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")
