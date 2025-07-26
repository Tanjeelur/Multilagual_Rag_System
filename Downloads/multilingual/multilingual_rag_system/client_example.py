"""
Example client for testing the RAG API.
This script demonstrates how to interact with the API endpoints.
"""

import requests
import json
import time
from typing import Dict, Any, Optional

class RAGClient:
    """Client for interacting with the RAG API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the RAG client.
        
        Args:
            base_url (str): Base URL of the RAG API server
        """
        self.base_url = base_url
        self.session_id = None
        
    def check_status(self) -> Dict[str, Any]:
        """Check API status."""
        try:
            response = requests.get(f"{self.base_url}/status")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def query(self, query: str, session_id: Optional[str] = None, include_sources: bool = True) -> Dict[str, Any]:
        """Send a query to the RAG system."""
        payload = {
            "query": query,
            "include_sources": include_sources
        }
        
        if session_id:
            payload["session_id"] = session_id
        
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            self.session_id = result.get("session_id")
            return result
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_history(self, session_id: str) -> Dict[str, Any]:
        """Get conversation history."""
        try:
            response = requests.get(f"{self.base_url}/history/{session_id}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def clear_history(self, session_id: str) -> Dict[str, Any]:
        """Clear conversation history."""
        try:
            response = requests.delete(f"{self.base_url}/history/{session_id}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_test_queries(self) -> Dict[str, Any]:
        """Get sample test queries."""
        try:
            response = requests.get(f"{self.base_url}/test-queries")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        try:
            response = requests.get(f"{self.base_url}/statistics")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": str(e)}

def main():
    """Test the RAG API with various queries."""
    print("=== Multilingual RAG System Client ===\n")
    
    client = RAGClient()
    
    # Check status
    print("1. 🔍 Checking API status...")
    status = client.check_status()
    
    if "error" in status:
        print(f"❌ Error connecting to API: {status['error']}")
        print("Make sure the server is running with: python run_server.py")
        return
    
    print(json.dumps(status, indent=2, ensure_ascii=False))
    
    if not status.get("system_ready", False):
        print("\n⏳ API is initializing. Waiting 30 seconds...")
        time.sleep(30)
        
        # Check again
        status = client.check_status()
        if not status.get("system_ready", False):
            print("❌ API is not ready. Please wait longer and try again.")
            return
    
    print("\n✅ API is ready!")
    
    # Get test queries
    print("\n2. 📝 Getting test queries...")
    test_data = client.get_test_queries()
    
    if "error" in test_data:
        print(f"❌ Error getting test queries: {test_data['error']}")
        return
    
    # Test Bengali queries
    print("\n3. 🔍 Testing Bengali queries...")
    bengali_queries = test_data.get("bengali_queries", [])
    
    session_id = None
    
    for i, query in enumerate(bengali_queries[:3], 1):  # Test first 3
        print(f"\n--- Query {i} ---")
        print(f"❓ Question: {query}")
        
        result = client.query(query, session_id, include_sources=True)
        
        if "error" not in result:
            print(f"💬 Answer: {result['response']}")
            print(f"🌐 Language: {result['language']}")
            print(f"📄 Retrieved documents: {result['retrieved_documents']}")
            
            if not session_id:
                session_id = result['session_id']
                print(f"🆔 Session ID: {session_id[:8]}...")
            
            # Show sources
            if result.get('sources'):
                print("📚 Sources:")
                for j, source in enumerate(result['sources'][:2], 1):
                    print(f"   {j}. Page {source['page_number']} (similarity: {source['similarity_score']:.3f})")
                    print(f"      Preview: {source['content_preview']}")
        else:
            print(f"❌ Error: {result['error']}")
        
        time.sleep(2)  # Brief pause between queries
    
    # Test English queries
    print("\n4. 🔍 Testing English queries...")
    english_queries = test_data.get("english_queries", [])
    
    for i, query in enumerate(english_queries[:2], 1):  # Test first 2
        print(f"\n--- English Query {i} ---")
        print(f"❓ Question: {query}")
        
        result = client.query(query, session_id, include_sources=False)
        
        if "error" not in result:
            print(f"💬 Answer: {result['response']}")
            print(f"🌐 Language: {result['language']}")
        else:
            print(f"❌ Error: {result['error']}")
        
        time.sleep(2)
    
    # Get conversation history
    if session_id:
        print(f"\n5. 💭 Getting conversation history...")
        history = client.get_history(session_id)
        
        if "error" not in history:
            print(f"📊 Total conversations: {history['total_turns']}")
            print("Recent conversations:")
            for turn in history['history'][-3:]:  # Show last 3
                print(f"   👤 User: {turn['user_query'][:60]}...")
                print(f"   🤖 Bot: {turn['assistant_response'][:60]}...")
                print(f"   ⏰ Time: {turn['timestamp']}")
                print()
        else:
            print(f"❌ Error getting history: {history['error']}")
    
    # Get system statistics
    print("6. 📊 Getting system statistics...")
    stats = client.get_statistics()
    
    if "error" not in stats:
        ltm = stats.get('long_term_memory', {})
        stm = stats.get('short_term_memory', {})
        
        print(f"   📚 Total documents: {ltm.get('total_documents', 0)}")
        print(f"   💬 Active sessions: {stm.get('active_sessions', 0)}")
        print(f"   🔧 Embedding model: {ltm.get('embedding_model', 'N/A')}")
        print(f"   ✅ System health: {'Healthy' if stats.get('system_health', {}).get('redis_healthy') else 'Degraded'}")
    else:
        print(f"❌ Error getting statistics: {stats['error']}")
    
    print("\n=== Testing completed ===")
    print(f"🎯 Total queries tested: 5")
    print(f"🆔 Session used: {session_id[:8] if session_id else 'None'}...")
    print("✅ Demo finished successfully!")

if __name__ == "__main__":
    main()
