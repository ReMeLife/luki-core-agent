"""
Memory Retriever for LUKi Agent

Interfaces with luki-memory-service to retrieve relevant ELR snippets and user data
for context building and personalization.
"""

import httpx
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..config import settings


@dataclass
class MemorySearchResult:
    """Result from memory search"""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    chunk_id: str


class MemoryRetriever:
    """
    Retrieves relevant memories from luki-memory-service
    """
    
    def __init__(self, memory_service_url: Optional[str] = None):
        self.memory_service_url = memory_service_url or settings.memory_service_url
        self.timeout = settings.memory_service_timeout
        self.session = httpx.AsyncClient(timeout=self.timeout)
    
    async def search_memories(
        self,
        query: str,
        user_id: str,
        top_k: int = 6,
        similarity_threshold: float = 0.25,
        content_types: Optional[List[str]] = None,
        time_range: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories using vector similarity
        
        Args:
            query: Search query text
            user_id: User identifier
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            content_types: Filter by content types (elr, activity, etc.)
            time_range: Filter by time range
            
        Returns:
            List of relevant memory chunks
        """
        try:
            search_payload = {
                "query": query,
                "user_id": user_id,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
            }
            
            if content_types:
                search_payload["content_types"] = content_types
            
            if time_range:
                search_payload["time_range"] = time_range
            
            response = await self.session.post(
                f"{self.memory_service_url}/v1/search/semantic",
                json=search_payload,
                headers=self._get_headers()
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get("results", [])
            
        except Exception as e:
            print(f"Memory search error: {e}")
            return []
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get user profile data (preferences, facts, etc.)
        
        Args:
            user_id: User identifier
            
        Returns:
            User profile dictionary
        """
        try:
            response = await self.session.get(
                f"{self.memory_service_url}/v1/kv/profile/{user_id}",
                headers=self._get_headers()
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Profile retrieval error: {e}")
            return {}
    
    async def get_recent_activities(
        self, 
        user_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent user activities
        
        Args:
            user_id: User identifier
            limit: Maximum number of activities
            
        Returns:
            List of recent activities
        """
        try:
            response = await self.session.get(
                f"{self.memory_service_url}/v1/activities/recent",
                params={"user_id": user_id, "limit": limit},
                headers=self._get_headers()
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get("activities", [])
            
        except Exception as e:
            print(f"Recent activities error: {e}")
            return []
    
    async def update_conversation_summary(
        self,
        user_id: str,
        session_id: str,
        summary: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update conversation summary in memory service
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            summary: Conversation summary
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            payload = {
                "user_id": user_id,
                "session_id": session_id,
                "summary": summary,
                "metadata": metadata or {}
            }
            
            response = await self.session.post(
                f"{self.memory_service_url}/v1/sessions/summary",
                json=payload,
                headers=self._get_headers()
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            print(f"Summary update error: {e}")
            return False
    
    async def store_conversation_turn(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a conversation turn
        
        Args:
            user_id: User identifier
            session_id: Session identifier
            role: Role (user, assistant)
            content: Message content
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            payload = {
                "user_id": user_id,
                "session_id": session_id,
                "role": role,
                "content": content,
                "metadata": metadata or {}
            }
            
            response = await self.session.post(
                f"{self.memory_service_url}/v1/sessions/turn",
                json=payload,
                headers=self._get_headers()
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            print(f"Turn storage error: {e}")
            return False
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"luki-core-agent/{settings.service_version}"
        }
        
        if settings.modules_token:
            headers["Authorization"] = f"Bearer {settings.modules_token}"
        
        return headers
    
    async def close(self):
        """Close the HTTP session"""
        await self.session.aclose()
    
    def __del__(self):
        """Cleanup on deletion"""
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.close())
        except:
            pass
