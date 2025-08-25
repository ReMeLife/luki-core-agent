#!/usr/bin/env python3
"""
Memory Service API Client for LUKi Core Agent

Provides interface to the LUKi Memory Service for ELR retrieval and search.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemorySearchResult:
    """Result from memory search"""
    content: str
    similarity_score: float
    metadata: Dict[str, Any]
    chunk_id: str
    created_at: datetime


@dataclass
class UserMemoryStats:
    """User memory statistics"""
    user_id: str
    total_memories: int
    total_chunks: int
    content_type_breakdown: Dict[str, int]
    sensitivity_breakdown: Dict[str, int]
    earliest_memory: Optional[datetime]
    latest_memory: Optional[datetime]
    storage_size_mb: float


class MemoryServiceClient:
    """Client for LUKi Memory Service API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """Initialize memory service client
        
        Args:
            base_url: Base URL of the memory service API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._session = None
        self._auth_token = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
    
    async def connect(self):
        """Initialize HTTP session and authenticate"""
        if self._session is None:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
        
        # Get service token for authentication
        await self._authenticate()
    
    async def disconnect(self):
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None
    
    async def _authenticate(self):
        """Get service authentication token"""
        try:
            async with self._session.post(f"{self.base_url}/auth/service-token") as response:
                if response.status == 200:
                    data = await response.json()
                    self._auth_token = data.get('access_token')
                    logger.info("Successfully authenticated with memory service")
                else:
                    error_text = await response.text()
                    logger.error(f"Authentication failed: {response.status} - {error_text}")
                    raise Exception(f"Authentication failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to authenticate with memory service: {e}")
            raise
    
    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers"""
        if not self._auth_token:
            raise Exception("Not authenticated - call connect() first")
        return {
            'Authorization': f'Bearer {self._auth_token}',
            'Content-Type': 'application/json'
        }
    
    async def search_memories(
        self,
        user_id: str,
        query: str,
        k: int = 5,
        content_types: Optional[List[str]] = None,
        sensitivity_filter: Optional[List[str]] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[MemorySearchResult]:
        """Search user memories using semantic similarity
        
        Args:
            user_id: User identifier
            query: Search query
            k: Number of results to return
            content_types: Filter by content types
            sensitivity_filter: Filter by sensitivity levels
            date_from: Filter memories from this date
            date_to: Filter memories to this date
            
        Returns:
            List of memory search results
        """
        if not self._session:
            await self.connect()
        
        request_data = {
            "user_id": user_id,
            "query": query,
            "k": k
        }
        
        if content_types:
            request_data["content_types"] = content_types
        if sensitivity_filter:
            request_data["sensitivity_filter"] = sensitivity_filter
        if date_from:
            request_data["date_from"] = date_from.isoformat()
        if date_to:
            request_data["date_to"] = date_to.isoformat()
        
        try:
            async with self._session.post(
                f"{self.base_url}/search/memories",
                json=request_data,
                headers=self._get_auth_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success", False):
                        results = []
                        for result in data.get("results", []):
                            results.append(MemorySearchResult(
                                content=result["content"],
                                similarity_score=result["similarity_score"],
                                metadata=result["metadata"],
                                chunk_id=result["chunk_id"],
                                created_at=datetime.fromisoformat(result["created_at"].replace('Z', '+00:00'))
                            ))
                        return results
                    else:
                        logger.warning(f"Memory search failed for user {user_id}: {data}")
                        return []
                else:
                    error_text = await response.text()
                    logger.error(f"Memory search request failed: {response.status} - {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Error searching memories for user {user_id}: {e}")
            return []
    
    async def get_user_memory_stats(self, user_id: str) -> Optional[UserMemoryStats]:
        """Get comprehensive statistics about a user's stored memories
        
        Args:
            user_id: User identifier
            
        Returns:
            User memory statistics or None if failed
        """
        if not self._session:
            await self.connect()
        
        try:
            async with self._session.get(
                f"{self.base_url}/search/memories/{user_id}/stats",
                headers=self._get_auth_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return UserMemoryStats(
                        user_id=data["user_id"],
                        total_memories=data["total_memories"],
                        total_chunks=data["total_chunks"],
                        content_type_breakdown=data["content_type_breakdown"],
                        sensitivity_breakdown=data["sensitivity_breakdown"],
                        earliest_memory=datetime.fromisoformat(data["earliest_memory"].replace('Z', '+00:00')) if data.get("earliest_memory") else None,
                        latest_memory=datetime.fromisoformat(data["latest_memory"].replace('Z', '+00:00')) if data.get("latest_memory") else None,
                        storage_size_mb=data["storage_size_mb"]
                    )
                else:
                    error_text = await response.text()
                    logger.error(f"Memory stats request failed: {response.status} - {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Error getting memory stats for user {user_id}: {e}")
            return None
    
    async def find_similar_memories(
        self,
        user_id: str,
        chunk_id: str,
        k: int = 5
    ) -> List[MemorySearchResult]:
        """Find memories similar to a specific memory chunk
        
        Args:
            user_id: User identifier
            chunk_id: ID of the reference memory chunk
            k: Number of similar memories to return
            
        Returns:
            List of similar memory results
        """
        if not self._session:
            await self.connect()
        
        try:
            async with self._session.get(
                f"{self.base_url}/search/memories/{user_id}/similar/{chunk_id}",
                params={"k": k},
                headers=self._get_auth_headers()
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("success", False):
                        results = []
                        for result in data.get("similar_memories", []):
                            results.append(MemorySearchResult(
                                content=result["content"],
                                similarity_score=result["similarity_score"],
                                metadata=result["metadata"],
                                chunk_id=result["chunk_id"],
                                created_at=datetime.fromisoformat(result["created_at"].replace('Z', '+00:00'))
                            ))
                        return results
                    else:
                        logger.warning(f"Similar memory search failed: {data}")
                        return []
                else:
                    error_text = await response.text()
                    logger.error(f"Similar memory request failed: {response.status} - {error_text}")
                    return []
        except Exception as e:
            logger.error(f"Error finding similar memories: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if memory service is healthy
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            if not self._session:
                await self.connect()
            
            async with self._session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("status") == "healthy"
                return False
        except Exception as e:
            logger.error(f"Memory service health check failed: {e}")
            return False


# Singleton instance for global use
_memory_client = None

async def get_memory_client() -> MemoryServiceClient:
    """Get global memory service client instance"""
    global _memory_client
    if _memory_client is None:
        _memory_client = MemoryServiceClient()
        await _memory_client.connect()
    return _memory_client

async def close_memory_client():
    """Close global memory service client"""
    global _memory_client
    if _memory_client:
        await _memory_client.disconnect()
        _memory_client = None
