"""
LUKi Memory Service Client

This module provides HTTP client functionality to connect with the luki-memory-service
for retrieving ELR (Electronic Life Record) data, user preferences, and insights.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import aiohttp
from aiohttp import ClientTimeout, ClientError

from ..config import settings

logger = logging.getLogger(__name__)

@dataclass
class ELRMemory:
    """Represents an ELR memory retrieved from the memory service"""
    memory_id: str
    content: str
    timestamp: datetime
    relevance_score: float
    memory_type: str  # 'activity', 'health', 'mood', 'goal', 'interaction'
    metadata: Dict[str, Any]
    embedding_vector: Optional[List[float]] = None

@dataclass
class UserInsight:
    """Represents a user insight derived from ELR analysis"""
    insight_id: str
    insight_text: str
    confidence_score: float
    insight_type: str  # 'pattern', 'trend', 'recommendation', 'correlation'
    supporting_memories: List[str]  # memory_ids
    created_at: datetime
    metadata: Dict[str, Any]

@dataclass
class UserPreferences:
    """User preferences and profile information"""
    user_id: str
    communication_style: str
    focus_areas: List[str]
    goals: List[str]
    interests: List[str]
    privacy_settings: Dict[str, Any]
    last_updated: datetime
    preferences: Dict[str, Any]

class MemoryServiceClient:
    """
    HTTP client for connecting to luki-memory-service
    
    Provides methods for retrieving ELR memories, user insights,
    preferences, and performing semantic searches.
    """
    
    def __init__(self, base_url: Optional[str] = None, timeout: int = 30):
        self.base_url = base_url or settings.memory_service_url or "http://localhost:8001"
        self.timeout = ClientTimeout(total=timeout)
        self.session = None
        self._auth_token = None
        
        logger.info(f"Initialized Memory Service Client for {self.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def _ensure_session(self):
        """Ensure we have an active session"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to memory service"""
        await self._ensure_session()
        
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "luki-core-agent/1.0"
        }
        
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
            if not self.session:
                raise ClientError("Session not initialized")
                
            async with self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=headers
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    logger.debug(f"Request successful: {response.status}")
                    return result
                elif response.status == 404:
                    logger.warning(f"Resource not found: {url}")
                    return {}
                else:
                    error_text = await response.text()
                    logger.error(f"Request failed: {response.status} - {error_text}")
                    raise ClientError(f"Memory service error: {response.status}")
                    
        except asyncio.TimeoutError:
            logger.error(f"Request timeout for {url}")
            raise
        except Exception as e:
            logger.error(f"Request error for {url}: {e}")
            raise
    
    async def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        memory_types: Optional[List[str]] = None,
        time_range: Optional[tuple] = None
    ) -> List[ELRMemory]:
        """
        Perform semantic search for relevant memories
        
        Args:
            user_id: User identifier
            query: Search query (will be embedded for semantic search)
            limit: Maximum number of results
            memory_types: Filter by memory types
            time_range: (start_date, end_date) tuple for time filtering
            
        Returns:
            List of relevant ELR memories
        """
        try:
            params = {
                "user_id": user_id,
                "query": query,
                "limit": limit
            }
            
            if memory_types:
                params["memory_types"] = ",".join(memory_types)
            
            if time_range:
                start_date, end_date = time_range
                params["start_date"] = start_date.isoformat()
                params["end_date"] = end_date.isoformat()
            
            result = await self._make_request("GET", "/api/v1/memories/search", params=params)
            
            memories = []
            for memory_data in result.get("memories", []):
                memory = ELRMemory(
                    memory_id=memory_data["memory_id"],
                    content=memory_data["content"],
                    timestamp=datetime.fromisoformat(memory_data["timestamp"]),
                    relevance_score=memory_data.get("relevance_score", 0.0),
                    memory_type=memory_data["memory_type"],
                    metadata=memory_data.get("metadata", {}),
                    embedding_vector=memory_data.get("embedding_vector")
                )
                memories.append(memory)
            
            logger.info(f"Retrieved {len(memories)} memories for query: {query[:50]}...")
            return memories
            
        except Exception as e:
            logger.error(f"Memory search error: {e}")
            return []
    
    async def get_user_insights(
        self,
        user_id: str,
        insight_types: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[UserInsight]:
        """
        Get user insights and patterns from ELR analysis
        
        Args:
            user_id: User identifier
            insight_types: Filter by insight types
            limit: Maximum number of insights
            
        Returns:
            List of user insights
        """
        try:
            params = {
                "user_id": user_id,
                "limit": limit
            }
            
            if insight_types:
                params["insight_types"] = ",".join(insight_types)
            
            result = await self._make_request("GET", "/api/v1/insights", params=params)
            
            insights = []
            for insight_data in result.get("insights", []):
                insight = UserInsight(
                    insight_id=insight_data["insight_id"],
                    insight_text=insight_data["insight_text"],
                    confidence_score=insight_data.get("confidence_score", 0.0),
                    insight_type=insight_data["insight_type"],
                    supporting_memories=insight_data.get("supporting_memories", []),
                    created_at=datetime.fromisoformat(insight_data["created_at"]),
                    metadata=insight_data.get("metadata", {})
                )
                insights.append(insight)
            
            logger.info(f"Retrieved {len(insights)} insights for user {user_id}")
            return insights
            
        except Exception as e:
            logger.error(f"User insights error: {e}")
            return []
    
    async def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """
        Get user preferences and profile information
        
        Args:
            user_id: User identifier
            
        Returns:
            User preferences or None if not found
        """
        try:
            result = await self._make_request("GET", f"/api/v1/users/{user_id}/preferences")
            
            if not result:
                return None
            
            preferences = UserPreferences(
                user_id=user_id,
                communication_style=result.get("communication_style", "supportive"),
                focus_areas=result.get("focus_areas", []),
                goals=result.get("goals", []),
                interests=result.get("interests", []),
                privacy_settings=result.get("privacy_settings", {}),
                last_updated=datetime.fromisoformat(result["last_updated"]),
                preferences=result.get("preferences", {})
            )
            
            logger.info(f"Retrieved preferences for user {user_id}")
            return preferences
            
        except Exception as e:
            logger.error(f"User preferences error: {e}")
            return None
    
    async def get_recent_activities(
        self,
        user_id: str,
        days: int = 7,
        activity_types: Optional[List[str]] = None
    ) -> List[ELRMemory]:
        """
        Get recent user activities from ELR
        
        Args:
            user_id: User identifier
            days: Number of days to look back
            activity_types: Filter by activity types
            
        Returns:
            List of recent activity memories
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            params = {
                "user_id": user_id,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "memory_type": "activity"
            }
            
            if activity_types:
                params["activity_types"] = ",".join(activity_types)
            
            result = await self._make_request("GET", "/api/v1/memories", params=params)
            
            activities = []
            for memory_data in result.get("memories", []):
                memory = ELRMemory(
                    memory_id=memory_data["memory_id"],
                    content=memory_data["content"],
                    timestamp=datetime.fromisoformat(memory_data["timestamp"]),
                    relevance_score=1.0,  # Recent activities are highly relevant
                    memory_type=memory_data["memory_type"],
                    metadata=memory_data.get("metadata", {})
                )
                activities.append(memory)
            
            logger.info(f"Retrieved {len(activities)} recent activities for user {user_id}")
            return activities
            
        except Exception as e:
            logger.error(f"Recent activities error: {e}")
            return []
    
    async def get_health_metrics(
        self,
        user_id: str,
        metric_types: Optional[List[str]] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get user health metrics and trends
        
        Args:
            user_id: User identifier
            metric_types: Filter by metric types (sleep, exercise, mood, etc.)
            days: Number of days to analyze
            
        Returns:
            Health metrics and trends
        """
        try:
            params = {
                "user_id": user_id,
                "days": days
            }
            
            if metric_types:
                params["metric_types"] = ",".join(metric_types)
            
            result = await self._make_request("GET", "/api/v1/health/metrics", params=params)
            
            logger.info(f"Retrieved health metrics for user {user_id}")
            return result.get("metrics", {})
            
        except Exception as e:
            logger.error(f"Health metrics error: {e}")
            return {}
    
    async def store_interaction(
        self,
        user_id: str,
        interaction_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a new interaction in the user's ELR
        
        Args:
            user_id: User identifier
            interaction_type: Type of interaction (chat, goal_set, etc.)
            content: Interaction content
            metadata: Additional metadata
            
        Returns:
            Success status
        """
        try:
            data = {
                "user_id": user_id,
                "interaction_type": interaction_type,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            await self._make_request("POST", "/api/v1/interactions", data=data)
            
            logger.info(f"Stored interaction for user {user_id}: {interaction_type}")
            return True
            
        except Exception as e:
            logger.error(f"Store interaction error: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check if memory service is available"""
        try:
            result = await self._make_request("GET", "/health")
            return result.get("status") == "healthy"
        except Exception as e:
            logger.error(f"Memory service health check failed: {e}")
            return False
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Closed memory service client session")

# Global client instance for reuse
_memory_client = None

async def get_memory_client() -> MemoryServiceClient:
    """Get or create global memory client instance"""
    global _memory_client
    if _memory_client is None:
        _memory_client = MemoryServiceClient()
    return _memory_client

async def close_memory_client():
    """Close global memory client"""
    global _memory_client
    if _memory_client:
        await _memory_client.close()
        _memory_client = None
