"""
Module Client for LUKi Core Agent
Handles communication with cognitive, engagement, reporting, and security modules
"""

import httpx
import logging
import json
from typing import Dict, List, Optional, Any
from .config import settings

logger = logging.getLogger(__name__)

class ModuleClient:
    """HTTP client for communicating with LUKi module services"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.cognitive_url = settings.cognitive_service_url
        self.engagement_url = settings.engagement_service_url
        self.security_url = settings.security_service_url
        self.reporting_url = settings.reporting_service_url
        
        logger.info(f"ModuleClient initialized with URLs:")
        logger.info(f"  Cognitive: {self.cognitive_url}")
        logger.info(f"  Engagement: {self.engagement_url}")
        logger.info(f"  Security: {self.security_url}")
        logger.info(f"  Reporting: {self.reporting_url}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all module services"""
        results = {}
        
        for name, url in [
            ("cognitive", self.cognitive_url),
            ("engagement", self.engagement_url),
            ("security", self.security_url),
            ("reporting", self.reporting_url)
        ]:
            try:
                response = await self.client.get(f"{url}/health", timeout=10.0)
                results[name] = response.status_code == 200
            except Exception as e:
                logger.warning(f"Health check failed for {name}: {e}")
                results[name] = False
        
        return results
    
    # Cognitive Module Methods
    async def get_recommendations(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get activity recommendations from cognitive module"""
        try:
            response = await self.client.post(
                f"{self.cognitive_url}/recommendations/{user_id}",
                json={"context": context}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return {"status": "error", "message": str(e)}
    
    async def analyze_patterns(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user patterns with cognitive module"""
        try:
            response = await self.client.post(
                f"{self.cognitive_url}/analyze/{user_id}",
                json=data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to analyze patterns: {e}")
            return {"status": "error", "message": str(e)}
    
    # Engagement Module Methods
    async def track_interaction(self, user_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track user interaction with engagement module"""
        try:
            response = await self.client.post(
                f"{self.engagement_url}/interactions/{user_id}",
                json=interaction_data
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to track interaction: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_engagement_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get engagement metrics for user"""
        try:
            response = await self.client.get(f"{self.engagement_url}/metrics/{user_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get engagement metrics: {e}")
            return {"status": "error", "message": str(e)}
    
    # Security Module Methods
    async def check_consent(self, user_id: str) -> Dict[str, Any]:
        """Check user consent status"""
        try:
            response = await self.client.get(f"{self.security_url}/consent/{user_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to check consent: {e}")
            return {"status": "error", "message": str(e)}
    
    async def update_privacy_settings(self, user_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update user privacy settings"""
        try:
            response = await self.client.post(
                f"{self.security_url}/privacy/{user_id}/settings",
                json=settings
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to update privacy settings: {e}")
            return {"status": "error", "message": str(e)}
    
    # Reporting Module Methods
    async def generate_wellbeing_report(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Generate wellbeing report for user"""
        try:
            response = await self.client.post(
                f"{self.reporting_url}/reports/{user_id}/wellbeing",
                params={"days": days}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to generate wellbeing report: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_user_trends(self, user_id: str) -> Dict[str, Any]:
        """Get user trends and patterns"""
        try:
            response = await self.client.get(f"{self.reporting_url}/reports/{user_id}/trends")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get user trends: {e}")
            return {"status": "error", "message": str(e)}
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Global module client instance
module_client = ModuleClient()
