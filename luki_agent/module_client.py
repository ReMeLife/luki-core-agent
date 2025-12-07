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
        """Get activity recommendations from cognitive module.

        This method is aligned with the cognitive service's `/recommendations`
        endpoint, which expects a RecommendationRequest-style JSON body with
        `user_id`, an optional `context` dict, and optional top-level fields
        like `current_mood`, `available_duration`, etc.
        """
        try:
            safe_context: Dict[str, Any] = context or {}

            # Build payload compatible with luki-modules-cognitive main API
            payload: Dict[str, Any] = {
                "user_id": user_id,
                "context": safe_context,
                # Mirror commonly used fields from context so the service can
                # access them directly via the request model.
                "current_mood": safe_context.get("current_mood"),
                "available_duration": safe_context.get("available_duration"),
                "carer_available": safe_context.get("carer_available", True),
                "group_setting": safe_context.get("group_setting", False),
                "specific_request": safe_context.get("specific_request"),
                "max_recommendations": safe_context.get("max_recommendations"),
            }

            try:
                response = await self.client.post(
                    f"{self.cognitive_url}/recommendations",
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                # Surface structured error information so tools can distinguish
                # policy/consent denials (e.g. 403) from generic failures.
                try:
                    detail: Any = e.response.json()
                except Exception:
                    detail = e.response.text

                logger.error(
                    "Failed to get recommendations (HTTP %s): %s",
                    e.response.status_code,
                    detail,
                )
                return {
                    "status": "error",
                    "status_code": e.response.status_code,
                    "error": "http_error",
                    "detail": detail,
                }
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            # Keep a simple error shape so downstream tools can detect failures
            return {"status": "error", "message": str(e)}

    async def get_world_day_activities(self, user_id: str) -> Dict[str, Any]:
        """Get today's world day activities (ReMeMades) from cognitive module"""
        try:
            response = await self.client.get(
                f"{self.cognitive_url}/world-day-activities/{user_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get world day activities: {e}")
            return {"status": "error", "message": str(e)}
    
    # Life Story Recording Methods
    async def start_life_story_session(self, user_id: str) -> Dict[str, Any]:
        """Start a new life story recording session"""
        try:
            response = await self.client.post(
                f"{self.cognitive_url}/life-story/start",
                json={"user_id": user_id}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text
            logger.error(f"Failed to start life story session (HTTP {e.response.status_code}): {detail}")
            return {
                "status": "error",
                "status_code": e.response.status_code,
                "error": "http_error",
                "detail": detail,
            }
        except Exception as e:
            logger.error(f"Failed to start life story session: {e}")
            return {"status": "error", "message": str(e)}
    
    async def continue_life_story_session(
        self,
        user_id: str,
        session_id: str,
        response_text: str,
        skip_phase: bool = False,
        approximate_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Continue a life story session with a new response"""
        try:
            payload = {
                "user_id": user_id,
                "session_id": session_id,
                "response_text": response_text,
                "skip_phase": skip_phase,
            }
            if approximate_date:
                payload["approximate_date"] = approximate_date
            
            response = await self.client.post(
                f"{self.cognitive_url}/life-story/continue",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            try:
                detail = e.response.json()
            except Exception:
                detail = e.response.text
            logger.error(f"Failed to continue life story session (HTTP {e.response.status_code}): {detail}")
            return {
                "status": "error",
                "status_code": e.response.status_code,
                "error": "http_error",
                "detail": detail,
            }
        except Exception as e:
            logger.error(f"Failed to continue life story session: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_life_story_sessions(
        self,
        user_id: str,
        include_chunks: bool = False,
    ) -> Dict[str, Any]:
        """Get all life story sessions for a user"""
        try:
            params = {"include_chunks": str(include_chunks).lower()}
            response = await self.client.get(
                f"{self.cognitive_url}/life-story/sessions/{user_id}",
                params=params
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get life story sessions: {e}")
            return {"status": "error", "message": str(e)}
    
    async def delete_life_story_session(
        self,
        user_id: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """Delete a life story session"""
        try:
            response = await self.client.delete(
                f"{self.cognitive_url}/life-story/sessions/{session_id}",
                params={"user_id": user_id}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to delete life story session: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_life_story_phases(self) -> Dict[str, Any]:
        """Get all available life story phases"""
        try:
            response = await self.client.get(
                f"{self.cognitive_url}/life-story/phases"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get life story phases: {e}")
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
    
    async def generate_photo_reminiscence_images(
        self,
        user_id: str,
        activity_title: Optional[str],
        answers: List[str],
        n: int = 1,
    ) -> Dict[str, Any]:
        """Call cognitive image service for photo reminiscence.

        Special-case HTTP 429 from the cognitive module so that callers can
        surface a friendly cooldown message to the user instead of treating it
        as a generic 5xx-style failure.
        """

        try:
            payload: Dict[str, Any] = {
                "user_id": user_id,
                "activity_title": activity_title,
                "answers": answers,
                "n": n,
            }
            response = await self.client.post(
                f"{self.cognitive_url}/images/photo-reminiscence",
                json=payload,
            )

            # If the cognitive service returns a 429, preserve the structured
            # detail so upstream layers (core API, gateway, frontends) can
            # display a clear cooldown message.
            if response.status_code == 429:
                try:
                    raw = response.json()
                except ValueError:
                    raw = {"detail": response.text}

                if isinstance(raw, dict):
                    container = raw
                    inner = container.get("detail", raw)
                else:
                    inner = {"message": str(raw)}

                return {
                    "status": "rate_limited",
                    "status_code": 429,
                    "detail": inner,
                }

            # All other non-success statuses are treated as generic HTTP errors
            # and reported back to the caller with the status code for context.
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            try:
                detail: Any
                try:
                    detail = e.response.json()
                except ValueError:
                    detail = e.response.text
            except Exception:
                detail = "<unparseable error body>"

            logger.error(
                "Failed to generate photo reminiscence images (HTTP %s): %s",
                e.response.status_code,
                detail,
            )
            return {
                "status": "error",
                "status_code": e.response.status_code,
                "error": "http_error",
                "detail": detail,
            }
        except Exception as e:
            logger.error(f"Failed to generate photo reminiscence images: {e}")
            return {"status": "error", "message": str(e)}
    
    # Engagement Module Methods
    async def track_interaction(self, user_id: str, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track user interaction with engagement module"""
        try:
            # Infer a high-level interaction_type for the engagement service
            interaction_type = (
                interaction_data.get("interaction_type")
                or interaction_data.get("request_type")
                or "interaction"
            )

            payload = {
                "user_id": user_id,
                "interaction_type": interaction_type,
                "content": interaction_data,
            }

            response = await self.client.post(
                f"{self.engagement_url}/interactions",
                json=payload,
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
    
    async def enforce_policy(
        self,
        user_id: str,
        requested_scopes: List[str],
        requester_role: str = "agent",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "user_id": user_id,
            "requester_role": requester_role,
            "requested_scopes": requested_scopes,
        }
        if context:
            payload["context"] = context

        try:
            response = await self.client.post(
                f"{self.security_url}/policy/enforce",
                json=payload,
            )
            try:
                raw = response.json()
            except ValueError:
                raw = {"detail": response.text}

            data: Dict[str, Any]
            if isinstance(raw, dict):
                data = raw
            else:
                data = {"detail": raw}

            if response.status_code == 200:
                return {
                    "allowed": bool(data.get("allowed", True)),
                    "scopes_checked": data.get("scopes_checked", []),
                    "reason": data.get("reason", "consent_valid"),
                    "detail": data.get("detail"),
                }

            return {
                "allowed": False,
                "error": data.get("error", "policy_denied"),
                "detail": data.get("detail"),
                "status_code": response.status_code,
            }
        except Exception as e:
            logger.error(f"Policy enforcement request failed: {e}")
            return {
                "allowed": False,
                "error": "policy_request_failed",
                "detail": str(e),
            }
    
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
