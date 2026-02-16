"""
Module Client for LUKi Core Agent
Handles communication with cognitive, engagement, reporting, and security modules.

Uses per-service circuit breakers and retry with exponential backoff from
resilience.py so transient downstream failures don't cascade into user-
visible errors.
"""

import asyncio
import httpx
import logging
import json
from typing import Dict, List, Optional, Any
from .config import settings
from .resilience import RetryConfig, CircuitBreaker

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resilience defaults – tuned for inter-service calls
# ---------------------------------------------------------------------------
_MODULE_RETRY_CONFIG = RetryConfig(
    max_attempts=3,
    initial_delay=0.5,
    max_delay=8.0,
    exponential_base=2.0,
    jitter=True,
    retryable_exceptions=(httpx.ConnectError, httpx.ReadTimeout, httpx.PoolTimeout),
)


class ModuleClient:
    """HTTP client for communicating with LUKi module services.

    Each downstream service gets its own :class:`CircuitBreaker` so a single
    unhealthy module cannot block calls to healthy ones.  Transient failures
    (connection errors, timeouts) are retried with exponential backoff before
    the circuit breaker trips.
    """

    def __init__(self) -> None:
        self.client = httpx.AsyncClient(timeout=30.0)
        self.cognitive_url = settings.cognitive_service_url
        self.engagement_url = settings.engagement_service_url
        self.security_url = settings.security_service_url
        self.reporting_url = settings.reporting_service_url

        # Per-service circuit breakers
        self._breakers: Dict[str, CircuitBreaker] = {
            "cognitive": CircuitBreaker(failure_threshold=5, recovery_timeout=30.0),
            "engagement": CircuitBreaker(failure_threshold=5, recovery_timeout=30.0),
            "security": CircuitBreaker(failure_threshold=5, recovery_timeout=30.0),
            "reporting": CircuitBreaker(failure_threshold=5, recovery_timeout=30.0),
        }

        logger.info(
            "ModuleClient initialised with circuit breakers",
            extra={
                "cognitive": self.cognitive_url,
                "engagement": self.engagement_url,
                "security": self.security_url,
                "reporting": self.reporting_url,
            },
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    # ------------------------------------------------------------------
    # Internal resilient request helper
    # ------------------------------------------------------------------

    async def _resilient_request(
        self,
        service_name: str,
        method: str,
        url: str,
        *,
        retry_config: Optional[RetryConfig] = None,
        **kwargs,
    ) -> httpx.Response:
        """Execute an HTTP request with retry + circuit breaker.

        Retries on transient exceptions (connect, timeout, pool) up to
        ``retry_config.max_attempts``, then lets the circuit breaker
        record the failure.  Non-retryable HTTP errors (4xx) are raised
        immediately without consuming retry budget.
        """
        cfg = retry_config or _MODULE_RETRY_CONFIG
        breaker = self._breakers.get(service_name)

        async def _do_request() -> httpx.Response:
            last_exc: Optional[Exception] = None
            for attempt in range(cfg.max_attempts):
                try:
                    response = await self.client.request(method, url, **kwargs)
                    response.raise_for_status()
                    return response
                except cfg.retryable_exceptions as exc:
                    last_exc = exc
                    if attempt < cfg.max_attempts - 1:
                        delay = cfg.calculate_delay(attempt)
                        logger.warning(
                            "Retrying %s %s (%d/%d) in %.2fs: %s",
                            method, url, attempt + 1, cfg.max_attempts, delay, exc,
                            extra={"service": service_name, "attempt": attempt + 1},
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "All %d retries exhausted for %s %s",
                            cfg.max_attempts, method, url,
                            extra={"service": service_name, "error": str(exc)},
                        )
                except httpx.HTTPStatusError:
                    # 4xx / non-retryable errors – propagate immediately
                    raise
            raise last_exc  # type: ignore[misc]

        if breaker:
            return await breaker.call_async(_do_request)
        return await _do_request()

    # ------------------------------------------------------------------
    # Circuit breaker status (useful for /health introspection)
    # ------------------------------------------------------------------

    def get_circuit_status(self) -> Dict[str, str]:
        """Return the current circuit breaker state for each service."""
        return {name: cb.state for name, cb in self._breakers.items()}

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all module services"""
        results = {}

        for name, url in [
            ("cognitive", self.cognitive_url),
            ("engagement", self.engagement_url),
            ("security", self.security_url),
            ("reporting", self.reporting_url),
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

        This method is aligned with the cognitive service's ``/recommendations``
        endpoint.  Uses resilient request with retry + circuit breaker.
        """
        try:
            safe_context: Dict[str, Any] = context or {}

            payload: Dict[str, Any] = {
                "user_id": user_id,
                "context": safe_context,
                "current_mood": safe_context.get("current_mood"),
                "available_duration": safe_context.get("available_duration"),
                "carer_available": safe_context.get("carer_available", True),
                "group_setting": safe_context.get("group_setting", False),
                "specific_request": safe_context.get("specific_request"),
                "max_recommendations": safe_context.get("max_recommendations"),
            }

            try:
                response = await self._resilient_request(
                    "cognitive", "POST",
                    f"{self.cognitive_url}/recommendations",
                    json=payload,
                )
                return response.json()
            except httpx.HTTPStatusError as e:
                try:
                    detail: Any = e.response.json()
                except Exception:
                    detail = e.response.text

                logger.error(
                    "Failed to get recommendations (HTTP %s): %s",
                    e.response.status_code, detail,
                )
                return {
                    "status": "error",
                    "status_code": e.response.status_code,
                    "error": "http_error",
                    "detail": detail,
                }
        except Exception as e:
            logger.error(f"Failed to get recommendations: {e}")
            return {"status": "error", "message": str(e)}

    async def get_world_day_activities(self, user_id: str) -> Dict[str, Any]:
        """Get today's world day activities (ReMeMades) from cognitive module"""
        try:
            response = await self._resilient_request(
                "cognitive", "GET",
                f"{self.cognitive_url}/world-day-activities/{user_id}",
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get world day activities: {e}")
            return {"status": "error", "message": str(e)}
    
    # Life Story Recording Methods
    async def start_life_story_session(self, user_id: str) -> Dict[str, Any]:
        """Start a new life story recording session"""
        try:
            response = await self._resilient_request(
                "cognitive", "POST",
                f"{self.cognitive_url}/life-story/start",
                json={"user_id": user_id},
            )
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
            payload: Dict[str, Any] = {
                "user_id": user_id,
                "session_id": session_id,
                "response_text": response_text,
                "skip_phase": skip_phase,
            }
            if approximate_date:
                payload["approximate_date"] = approximate_date

            response = await self._resilient_request(
                "cognitive", "POST",
                f"{self.cognitive_url}/life-story/continue",
                json=payload,
            )
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
            response = await self._resilient_request(
                "cognitive", "GET",
                f"{self.cognitive_url}/life-story/sessions/{user_id}",
                params=params,
            )
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
            response = await self._resilient_request(
                "cognitive", "DELETE",
                f"{self.cognitive_url}/life-story/sessions/{session_id}",
                params={"user_id": user_id},
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to delete life story session: {e}")
            return {"status": "error", "message": str(e)}
    
    async def get_life_story_phases(self) -> Dict[str, Any]:
        """Get all available life story phases"""
        try:
            response = await self._resilient_request(
                "cognitive", "GET",
                f"{self.cognitive_url}/life-story/phases",
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get life story phases: {e}")
            return {"status": "error", "message": str(e)}
    
    async def analyze_patterns(self, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user patterns with cognitive module"""
        try:
            response = await self._resilient_request(
                "cognitive", "POST",
                f"{self.cognitive_url}/analyze/{user_id}",
                json=data,
            )
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
        account_tier: str = "free",
    ) -> Dict[str, Any]:
        """Call cognitive image service for photo reminiscence.

        Special-case HTTP 429 from the cognitive module so that callers can
        surface a friendly cooldown message to the user instead of treating it
        as a generic 5xx-style failure.
        
        Args:
            account_tier: User's subscription tier (free, plus, pro) for rate limiting
        """

        try:
            payload: Dict[str, Any] = {
                "user_id": user_id,
                "activity_title": activity_title,
                "answers": answers,
                "n": n,
                "account_tier": account_tier,
            }
            # Use extended timeout for image generation (Together API can take 45+ seconds)
            response = await self.client.post(
                f"{self.cognitive_url}/images/photo-reminiscence",
                json=payload,
                timeout=90.0,
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

            response = await self._resilient_request(
                "engagement", "POST",
                f"{self.engagement_url}/interactions",
                json=payload,
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to track interaction: {e}")
            return {"status": "error", "message": str(e)}

    async def get_engagement_metrics(self, user_id: str) -> Dict[str, Any]:
        """Get engagement metrics for user"""
        try:
            response = await self._resilient_request(
                "engagement", "GET",
                f"{self.engagement_url}/metrics/{user_id}",
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get engagement metrics: {e}")
            return {"status": "error", "message": str(e)}
    
    # Security Module Methods
    async def check_consent(self, user_id: str) -> Dict[str, Any]:
        """Check user consent status"""
        try:
            response = await self._resilient_request(
                "security", "GET",
                f"{self.security_url}/consent/{user_id}",
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to check consent: {e}")
            return {"status": "error", "message": str(e)}

    async def update_privacy_settings(self, user_id: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update user privacy settings"""
        try:
            response = await self._resilient_request(
                "security", "POST",
                f"{self.security_url}/privacy/{user_id}/settings",
                json=settings,
            )
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
            response = await self._resilient_request(
                "reporting", "POST",
                f"{self.reporting_url}/reports/{user_id}/wellbeing",
                params={"days": days},
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to generate wellbeing report: {e}")
            return {"status": "error", "message": str(e)}

    async def get_user_trends(self, user_id: str) -> Dict[str, Any]:
        """Get user trends and patterns"""
        try:
            response = await self._resilient_request(
                "reporting", "GET",
                f"{self.reporting_url}/reports/{user_id}/trends",
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get user trends: {e}")
            return {"status": "error", "message": str(e)}
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Global module client instance
module_client = ModuleClient()
