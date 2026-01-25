"""
Resilience Patterns for LLM Calls

Provides retry, circuit breaker, and fallback patterns for LLM API calls
to handle rate limits, transient failures, and service degradation.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, Callable, List
from datetime import datetime

from luki_agent.resilience import (
    RetryConfig,
    retry_with_backoff,
    CircuitBreaker,
    with_timeout
)
from luki_agent.observability import metrics, track_latency

logger = logging.getLogger(__name__)


class LLMRateLimitError(Exception):
    """Raised when LLM API rate limit is exceeded"""
    pass


class LLMServiceError(Exception):
    """Raised when LLM service is unavailable"""
    pass


# Circuit breakers for different LLM providers
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_circuit_breaker(provider: str) -> CircuitBreaker:
    """Get or create circuit breaker for LLM provider"""
    if provider not in _circuit_breakers:
        _circuit_breakers[provider] = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=LLMServiceError
        )
    return _circuit_breakers[provider]


class LLMClient:
    """
    Resilient LLM client with retry, circuit breaker, and fallback patterns.
    """
    
    def __init__(
        self,
        primary_provider: str,
        fallback_providers: Optional[List[str]] = None,
        timeout_seconds: float = 30.0
    ):
        """
        Initialize LLM client with resilience patterns.
        
        Args:
            primary_provider: Primary LLM provider name
            fallback_providers: Optional list of fallback providers
            timeout_seconds: Request timeout
        """
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or []
        self.timeout_seconds = timeout_seconds
        
        logger.info(
            f"Initialized resilient LLM client",
            extra={
                "primary": primary_provider,
                "fallbacks": fallback_providers,
                "timeout": timeout_seconds
            }
        )
    
    @track_latency("llm.completion")
    async def completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get LLM completion with resilience patterns.
        
        Implements:
        1. Retry with exponential backoff
        2. Circuit breaker per provider
        3. Automatic fallback to backup providers
        4. Request timeout
        5. Metrics tracking
        
        Args:
            messages: Chat messages
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            **kwargs: Additional parameters
        
        Returns:
            Dict[str, Any]: LLM response with metadata
        
        Raises:
            LLMServiceError: If all providers fail
            TimeoutError: If request exceeds timeout
        """
        providers = [self.primary_provider] + self.fallback_providers
        last_error = None
        
        for provider_idx, provider in enumerate(providers):
            try:
                is_fallback = provider_idx > 0
                
                if is_fallback:
                    logger.warning(
                        f"Falling back to provider: {provider}",
                        extra={"provider": provider, "attempt": provider_idx + 1}
                    )
                    metrics.increment("llm.fallback", tags={"provider": provider})
                
                # Get circuit breaker for this provider
                breaker = get_circuit_breaker(provider)
                
                # Execute with circuit breaker and timeout
                result = await with_timeout(
                    breaker.call_async(
                        self._call_provider,
                        provider=provider,
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs
                    ),
                    timeout_seconds=self.timeout_seconds,
                    operation_name=f"llm.{provider}"
                )
                
                # Track success
                metrics.increment("llm.completion.success", tags={"provider": provider})
                
                # Add metadata
                result["_metadata"] = {
                    "provider": provider,
                    "fallback_used": is_fallback,
                    "attempt": provider_idx + 1,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                return result
                
            except (LLMRateLimitError, LLMServiceError) as e:
                last_error = e
                logger.warning(
                    f"Provider {provider} failed: {str(e)}",
                    extra={
                        "provider": provider,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                )
                metrics.increment(
                    "llm.completion.failure",
                    tags={"provider": provider, "error": type(e).__name__}
                )
                # Try next provider
                continue
                
            except Exception as e:
                last_error = e
                logger.error(
                    f"Unexpected error with provider {provider}: {str(e)}",
                    extra={"provider": provider, "error": str(e)},
                    exc_info=True
                )
                # Try next provider
                continue
        
        # All providers failed
        logger.error(
            "All LLM providers failed",
            extra={
                "providers": providers,
                "last_error": str(last_error)
            }
        )
        metrics.increment("llm.completion.all_failed")
        
        raise LLMServiceError(
            f"All LLM providers failed. Last error: {str(last_error)}"
        )
    
    @retry_with_backoff(RetryConfig(
        max_attempts=3,
        initial_delay=1.0,
        max_delay=10.0,
        retryable_exceptions=(LLMRateLimitError, LLMServiceError)
    ))
    async def _call_provider(
        self,
        provider: str,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make actual API call to LLM provider with retry logic.
        
        This method is wrapped with retry_with_backoff decorator
        to handle transient failures automatically.
        """
        # Import here to avoid circular dependencies
        from luki_agent.llm_backends import get_backend
        
        try:
            backend = get_backend(provider)
            
            response = await backend.complete(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Classify errors for appropriate handling
            if "rate limit" in error_msg or "429" in error_msg:
                raise LLMRateLimitError(f"Rate limit exceeded: {str(e)}")
            elif "503" in error_msg or "unavailable" in error_msg:
                raise LLMServiceError(f"Service unavailable: {str(e)}")
            else:
                # Re-raise as generic service error for retry
                raise LLMServiceError(f"LLM call failed: {str(e)}")
    
    def get_circuit_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all circuit breakers.
        
        Returns:
            Dict mapping provider names to circuit breaker states
        """
        return {
            provider: {
                "state": breaker.state,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time.isoformat() 
                    if breaker.last_failure_time else None
            }
            for provider, breaker in _circuit_breakers.items()
        }
