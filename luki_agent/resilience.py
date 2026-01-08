"""
Resilience utilities for LUKi Core Agent
Provides retry logic, circuit breakers, and fault tolerance patterns
"""

import asyncio
import logging
from typing import Callable, Any, Optional, Type, Tuple
from functools import wraps
import time
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior"""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff"""
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            import random
            delay = delay * (0.5 + random.random())
        
        return delay


def retry_with_backoff(config: Optional[RetryConfig] = None):
    """
    Decorator to retry function with exponential backoff
    
    Usage:
        @retry_with_backoff(RetryConfig(max_attempts=3))
        async def fetch_data():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = await func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(
                            f"Retry succeeded for {func.__name__} on attempt {attempt + 1}",
                            extra={"function": func.__name__, "attempt": attempt + 1}
                        )
                    
                    return result
                    
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}, "
                            f"retrying in {delay:.2f}s: {str(e)}",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_attempts": config.max_attempts,
                                "delay_seconds": delay,
                                "error": str(e)
                            }
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed for {func.__name__}",
                            extra={
                                "function": func.__name__,
                                "max_attempts": config.max_attempts,
                                "error": str(e)
                            }
                        )
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        logger.info(
                            f"Retry succeeded for {func.__name__} on attempt {attempt + 1}",
                            extra={"function": func.__name__, "attempt": attempt + 1}
                        )
                    
                    return result
                    
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts - 1:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"Attempt {attempt + 1}/{config.max_attempts} failed for {func.__name__}, "
                            f"retrying in {delay:.2f}s: {str(e)}",
                            extra={
                                "function": func.__name__,
                                "attempt": attempt + 1,
                                "max_attempts": config.max_attempts,
                                "delay_seconds": delay,
                                "error": str(e)
                            }
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {config.max_attempts} attempts failed for {func.__name__}",
                            extra={
                                "function": func.__name__,
                                "max_attempts": config.max_attempts,
                                "error": str(e)
                            }
                        )
            
            raise last_exception
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    Prevents cascading failures by stopping requests to failing services
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                logger.info(f"Circuit breaker entering half-open state for {func.__name__}")
            else:
                raise Exception(
                    f"Circuit breaker is OPEN for {func.__name__}. "
                    f"Service unavailable until {self.last_failure_time + timedelta(seconds=self.recovery_timeout)}"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection"""
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
                logger.info(f"Circuit breaker entering half-open state for {func.__name__}")
            else:
                raise Exception(
                    f"Circuit breaker is OPEN for {func.__name__}. "
                    f"Service unavailable until {self.last_failure_time + timedelta(seconds=self.recovery_timeout)}"
                )
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return datetime.utcnow() >= self.last_failure_time + timedelta(seconds=self.recovery_timeout)
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == "half_open":
            logger.info("Circuit breaker recovered, closing circuit")
        self.failure_count = 0
        self.state = "closed"
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error(
                f"Circuit breaker OPENED after {self.failure_count} failures. "
                f"Will attempt recovery in {self.recovery_timeout}s"
            )


class TimeoutError(Exception):
    """Raised when operation exceeds timeout"""
    pass


async def with_timeout(coro, timeout_seconds: float, operation_name: str = "operation"):
    """
    Execute coroutine with timeout
    
    Args:
        coro: Coroutine to execute
        timeout_seconds: Maximum time to wait
        operation_name: Name for logging
    
    Raises:
        TimeoutError: If operation exceeds timeout
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logger.error(
            f"{operation_name} exceeded timeout of {timeout_seconds}s",
            extra={"operation": operation_name, "timeout_seconds": timeout_seconds}
        )
        raise TimeoutError(f"{operation_name} timed out after {timeout_seconds}s")
