"""
Observability module for LUKi Core Agent
Provides structured metrics, tracing, and performance monitoring
"""

import time
import logging
from typing import Dict, Any, Optional, Callable
from functools import wraps
from datetime import datetime
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Thread-safe metrics collector for agent operations"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._histograms: Dict[str, list] = defaultdict(list)
        self._gauges: Dict[str, float] = {}
        self._last_reset = datetime.utcnow()
    
    def increment(self, metric_name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        with self._lock:
            key = self._build_key(metric_name, tags)
            self._counters[key] += value
    
    def record_histogram(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Record a histogram value (e.g., latency)"""
        with self._lock:
            key = self._build_key(metric_name, tags)
            self._histograms[key].append(value)
            
            # Keep only last 1000 values to prevent memory bloat
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
    
    def set_gauge(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric (e.g., active connections)"""
        with self._lock:
            key = self._build_key(metric_name, tags)
            self._gauges[key] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics"""
        with self._lock:
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": (datetime.utcnow() - self._last_reset).total_seconds(),
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {}
            }
            
            # Calculate histogram statistics
            for key, values in self._histograms.items():
                if values:
                    sorted_values = sorted(values)
                    metrics["histograms"][key] = {
                        "count": len(values),
                        "min": sorted_values[0],
                        "max": sorted_values[-1],
                        "mean": sum(values) / len(values),
                        "p50": sorted_values[len(values) // 2],
                        "p95": sorted_values[int(len(values) * 0.95)],
                        "p99": sorted_values[int(len(values) * 0.99)]
                    }
            
            return metrics
    
    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()
            self._gauges.clear()
            self._last_reset = datetime.utcnow()
    
    @staticmethod
    def _build_key(metric_name: str, tags: Optional[Dict[str, str]]) -> str:
        """Build metric key with tags"""
        if not tags:
            return metric_name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{metric_name}{{{tag_str}}}"


# Global metrics collector instance
metrics = MetricsCollector()


def track_latency(operation_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to track operation latency"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                error = type(e).__name__
                raise
            finally:
                latency = time.time() - start_time
                operation_tags = {**(tags or {}), "operation": operation_name}
                if error:
                    operation_tags["error"] = error
                    metrics.increment(f"{operation_name}.errors", tags=operation_tags)
                
                metrics.record_histogram(f"{operation_name}.latency_seconds", latency, tags=operation_tags)
                metrics.increment(f"{operation_name}.calls", tags=operation_tags)
                
                logger.debug(f"{operation_name} completed in {latency:.3f}s", extra={
                    "operation": operation_name,
                    "latency_seconds": latency,
                    "error": error
                })
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            error = None
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = type(e).__name__
                raise
            finally:
                latency = time.time() - start_time
                operation_tags = {**(tags or {}), "operation": operation_name}
                if error:
                    operation_tags["error"] = error
                    metrics.increment(f"{operation_name}.errors", tags=operation_tags)
                
                metrics.record_histogram(f"{operation_name}.latency_seconds", latency, tags=operation_tags)
                metrics.increment(f"{operation_name}.calls", tags=operation_tags)
                
                logger.debug(f"{operation_name} completed in {latency:.3f}s", extra={
                    "operation": operation_name,
                    "latency_seconds": latency,
                    "error": error
                })
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class PerformanceProfiler:
    """Context manager for profiling code blocks"""
    
    def __init__(self, operation_name: str, tags: Optional[Dict[str, str]] = None):
        self.operation_name = operation_name
        self.tags = tags or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        latency = time.time() - self.start_time
        operation_tags = {**self.tags, "operation": self.operation_name}
        
        if exc_type:
            operation_tags["error"] = exc_type.__name__
            metrics.increment(f"{self.operation_name}.errors", tags=operation_tags)
        
        metrics.record_histogram(f"{self.operation_name}.latency_seconds", latency, tags=operation_tags)
        metrics.increment(f"{self.operation_name}.calls", tags=operation_tags)


def log_structured(level: str, message: str, **kwargs):
    """Log with structured context"""
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message, extra=kwargs)
