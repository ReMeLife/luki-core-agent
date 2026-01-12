"""
Tool execution monitoring for LUKi Core Agent
Tracks tool call metrics, success rates, and performance patterns
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class ToolCallStatus(str, Enum):
    """Status of tool execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"


@dataclass
class ToolCallMetrics:
    """Metrics for individual tool"""
    tool_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    validation_errors: int = 0
    total_latency_seconds: float = 0.0
    min_latency_seconds: Optional[float] = None
    max_latency_seconds: Optional[float] = None
    latency_samples: List[float] = field(default_factory=list)
    error_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_called: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_calls == 0:
            return 0.0
        return (self.successful_calls / self.total_calls) * 100.0
    
    @property
    def average_latency_seconds(self) -> float:
        """Calculate average latency"""
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_seconds / self.total_calls
    
    @property
    def p95_latency_seconds(self) -> Optional[float]:
        """Calculate 95th percentile latency"""
        if not self.latency_samples:
            return None
        sorted_latencies = sorted(self.latency_samples)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    @property
    def p99_latency_seconds(self) -> Optional[float]:
        """Calculate 99th percentile latency"""
        if not self.latency_samples:
            return None
        sorted_latencies = sorted(self.latency_samples)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            "tool_name": self.tool_name,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "timeout_calls": self.timeout_calls,
            "validation_errors": self.validation_errors,
            "success_rate_percent": round(self.success_rate, 2),
            "average_latency_seconds": round(self.average_latency_seconds, 3),
            "min_latency_seconds": round(self.min_latency_seconds, 3) if self.min_latency_seconds else None,
            "max_latency_seconds": round(self.max_latency_seconds, 3) if self.max_latency_seconds else None,
            "p95_latency_seconds": round(self.p95_latency_seconds, 3) if self.p95_latency_seconds else None,
            "p99_latency_seconds": round(self.p99_latency_seconds, 3) if self.p99_latency_seconds else None,
            "top_errors": dict(sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            "last_called": self.last_called.isoformat() if self.last_called else None,
            "last_success": self.last_success.isoformat() if self.last_success else None,
            "last_failure": self.last_failure.isoformat() if self.last_failure else None
        }


class ToolMonitor:
    """Monitor and track tool execution metrics"""
    
    def __init__(self, max_samples_per_tool: int = 1000):
        """
        Initialize tool monitor
        
        Args:
            max_samples_per_tool: Maximum latency samples to keep per tool
        """
        self._lock = threading.Lock()
        self._metrics: Dict[str, ToolCallMetrics] = {}
        self._max_samples = max_samples_per_tool
        self._start_time = datetime.utcnow()
    
    def record_tool_call(
        self,
        tool_name: str,
        status: ToolCallStatus,
        latency_seconds: float,
        error: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """
        Record a tool call execution
        
        Args:
            tool_name: Name of the tool
            status: Execution status
            latency_seconds: Execution time
            error: Error message if failed
            parameters: Tool parameters (for logging)
        """
        with self._lock:
            # Get or create metrics for this tool
            if tool_name not in self._metrics:
                self._metrics[tool_name] = ToolCallMetrics(tool_name=tool_name)
            
            metrics = self._metrics[tool_name]
            now = datetime.utcnow()
            
            # Update call counts
            metrics.total_calls += 1
            metrics.last_called = now
            
            # Update status-specific counts
            if status == ToolCallStatus.SUCCESS:
                metrics.successful_calls += 1
                metrics.last_success = now
            elif status == ToolCallStatus.FAILURE:
                metrics.failed_calls += 1
                metrics.last_failure = now
                if error:
                    metrics.error_counts[error] += 1
            elif status == ToolCallStatus.TIMEOUT:
                metrics.timeout_calls += 1
                metrics.last_failure = now
                metrics.error_counts["timeout"] += 1
            elif status == ToolCallStatus.VALIDATION_ERROR:
                metrics.validation_errors += 1
                metrics.last_failure = now
                if error:
                    metrics.error_counts[error] += 1
            
            # Update latency metrics
            metrics.total_latency_seconds += latency_seconds
            
            if metrics.min_latency_seconds is None or latency_seconds < metrics.min_latency_seconds:
                metrics.min_latency_seconds = latency_seconds
            
            if metrics.max_latency_seconds is None or latency_seconds > metrics.max_latency_seconds:
                metrics.max_latency_seconds = latency_seconds
            
            # Store latency sample (keep last N samples)
            metrics.latency_samples.append(latency_seconds)
            if len(metrics.latency_samples) > self._max_samples:
                metrics.latency_samples = metrics.latency_samples[-self._max_samples:]
            
            # Log the call
            log_level = logging.INFO if status == ToolCallStatus.SUCCESS else logging.WARNING
            logger.log(
                log_level,
                f"Tool call: {tool_name}",
                extra={
                    "tool_name": tool_name,
                    "status": status.value,
                    "latency_seconds": round(latency_seconds, 3),
                    "error": error,
                    "success_rate": round(metrics.success_rate, 2),
                    "total_calls": metrics.total_calls
                }
            )
            
            # Alert on concerning patterns
            self._check_alerts(tool_name, metrics)
    
    def _check_alerts(self, tool_name: str, metrics: ToolCallMetrics):
        """Check for concerning patterns and log alerts"""
        
        # Alert on low success rate (if enough calls)
        if metrics.total_calls >= 10 and metrics.success_rate < 50.0:
            logger.error(
                f"Tool {tool_name} has low success rate",
                extra={
                    "tool_name": tool_name,
                    "success_rate": round(metrics.success_rate, 2),
                    "total_calls": metrics.total_calls,
                    "failed_calls": metrics.failed_calls
                }
            )
        
        # Alert on high latency
        if metrics.average_latency_seconds > 10.0:
            logger.warning(
                f"Tool {tool_name} has high average latency",
                extra={
                    "tool_name": tool_name,
                    "average_latency_seconds": round(metrics.average_latency_seconds, 2),
                    "max_latency_seconds": round(metrics.max_latency_seconds, 2)
                }
            )
        
        # Alert on repeated errors
        if metrics.error_counts:
            most_common_error = max(metrics.error_counts.items(), key=lambda x: x[1])
            if most_common_error[1] >= 5:
                logger.warning(
                    f"Tool {tool_name} has repeated errors",
                    extra={
                        "tool_name": tool_name,
                        "error": most_common_error[0],
                        "count": most_common_error[1],
                        "total_calls": metrics.total_calls
                    }
                )
    
    def get_tool_metrics(self, tool_name: str) -> Optional[ToolCallMetrics]:
        """Get metrics for specific tool"""
        with self._lock:
            return self._metrics.get(tool_name)
    
    def get_all_metrics(self) -> Dict[str, ToolCallMetrics]:
        """Get metrics for all tools"""
        with self._lock:
            return self._metrics.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tool metrics"""
        with self._lock:
            total_calls = sum(m.total_calls for m in self._metrics.values())
            total_successes = sum(m.successful_calls for m in self._metrics.values())
            total_failures = sum(m.failed_calls for m in self._metrics.values())
            
            overall_success_rate = 0.0
            if total_calls > 0:
                overall_success_rate = (total_successes / total_calls) * 100.0
            
            # Get top tools by usage
            top_tools = sorted(
                self._metrics.values(),
                key=lambda m: m.total_calls,
                reverse=True
            )[:10]
            
            # Get problematic tools (low success rate)
            problematic_tools = [
                m for m in self._metrics.values()
                if m.total_calls >= 5 and m.success_rate < 80.0
            ]
            problematic_tools.sort(key=lambda m: m.success_rate)
            
            uptime_seconds = (datetime.utcnow() - self._start_time).total_seconds()
            
            return {
                "uptime_seconds": round(uptime_seconds, 2),
                "total_tools_registered": len(self._metrics),
                "total_calls": total_calls,
                "total_successes": total_successes,
                "total_failures": total_failures,
                "overall_success_rate_percent": round(overall_success_rate, 2),
                "top_tools": [m.to_dict() for m in top_tools],
                "problematic_tools": [m.to_dict() for m in problematic_tools]
            }
    
    def reset_metrics(self, tool_name: Optional[str] = None):
        """
        Reset metrics for a specific tool or all tools
        
        Args:
            tool_name: Tool name to reset, or None for all tools
        """
        with self._lock:
            if tool_name:
                if tool_name in self._metrics:
                    del self._metrics[tool_name]
                    logger.info(f"Reset metrics for tool: {tool_name}")
            else:
                self._metrics.clear()
                self._start_time = datetime.utcnow()
                logger.info("Reset all tool metrics")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of tool ecosystem
        
        Returns:
            Dictionary with health indicators
        """
        with self._lock:
            summary = self.get_summary()
            
            # Determine health status
            if summary["total_calls"] == 0:
                status = "unknown"
            elif summary["overall_success_rate_percent"] >= 95.0:
                status = "healthy"
            elif summary["overall_success_rate_percent"] >= 80.0:
                status = "degraded"
            else:
                status = "unhealthy"
            
            return {
                "status": status,
                "overall_success_rate": summary["overall_success_rate_percent"],
                "total_calls": summary["total_calls"],
                "problematic_tool_count": len(summary["problematic_tools"]),
                "problematic_tools": [t["tool_name"] for t in summary["problematic_tools"][:5]]
            }


# Global tool monitor instance
_tool_monitor: Optional[ToolMonitor] = None


def get_tool_monitor() -> ToolMonitor:
    """Get the global tool monitor instance"""
    global _tool_monitor
    if _tool_monitor is None:
        _tool_monitor = ToolMonitor()
    return _tool_monitor


def track_tool_call(tool_name: str):
    """Decorator to automatically track tool calls"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            status = ToolCallStatus.SUCCESS
            error = None
            
            try:
                result = await func(*args, **kwargs)
                return result
            except TimeoutError as e:
                status = ToolCallStatus.TIMEOUT
                error = str(e)
                raise
            except ValueError as e:
                status = ToolCallStatus.VALIDATION_ERROR
                error = str(e)
                raise
            except Exception as e:
                status = ToolCallStatus.FAILURE
                error = type(e).__name__
                raise
            finally:
                latency = time.time() - start_time
                get_tool_monitor().record_tool_call(
                    tool_name=tool_name,
                    status=status,
                    latency_seconds=latency,
                    error=error
                )
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            status = ToolCallStatus.SUCCESS
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except TimeoutError as e:
                status = ToolCallStatus.TIMEOUT
                error = str(e)
                raise
            except ValueError as e:
                status = ToolCallStatus.VALIDATION_ERROR
                error = str(e)
                raise
            except Exception as e:
                status = ToolCallStatus.FAILURE
                error = type(e).__name__
                raise
            finally:
                latency = time.time() - start_time
                get_tool_monitor().record_tool_call(
                    tool_name=tool_name,
                    status=status,
                    latency_seconds=latency,
                    error=error
                )
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator
