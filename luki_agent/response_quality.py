"""
Response Quality Tracker for LUKi Core Agent

Lightweight signal collection for the Phase F RLHF-lite pipeline.
Tracks per-response quality indicators that can later feed into prompt
evaluation, golden-prompt gating, and the thumbs-up/down feedback loop
described in the progression blueprint.

Signals tracked:
- Safety filter trigger rate (how often the safety chain intervenes)
- Retrieval hit rate (fraction of responses with relevant memory context)
- Response latency distribution by request type
- Tool call success/failure rates
- Token budget utilisation efficiency

All data is kept in-memory with bounded storage (rolling window) and
exposed via a ``get_quality_report()`` method for the /metrics endpoint.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# Maximum number of signal records to retain (rolling window)
_MAX_SIGNAL_WINDOW = 2000


@dataclass
class ResponseSignal:
    """A single response quality observation."""

    timestamp: float
    user_id: str
    request_type: str  # "chat", "tool_call", "streaming"
    latency_seconds: float
    had_retrieval_context: bool
    safety_filtered: bool
    tool_calls_attempted: int = 0
    tool_calls_succeeded: int = 0
    tokens_used: int = 0
    token_budget: int = 0


class ResponseQualityTracker:
    """Collects and aggregates response quality signals.

    Thread-safe.  Designed to be called from the main chat handler after
    each response is produced.
    """

    def __init__(self, window_size: int = _MAX_SIGNAL_WINDOW) -> None:
        self._lock = threading.Lock()
        self._signals: Deque[ResponseSignal] = deque(maxlen=window_size)
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        user_id: str,
        request_type: str,
        latency_seconds: float,
        had_retrieval_context: bool = False,
        safety_filtered: bool = False,
        tool_calls_attempted: int = 0,
        tool_calls_succeeded: int = 0,
        tokens_used: int = 0,
        token_budget: int = 0,
    ) -> None:
        """Record a response quality signal after completing a request."""
        signal = ResponseSignal(
            timestamp=time.time(),
            user_id=user_id,
            request_type=request_type,
            latency_seconds=latency_seconds,
            had_retrieval_context=had_retrieval_context,
            safety_filtered=safety_filtered,
            tool_calls_attempted=tool_calls_attempted,
            tool_calls_succeeded=tool_calls_succeeded,
            tokens_used=tokens_used,
            token_budget=token_budget,
        )
        with self._lock:
            self._signals.append(signal)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_quality_report(self) -> Dict[str, Any]:
        """Return an aggregated quality report over the current window."""
        with self._lock:
            signals = list(self._signals)

        if not signals:
            return {
                "total_responses": 0,
                "window_size": 0,
                "uptime_seconds": round(time.monotonic() - self._start_time, 1),
            }

        total = len(signals)
        safety_count = sum(1 for s in signals if s.safety_filtered)
        retrieval_count = sum(1 for s in signals if s.had_retrieval_context)
        latencies = sorted(s.latency_seconds for s in signals)

        # Tool call aggregation
        tool_attempted = sum(s.tool_calls_attempted for s in signals)
        tool_succeeded = sum(s.tool_calls_succeeded for s in signals)

        # Token efficiency
        signals_with_budget = [s for s in signals if s.token_budget > 0]
        avg_token_utilisation = 0.0
        if signals_with_budget:
            avg_token_utilisation = sum(
                s.tokens_used / s.token_budget for s in signals_with_budget
            ) / len(signals_with_budget)

        # Latency by request type
        latency_by_type: Dict[str, List[float]] = {}
        for s in signals:
            latency_by_type.setdefault(s.request_type, []).append(s.latency_seconds)

        latency_report: Dict[str, Dict[str, float]] = {}
        for rtype, lats in latency_by_type.items():
            sorted_lats = sorted(lats)
            latency_report[rtype] = {
                "count": len(sorted_lats),
                "mean_s": round(sum(sorted_lats) / len(sorted_lats), 3),
                "p50_s": round(sorted_lats[len(sorted_lats) // 2], 3),
                "p95_s": round(sorted_lats[int(len(sorted_lats) * 0.95)], 3),
                "max_s": round(sorted_lats[-1], 3),
            }

        return {
            "total_responses": total,
            "safety_filter_rate_pct": round(safety_count / total * 100, 2),
            "retrieval_hit_rate_pct": round(retrieval_count / total * 100, 2),
            "tool_call_success_rate_pct": (
                round(tool_succeeded / tool_attempted * 100, 2) if tool_attempted else None
            ),
            "avg_token_utilisation_pct": round(avg_token_utilisation * 100, 2),
            "latency": {
                "overall": {
                    "mean_s": round(sum(latencies) / total, 3),
                    "p50_s": round(latencies[total // 2], 3),
                    "p95_s": round(latencies[int(total * 0.95)], 3),
                    "max_s": round(latencies[-1], 3),
                },
                "by_type": latency_report,
            },
            "window_size": total,
            "uptime_seconds": round(time.monotonic() - self._start_time, 1),
        }

    def reset(self) -> None:
        """Clear all signals (useful for testing)."""
        with self._lock:
            self._signals.clear()
            self._start_time = time.monotonic()


# ---------------------------------------------------------------------------
# Global instance
# ---------------------------------------------------------------------------
_tracker: Optional[ResponseQualityTracker] = None


def get_response_quality_tracker() -> ResponseQualityTracker:
    """Return the global response quality tracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = ResponseQualityTracker()
    return _tracker
