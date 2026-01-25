"""
Metrics API for LUKi Core Agent

Exposes agent metrics via HTTP endpoint for monitoring and debugging.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException
from luki_agent.observability import metrics

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/metrics")
async def get_agent_metrics() -> Dict[str, Any]:
    """
    Get comprehensive agent metrics.
    
    Returns metrics including:
    - Request counts and latencies
    - Tool execution statistics  
    - LLM call metrics
    - Memory retrieval statistics
    - Error rates
    
    Returns:
        Dict[str, Any]: Comprehensive metrics dictionary
    """
    try:
        return metrics.get_metrics()
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")


@router.get("/metrics/summary")
async def get_metrics_summary() -> Dict[str, Any]:
    """
    Get high-level metrics summary.
    
    Provides aggregated statistics suitable for dashboards:
    - Total requests
    - Average latency
    - Error rate
    - Most used tools
    
    Returns:
        Dict[str, Any]: Metrics summary
    """
    try:
        all_metrics = metrics.get_metrics()
        
        # Calculate summary statistics
        counters = all_metrics.get("counters", {})
        histograms = all_metrics.get("histograms", {})
        
        # Extract key metrics
        total_requests = sum(v for k, v in counters.items() if "calls" in k)
        total_errors = sum(v for k, v in counters.items() if "errors" in k)
        
        # Calculate average latencies
        latencies = [
            v.get("mean", 0) 
            for k, v in histograms.items() 
            if "latency" in k
        ]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        
        # Get most called operations
        call_counts = {
            k.split(".calls")[0]: v 
            for k, v in counters.items() 
            if ".calls" in k
        }
        top_operations = sorted(
            call_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0,
            "average_latency_seconds": round(avg_latency, 3),
            "uptime_seconds": all_metrics.get("uptime_seconds", 0),
            "top_operations": dict(top_operations)
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics summary")


@router.post("/metrics/reset")
async def reset_metrics() -> Dict[str, str]:
    """
    Reset all collected metrics.
    
    Use with caution - clears all historical metric data.
    Intended for testing and development.
    
    Returns:
        Dict[str, str]: Confirmation message
    """
    try:
        metrics.reset()
        logger.info("Metrics reset successfully")
        return {"status": "success", "message": "Metrics reset"}
    except Exception as e:
        logger.error(f"Failed to reset metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset metrics")


@router.get("/health/metrics")
async def metrics_health() -> Dict[str, Any]:
    """
    Health check for metrics subsystem.
    
    Returns:
        Dict[str, Any]: Health status
    """
    return {
        "status": "healthy",
        "metrics_collection": "operational",
        "timestamp": metrics.get_metrics().get("timestamp")
    }
