#!/usr/bin/env python3
"""
Standalone FastAPI application for Railway deployment.
This bypasses ALL luki_agent imports to avoid dependency issues.
"""

import os
import logging
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, AsyncGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LUKi Core Agent - Standalone",
    description="Minimal standalone API for Railway deployment testing",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    message: str
    user_id: str = "anonymous"

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "LUKi Core Agent",
        "status": "running",
        "mode": "standalone",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy",
        "service": "luki-core-agent",
        "mode": "standalone",
        "port": os.getenv("PORT", "8080")
    }

@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """Simple chat endpoint with fallback response"""
    logger.info(f"Received chat request from {request.user_id}: {request.message}")
    
    # Simple fallback response
    response_text = f"Hello! I'm LUKi running in standalone mode. You said: '{request.message}'. Full functionality will be available once all services are deployed."
    
    return ChatResponse(
        response=response_text,
        status="success"
    )


@app.post("/v1/chat")
async def chat_v1(request: ChatRequest) -> ChatResponse:
    """Compatibility alias for /v1/chat expected by the API Gateway."""
    return await chat(request)


@app.post("/v1/chat/stream")
async def chat_stream_v1(request: ChatRequest):
    """Minimal streaming alias for /v1/chat/stream for standalone mode."""

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            result = await chat(request)
            payload = {"token": result.response}
            yield f"data: {json.dumps(payload)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            logger.error(f"Streaming chat error (standalone): {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/status")
async def status():
    """Service status endpoint"""
    return {
        "service": "luki-core-agent",
        "status": "running",
        "mode": "standalone",
        "environment": "railway",
        "port": os.getenv("PORT", "8080"),
        "dependencies": "minimal"
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting standalone LUKi API on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
