#!/usr/bin/env python3
"""
Standalone FastAPI application for Railway deployment.
This bypasses ALL luki_agent imports to avoid dependency issues.
"""

import os
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any

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
